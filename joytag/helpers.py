from PIL import Image as ImageModule
from PIL.Image import Image
from joytag.models import VisionModel
import torchvision.transforms.functional as TVF
import torch
from torchvision import transforms

# Most code are borrowed from: https://github.com/gokayfem/ComfyUI_VLM_nodes/nodes/joytag.py#L70

def download_joytag(target_dir: Path | str):
    print(f"Target directory for download: {target_dir}")
    
    path = snapshot_download(
        "fancyfeast/joytag",
        local_dir=target_dir,
        force_download=False, 
        local_files_only=False, 
        local_dir_use_symlinks="auto"  
    )

    return path

def prepare_image(image: Image, target_size: int) -> torch.Tensor:
	# Pad image to square
	image_shape = image.size
	max_dim = max(image_shape)
	pad_left = (max_dim - image_shape[0]) // 2
	pad_top = (max_dim - image_shape[1]) // 2

	padded_image = ImageModule.new('RGB', (max_dim, max_dim), (255, 255, 255))
	padded_image.paste(image, (pad_left, pad_top))

	# Resize image
	if max_dim != target_size:
		padded_image = padded_image.resize((target_size, target_size), ImageModule.BICUBIC)
	
	# Convert to tensor
	image_tensor = TVF.pil_to_tensor(padded_image) / 255.0

	# Normalize
	image_tensor = TVF.normalize(image_tensor, mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

	return image_tensor

def process_tag(tag: string):
	tag = tag.replace("(medium)", "")  # Remove (medium)
	tag = tag.replace("\\", "")  # Remove \
	tag = tag.replace("m/", "")  # Remove m/
	tag = tag.replace("-", "")  # Remove -
	tag = tag.replace("_", " ")  # Replace underscores with spaces
	tag = tag.strip()  # Remove leading and trailing spaces

	return tag

class JoytagInferencer:
    def __init__(self, basepath="/content/models/", device="cuda" if torch.cuda.is_available() else "cpu"):
        model_dir = Path(basepath) / "joytag/"

        if not model_dir.exists():
            download_joytag(target_path)

        self.model_dir = model_dir

        with open(model_dir / "top_tags.txt", "r") as f:
			self.top_tags = [line.strip() for line in f.readlines() if line.strip()]

        self.model = joytag.VisionModel.load_model(self.model_path, device=device)
        self.model.eval()
        self.device = device

	def tags(self, image: PIL.Image, topk_tags: int):
		@torch.no_grad()
		def predict(image: PIL.Image):
			image_tensor = prepare_image(image, model.image_size)
			batch = {
				'image': image_tensor.unsqueeze(0).to(self.device),
			}

			with torch.amp.autocast_mode.autocast(self.device, enabled=True):
				preds = model(batch)
				tag_preds = preds['tags'].sigmoid().cpu()
			
			scores = {top_tags[i]: tag_preds[0][i] for i in range(len(self.top_tags))}
			predicted_tags = [tag for tag, score in scores.items() if score > THRESHOLD]
			tag_string = ', '.join(predicted_tags)

			return tag_string, scores
			
		image = transforms.ToPILImage()(image[0].permute(2, 0, 1))
		_, scores = predict(image)

		# Get the top 50 tag and score pairs
		top_tags_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:tag_number]

		# Extract the tags from the pairs
		top_tags_processed = [process_tag(tag) for tag, _ in top_tags_scores]
		
		top_tags_full = [tag for tag in top_tags_processed if tag]

		# Concatenate the tags with a comma separator
		top_50_tags_string = ', '.join(top_tags_full)
		
		return (top_50_tags_string, )