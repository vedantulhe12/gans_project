import os
import sys
sys.path.append('./encoder4editing')
import os
import argparse
import numpy as np
from PIL import Image
from argparse import Namespace
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
import torch
from torchvision import transforms
from torchvision.transforms.functional import to_tensor, to_pil_image



# AnimeGAN
from model import Generator as AnimeGANGenerator
# Add this at the top of the file (below other imports)
from models.psp import pSp
# e4e Encoder
import sys
sys.path.append('encoder4editing')
from utils.common import tensor2im

def load_e4e_model(checkpoint_path, device, low_mem=False):
    opts = argparse.Namespace(
        checkpoint_path=checkpoint_path,
        device=device,
        low_mem=low_mem,
        encoder_type='Encoder4Editing',
        start_from_latent_avg=True,
        input_nc=3,
        n_styles=18,
        stylegan_size=1024,  # Required for `log_size`
        is_train=False,
        learn_in_w=False,
        output_size=1024,
        id_lambda=0,
        lpips_lambda=0,
        l2_lambda=1,
        w_discriminator_lambda=0,
        use_w_pool=False,
        w_pool_size=50,
        use_ballholder_loss=False,
        optim_type='adam',
        batch_size=1,
        resize_outputs=False
    )

    model = pSp(opts).to(device).eval()
    return model





def apply_emotion_edit(latent, boundary_path, intensity, device):
    boundary = np.load(boundary_path)
    boundary = torch.from_numpy(boundary).float().to(device).view(1, 1, -1)
    return latent + intensity * boundary


def run_interfacegan_edit(image_path, encoder, boundary_path, intensity, device):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        # Encode image to latent space
        generated_images, latents = encoder(img_tensor, return_latents=True)
        latent = latents  # Shape: (1, 18, 512)

        

        # Load and format boundary
        boundary_np = validate_boundary_file(boundary_path, emotion=args.emotion)
        boundary = torch.from_numpy(boundary_np).float().to(device)

# Then extract vector from boundary depending on shape
        boundary_vec = boundary if boundary.ndim == 1 else boundary[0]

        boundary_np = np.load(boundary_path)
        boundary = torch.from_numpy(boundary_np).float().to(device)

        if boundary.ndim == 1:
            boundary_vec = boundary  # Shape: [512]
        elif boundary.ndim == 2:
            boundary_vec = boundary[0]  # Shape: [512]
        elif boundary.ndim == 3 and boundary.shape[1:] == (1, 512):
            boundary_vec = boundary.squeeze()  # Shape: [512]
        else:
            raise ValueError(f"Unexpected boundary shape: {boundary.shape}")

        # Apply boundary to specific latent layers (e.g., 4 to 8)
        latent_edited = latent.clone()
        layers_to_edit = range(4, 9)
        for i in layers_to_edit:
            latent_edited[:, i, :] += intensity * boundary_vec

        # Generate image from edited latent
        generated_output = encoder.decoder([latent_edited], input_is_latent=True)
        generated_img = generated_output[0]  # First item is the image

        # Convert tensor to PIL
        generated_img = (generated_img.clamp(-1, 1) + 1) / 2
        generated_img = generated_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        generated_img = (generated_img * 255).astype('uint8')
        output_pil = Image.fromarray(generated_img)

        # Clean up
        del img_tensor, latent, latent_edited, generated_img
        if device == 'cpu':
            torch.cuda.empty_cache()
            

    return output_pil



def run_animegan(image_pil, checkpoint_path, device):
    # Convert PIL image [0–255] → Tensor [0–1]
    image_tensor = to_tensor(image_pil).unsqueeze(0)  # Shape: (1, 3, H, W)

    # Normalize [0–1] → [-1, 1] as expected by AnimeGAN
    image_tensor = image_tensor * 2 - 1

    model = AnimeGANGenerator().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    with torch.no_grad():
        output_tensor = model(image_tensor.to(device)).cpu().squeeze(0).clamp(-1, 1)
        output_tensor = output_tensor * 0.5 + 0.5  # Back to [0, 1] for viewing
        anime_pil = to_pil_image(output_tensor)
        print(f"[AnimeGAN Input] tensor range: min={image_tensor.min().item()}, max={image_tensor.max().item()}")


    return anime_pil


def validate_boundary_file(boundary_path, emotion, expected_dim=512):
    import numpy as np
    import os

    if not os.path.exists(boundary_path):
        raise FileNotFoundError(f"Boundary file not found for emotion '{emotion}': {boundary_path}")

    boundary = np.load(boundary_path)

    if boundary.ndim == 1:
        if boundary.shape[0] != expected_dim:
            raise ValueError(f"Boundary for '{emotion}' has incorrect size: expected ({expected_dim},), got {boundary.shape}")
    elif boundary.ndim == 2:
        if boundary.shape[1] != expected_dim:
            raise ValueError(f"Boundary for '{emotion}' has incorrect second dimension: expected ({expected_dim},), got {boundary.shape}")
    else:
        raise ValueError(f"Boundary for '{emotion}' has unsupported shape: {boundary.shape}")

    print(f"✅ Boundary file for '{emotion}' is valid: shape {boundary.shape}")
    return boundary


def main(args):
    # ✅ Define device FIRST
    device = torch.device(args.device)

    # Set deterministic algorithms for CPU stability
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load models
    print("Loading e4e...")
    encoder = load_e4e_model(args.e4e_checkpoint, device, args.low_mem)

    print("Loading AnimeGAN...")
    # AnimeGAN model loaded in function

    # Process image
    print("Applying emotion edit...")
    edited_face = run_interfacegan_edit(
        image_path=args.input_image,
        encoder=encoder,
        boundary_path=os.path.join(args.boundary_dir, f"boundary_{args.emotion}.npy"),
        intensity=args.intensity,
        device=device
    )

    print("Generating anime version...")
    anime_output = run_animegan(edited_face, args.animegan_checkpoint, device)

    # Save result
    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.input_image))[0]
    output_filename = f"{base_name}_{args.emotion}_int{args.intensity:.1f}.png"
    output_path = os.path.join(args.output_dir, output_filename)


    anime_output.save(output_path)
    print(f"Anime image saved at: {output_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--low_mem', action='store_true', help='Run in low memory mode (for GPUs <6GB)')
    parser.add_argument('--input_image', type=str, required=True, help='Path to input human face image')
    parser.add_argument('--emotion', type=str, required=True, help='Emotion name (e.g., happy, sad, angry)')
    parser.add_argument('--intensity', type=float, default=1.0, help='Emotion intensity (e.g., 0.0 to 2.0)')
    parser.add_argument('--e4e_checkpoint', type=str, default='encoder4editing/pretrained_models/e4e_ffhq_encode.pt')
    parser.add_argument('--animegan_checkpoint', type=str, default='weights/paprika.pt')
    parser.add_argument('--boundary_dir', type=str, default='boundaries')
    parser.add_argument('--output_dir', type=str, default='samples/results')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    main(args)