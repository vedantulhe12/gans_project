import os


import sys
sys.path.append('./encoder4editing')
import os
import argparse
import numpy as np
from PIL import Image
from argparse import Namespace
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


def load_e4e_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    opts = ckpt['opts']

    # ✅ Convert to Namespace
    if isinstance(opts, dict):
        opts = Namespace(**opts)

    # ✅ Force device to CPU
    opts.device = device

    opts.checkpoint_path = checkpoint_path
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
        # ✅ Get both images and latents
        generated_images, latents = encoder(img_tensor, return_latents=True)
        latent = latents[0]  # Correct latent shape: (1, 18, 512)
        
        latent_edited = apply_emotion_edit(latent, boundary_path, intensity, device)
        generated_img = encoder.decoder(latent_edited, input_is_latent=True)
        output_pil = tensor2im(generated_img[0])
    
    return output_pil


def run_animegan(image_pil, checkpoint_path, device):
    image_tensor = to_tensor(image_pil).unsqueeze(0) * 2 - 1
    model = AnimeGANGenerator().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    with torch.no_grad():
        output_tensor = model(image_tensor.to(device)).cpu().squeeze(0).clip(-1, 1) * 0.5 + 0.5
        anime_pil = to_pil_image(output_tensor)
    return anime_pil


def main(args):
    device = torch.device(args.device)

    # Load models
    print("Loading e4e...")
    encoder = load_e4e_model(args.e4e_checkpoint, device)
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
    output_path = os.path.join(args.output_dir, os.path.basename(args.input_image))
    anime_output.save(output_path)
    print(f"Anime image saved at: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
