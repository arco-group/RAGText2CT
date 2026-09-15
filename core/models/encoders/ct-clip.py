import sys
sys.path.append('/mimer/NOBACKUP/groups/naiss2023-6-336/dmolino/MedCoDi-M')
sys.path.append('transformer_maskgit/transformer_maskgit')

import torch
from torch import nn
import torch.nn.functional as F
from transformer_maskgit.transformer_maskgit import CTViT
# from ct_clip import CTCLIP, TextTransformer
# from CTCLIPTrainer import CTClipTrainer

def l2norm(t):
    return F.normalize(t, dim = -1)

image_encoder = CTViT(
    dim=768,
    codebook_size=8192,
    image_size = 512,
    patch_size = 16,
    temporal_patch_size = 16,
    spatial_depth = 4,
    temporal_depth = 4,
    dim_head = 32,
    heads=8
).to('cuda')
to_visual_latent = nn.Linear(786432, 768, bias = False).to('cuda')

tensor = torch.randn(1, 1, 128, 512, 512).to('cuda')

enc_image = image_encoder(tensor, return_encoded_tokens=True)
#print("This is visual encoding")

h_r, w_r, z_r = enc_image.shape[1], enc_image.shape[2], enc_image.shape[3]
enc_image_send = enc_image
enc_image = torch.mean(enc_image, dim=1)
enc_image = enc_image.view(enc_image.shape[0], -1)
image_embeds = enc_image[:, :] if enc_image.ndim == 3 else enc_image
image_latents = to_visual_latent(image_embeds)

image_latents = l2norm(image_latents)

