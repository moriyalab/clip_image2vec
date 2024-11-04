import clip
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Detect device: ", device)
model, preprocess = clip.load("ViT-B/32", device=device)
