import clip
import torch
from PIL import Image
import glob

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def get_image_features(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    return image_features / image_features.norm(dim=-1, keepdim=True)  # 正規化


image_features = []
image_paths = glob.glob("./images/*")
for image_path in image_paths:
    print(image_path)
    image_features.append(get_image_features(image_path))

similarities = []
for index, feature in enumerate(image_features):
    for index_, feature_ in enumerate(image_features):
        if index_ == index:
            continue
        similarities.append((feature @ feature_.T).item())

similarity_average = sum(similarities) /  len(similarities)
print(similarity_average)

# 0.8536337896439236
# 0.9495137247899262