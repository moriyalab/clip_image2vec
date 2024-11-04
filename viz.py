import clip
import torch
from PIL import Image
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def get_image_features(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    return image_features / image_features.norm(dim=-1, keepdim=True)  # 正規化


# 特徴量の抽出
image_features = []
image_paths = glob.glob("./images/*")
for image_path in image_paths:
    image_features.append(get_image_features(image_path).cpu().numpy().flatten())

# リストを numpy 配列に変換
image_features = np.array(image_features)

# 特徴ベクトルを次元削減 (ここでは t-SNE を使用)
# perplexity を画像の数より小さい値に設定
tsne = TSNE(n_components=2, perplexity=5, random_state=0)
reduced_features = tsne.fit_transform(image_features)

# プロット（ラベルなし）
plt.figure(figsize=(10, 10))
plt.scatter(reduced_features[:, 0], reduced_features[:, 1])
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.title("Image Features Visualization with t-SNE")
plt.show()
