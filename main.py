import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm.notebook import tqdm
import random

base_dir = "data/skin_moles"

benign_dir = os.path.join(base_dir, "benign")
melanoma_dir = os.path.join(base_dir, "melanoma")

# Liczba obrazów w każdej kategorii
benign_files = os.listdir(benign_dir) if os.path.exists(benign_dir) else []
melanoma_files = os.listdir(melanoma_dir) if os.path.exists(melanoma_dir) else []

print(f"Liczba obrazów znamion łagodnych (benign): {len(benign_files)}")
print(f"Liczba obrazów czerniaka (melanoma): {len(melanoma_files)}")
print(f"Łączna liczba obrazów: {len(benign_files) + len(melanoma_files)}")

# Balans klas
total = len(benign_files) + len(melanoma_files)
print(f"Procent obrazów benign: {len(benign_files)/total*100:.2f}%")
print(f"Procent obrazów melanoma: {len(melanoma_files)/total*100:.2f}%")

# Wizualizujemy przykładowe obrazy z obu kategorii
plt.figure(figsize=(12, 6))

for i, img_name in enumerate(random.sample(benign_files, min(3, len(benign_files)))):
    img_path = os.path.join(benign_dir, img_name)
    img = Image.open(img_path)
    plt.subplot(2, 3, i+1)
    plt.imshow(img)
    plt.title(f"Benign: {img_name}")
    plt.axis('off')

for i, img_name in enumerate(random.sample(melanoma_files, min(3, len(melanoma_files)))):
    img_path = os.path.join(melanoma_dir, img_name)
    img = Image.open(img_path)
    plt.subplot(2, 3, i+4)
    plt.imshow(img)
    plt.title(f"Melanoma: {img_name}")
    plt.axis('off')

plt.tight_layout()
plt.show()

# Rozmiary obrazów
image_sizes = []

for img_dir, category in [(benign_dir, "benign"), (melanoma_dir, "melanoma")]:
    img_files = os.listdir(img_dir)
    for img_name in random.sample(img_files, min(10, len(img_files))):
        img_path = os.path.join(img_dir, img_name)
        img = Image.open(img_path)
        width, height = img.size
        image_sizes.append((width, height, category))

sizes_df = pd.DataFrame(image_sizes, columns=["width", "height", "category"])
print("\nStatystyki rozmiarów obrazów:")
print(sizes_df.describe())

# Typy plików
file_extensions = {}
for category, dir_path in [("benign", benign_dir), ("melanoma", melanoma_dir)]:
    for filename in os.listdir(dir_path):
        ext = os.path.splitext(filename)[1].lower()
        if ext in file_extensions:
            file_extensions[ext] += 1
        else:
            file_extensions[ext] = 1

print("\nTypy plików w zbiorze danych:")
for ext, count in file_extensions.items():
    print(f"{ext}: {count} plików")