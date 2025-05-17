import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm

base_dir = "data/skin_moles"
benign_dir = os.path.join(base_dir, "benign")
melanoma_dir = os.path.join(base_dir, "melanoma")

train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

train_benign_dir = os.path.join(train_dir, "benign")
train_melanoma_dir = os.path.join(train_dir, "melanoma")
test_benign_dir = os.path.join(test_dir, "benign")
test_melanoma_dir = os.path.join(test_dir, "melanoma")

for dir_path in [train_dir, test_dir, train_benign_dir, train_melanoma_dir, test_benign_dir, test_melanoma_dir]:
    os.makedirs(dir_path, exist_ok=True)

# Podział obrazów na zbiory treningowy (70%) i testowy (30%)
def split_data(source_dir, train_dir, test_dir, split_ratio=0.7):
    files = os.listdir(source_dir)
    train_size = int(len(files) * split_ratio)
    
    # Pomieszane dane i podział na zbiory
    np.random.shuffle(files)
    train_files = files[:train_size]
    test_files = files[train_size:]
    
    for file_name in tqdm(train_files, desc=f"Kopiowanie plików treningowych z {os.path.basename(source_dir)}"):
        src = os.path.join(source_dir, file_name)
        dst = os.path.join(train_dir, file_name)
        shutil.copy(src, dst)
    
    for file_name in tqdm(test_files, desc=f"Kopiowanie plików testowych z {os.path.basename(source_dir)}"):
        src = os.path.join(source_dir, file_name)
        dst = os.path.join(test_dir, file_name)
        shutil.copy(src, dst)
    
    return len(train_files), len(test_files)

print("Dzielenie danych na zbiory treningowy i testowy...")
benign_train_count, benign_test_count = split_data(benign_dir, train_benign_dir, test_benign_dir)
melanoma_train_count, melanoma_test_count = split_data(melanoma_dir, train_melanoma_dir, test_melanoma_dir)

print(f"\nLiczba obrazów treningowych (benign): {benign_train_count}")
print(f"Liczba obrazów treningowych (melanoma): {melanoma_train_count}")
print(f"Łącznie obrazów treningowych: {benign_train_count + melanoma_train_count}")

print(f"\nLiczba obrazów testowych (benign): {benign_test_count}")
print(f"Liczba obrazów testowych (melanoma): {melanoma_test_count}")
print(f"Łącznie obrazów testowych: {benign_test_count + melanoma_test_count}")

# Parametry preprocessingu
img_width, img_height = 224, 224
batch_size = 32

# Generator z augmentacją dla danych treningowych
train_datagen = ImageDataGenerator(
    rescale=1./255,             # normalizacja (konieczna)
    rotation_range=20,          # zmniejszony losowy obrót (bardziej ostrożny)
    horizontal_flip=True,       # losowe odbicie w poziomie (bezpieczne)
    vertical_flip=True,         # losowe odbicie w pionie (bezpieczne)
    zoom_range=0.1,             # bardzo delikatne przybliżenie (zmniejszone)
    fill_mode='nearest'         # strategia wypełniania nowych pikseli
)

# Generator bez augmentacji dla danych testowych (tylko normalizacja)
test_datagen = ImageDataGenerator(rescale=1./255)

# Przygotuj generatory
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

# Sprawdź, jak wyglądają augmentowane obrazy
def visualize_augmentation(datagen, image_path, n_samples=5):
    img = plt.imread(image_path)
    img = img.reshape((1,) + img.shape)
    
    plt.figure(figsize=(15, 3))
    plt.subplot(1, n_samples+1, 1)
    plt.imshow(img[0])
    plt.title('Oryginalny obraz')
    plt.axis('off')
    
    i = 2
    for batch in datagen.flow(img, batch_size=1):
        plt.subplot(1, n_samples+1, i)
        plt.imshow(batch[0])
        plt.title(f'Augmentacja {i-1}')
        plt.axis('off')
        i += 1
        if i > n_samples+1:
            break
    
    plt.tight_layout()
    plt.show()

# Wizualizuj augmentację na przykładowym obrazie benign
sample_benign = os.path.join(train_benign_dir, os.listdir(train_benign_dir)[0])
print("\nPrzykłady augmentacji dla znamienia łagodnego (benign):")
visualize_augmentation(train_datagen, sample_benign)

# Wizualizuj augmentację na przykładowym obrazie melanoma
sample_melanoma = os.path.join(train_melanoma_dir, os.listdir(train_melanoma_dir)[0])
print("\nPrzykłady augmentacji dla czerniaka (melanoma):")
visualize_augmentation(train_datagen, sample_melanoma)

print("\nPrzygotowanie danych zakończone!")