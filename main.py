# main.py (zaktualizowany z opcją ulepszonego CNN)
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_folder_structure():
    """Tworzy strukturę folderów dla wyników projektowych i danych."""
    folders = [
        "data/skin_moles/train/benign",
        "data/skin_moles/train/melanoma",
        "data/skin_moles/test/benign",
        "data/skin_moles/test/melanoma",
        "results",
        "results/random_forest", 
        "results/random_forest/models",
        "results/random_forest/plots",
        "results/cnn",
        "results/cnn/models",
        "results/cnn/plots",
        "results/cnn_improved",      # Nowy folder dla ulepszonego CNN
        "results/cnn_improved/models",
        "results/cnn_improved/plots",
        "results/transfer_learning",
        "results/transfer_learning/models",
        "results/transfer_learning/plots"
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    
    print("Struktura folderów została utworzona.")

def split_data(source_dir, train_dir, test_dir, split_ratio=0.8):
    """Dzieli zbiór danych na treningowy i testowy w podanej proporcji."""
    files = os.listdir(source_dir)
    
    # Jeśli folder docelowy jest już pełny, zakładamy, że dane są już podzielone
    train_files = os.listdir(train_dir)
    test_files = os.listdir(test_dir)
    
    if len(train_files) > 0 and len(test_files) > 0:
        print(f"Foldery {train_dir} i {test_dir} już zawierają dane.")
        print(f"Liczba plików treningowych: {len(train_files)}")
        print(f"Liczba plików testowych: {len(test_files)}")
        return len(train_files), len(test_files)
    
    # Pomieszaj dane i podziel na zbiory
    np.random.seed(42)  # Dla powtarzalności wyników
    np.random.shuffle(files)
    train_size = int(len(files) * split_ratio)
    
    train_files = files[:train_size]
    test_files = files[train_size:]
    
    # Kopiuj pliki do katalogów treningowego i testowego
    for file_name in tqdm(train_files, desc=f"Kopiowanie plików treningowych z {os.path.basename(source_dir)}"):
        src = os.path.join(source_dir, file_name)
        dst = os.path.join(train_dir, file_name)
        shutil.copy(src, dst)
    
    for file_name in tqdm(test_files, desc=f"Kopiowanie plików testowych z {os.path.basename(source_dir)}"):
        src = os.path.join(source_dir, file_name)
        dst = os.path.join(test_dir, file_name)
        shutil.copy(src, dst)
    
    return len(train_files), len(test_files)

def visualize_data_augmentation():
    """Wizualizuje augmentację danych na przykładowych obrazach."""
    # Ścieżki do danych
    base_dir = "data/skin_moles"
    train_dir = os.path.join(base_dir, "train")
    train_benign_dir = os.path.join(train_dir, "benign")
    train_melanoma_dir = os.path.join(train_dir, "melanoma")
    
    # Generator z augmentacją dla danych treningowych
    train_datagen = ImageDataGenerator(
        rescale=1./255,             # normalizacja
        rotation_range=20,          # ostrożny losowy obrót
        horizontal_flip=True,       # losowe odbicie w poziomie
        vertical_flip=True,         # losowe odbicie w pionie
        zoom_range=0.1,             # delikatne przybliżenie
        fill_mode='nearest'         # strategia wypełniania nowych pikseli
    )
    
    # Funkcja do wizualizacji augmentacji
    def visualize_augmentation(datagen, image_path, n_samples=5):
        img = plt.imread(image_path)
        img = img.reshape((1,) + img.shape)  # Reshape dla batch
        
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
    
    # Wybierz przykładowe obrazy
    benign_files = os.listdir(train_benign_dir)
    melanoma_files = os.listdir(train_melanoma_dir)
    
    if len(benign_files) > 0 and len(melanoma_files) > 0:
        sample_benign = os.path.join(train_benign_dir, benign_files[0])
        sample_melanoma = os.path.join(train_melanoma_dir, melanoma_files[0])
        
        print("\nPrzykłady augmentacji dla znamienia łagodnego (benign):")
        visualize_augmentation(train_datagen, sample_benign)
        
        print("\nPrzykłady augmentacji dla czerniaka (melanoma):")
        visualize_augmentation(train_datagen, sample_melanoma)
    else:
        print("Brak plików do wizualizacji augmentacji.")

def prepare_data(split_ratio=0.8):
    """Przygotowuje dane - podział na zbiory treningowy i testowy."""
    # Ścieżki do katalogów
    base_dir = "data/skin_moles"
    benign_dir = os.path.join(base_dir, "benign")
    melanoma_dir = os.path.join(base_dir, "melanoma")

    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")

    train_benign_dir = os.path.join(train_dir, "benign")
    train_melanoma_dir = os.path.join(train_dir, "melanoma")
    test_benign_dir = os.path.join(test_dir, "benign")
    test_melanoma_dir = os.path.join(test_dir, "melanoma")

    # Sprawdź czy katalogi źródłowe istnieją
    if not os.path.exists(benign_dir) or not os.path.exists(melanoma_dir):
        print(f"Błąd: katalogi źródłowe {benign_dir} lub {melanoma_dir} nie istnieją.")
        print("Upewnij się, że struktura folderów jest poprawna i dane są na swoim miejscu.")
        return False

    # Podział danych
    print(f"Dzielenie danych na zbiory treningowy ({split_ratio*100}%) i testowy ({(1-split_ratio)*100}%)...")
    benign_train_count, benign_test_count = split_data(benign_dir, train_benign_dir, test_benign_dir, split_ratio)
    melanoma_train_count, melanoma_test_count = split_data(melanoma_dir, train_melanoma_dir, test_melanoma_dir, split_ratio)

    print(f"\nLiczba obrazów treningowych (benign): {benign_train_count}")
    print(f"Liczba obrazów treningowych (melanoma): {melanoma_train_count}")
    print(f"Łącznie obrazów treningowych: {benign_train_count + melanoma_train_count}")

    print(f"\nLiczba obrazów testowych (benign): {benign_test_count}")
    print(f"Liczba obrazów testowych (melanoma): {melanoma_test_count}")
    print(f"Łącznie obrazów testowych: {benign_test_count + melanoma_test_count}")

    # Wizualizacja przykładów augmentacji danych
    visualize_data_augmentation()
    
    print("\nPrzygotowanie danych zakończone!")
    return True

def run_random_forest():
    """Uruchamia model Random Forest."""
    print("Uruchamianie modelu Random Forest...")
    import rf_model
    print("Model Random Forest zakończył działanie.")

def run_cnn():
    """Uruchamia model CNN."""
    print("Uruchamianie modelu CNN...")
    import cnn_model
    print("Model CNN zakończył działanie.")

def run_cnn_improved():
    """Uruchamia ulepszony model CNN."""
    print("Uruchamianie ulepszonego modelu CNN...")
    import cnn_improved_model
    print("Ulepszony model CNN zakończył działanie.")

def run_transfer_learning():
    """Uruchamia model Transfer Learning."""
    print("Uruchamianie modelu Transfer Learning...")
    import transfer_learning_model
    print("Model Transfer Learning zakończył działanie.")

def main():
    parser = argparse.ArgumentParser(description='Klasyfikacja znamion skórnych: benign vs. melanoma')
    parser.add_argument('--model', type=str, choices=['rf', 'cnn', 'cnn_improved', 'tl', 'all'], 
                        default='all', help='Wybierz model do uruchomienia')
    parser.add_argument('--split-ratio', type=float, default=0.8,
                        help='Proporcja podziału na zbiór treningowy (domyślnie: 0.8)')
    parser.add_argument('--skip-data-prep', action='store_true',
                        help='Pomiń przygotowanie danych (użyj, gdy dane są już podzielone)')
    
    args = parser.parse_args()
    
    # Tworzenie struktury folderów
    create_folder_structure()
    
    # Przygotowanie danych (podział na zbiory treningowy i testowy)
    if not args.skip_data_prep:
        success = prepare_data(split_ratio=args.split_ratio)
        if not success:
            print("Błąd podczas przygotowywania danych. Przerwanie programu.")
            return
    
    # Uruchamianie wybranych modeli
    if args.model == 'rf' or args.model == 'all':
        run_random_forest()
    
    if args.model == 'cnn' or args.model == 'all':
        run_cnn()
    
    if args.model == 'cnn_improved' or args.model == 'all':
        run_cnn_improved()
    
    if args.model == 'tl' or args.model == 'all':
        run_transfer_learning()
    
if __name__ == "__main__":
    main()