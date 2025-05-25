import os
import sys
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
        "results/random_forest_improved",
        "results/random_forest_improved/models",
        "results/random_forest_improved/plots", 
        "results/cnn",
        "results/cnn/models",
        "results/cnn/plots",
        "results/cnn_improved",
        "results/cnn_improved/models",
        "results/cnn_improved/plots",
        "results/cnn_v3",         
        "results/cnn_v3/models",
        "results/cnn_v3/plots",
        "results/cnn_v4",         
        "results/cnn_v4/models",
        "results/cnn_v4/plots",
        "results/transfer_learning",
        "results/transfer_learning/models",
        "results/transfer_learning/plots",
        "results/xgboost",
        "results/xgboost/models",
        "results/xgboost/plots",
        "results/efficientnet",
        "results/efficientnet/models",
        "results/efficientnet/plots",
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    
    print("Struktura folderów została utworzona.")

def split_data(source_dir, train_dir, test_dir, split_ratio=0.8, force_split=False):
    """Dzieli zbiór danych na treningowy i testowy w podanej proporcji.
    
    Args:
        source_dir: Ścieżka do folderu źródłowego z obrazami
        train_dir: Ścieżka do folderu treningowego
        test_dir: Ścieżka do folderu testowego
        split_ratio: Proporcja podziału na zbiór treningowy (domyślnie 0.8)
        force_split: Czy wymusić ponowny podział danych (domyślnie False)
    """
    if not os.path.exists(source_dir):
        print(f"Błąd: folder źródłowy {source_dir} nie istnieje.")
        return 0, 0
    
    files = os.listdir(source_dir)
    
    train_files = os.listdir(train_dir) if os.path.exists(train_dir) else []
    test_files = os.listdir(test_dir) if os.path.exists(test_dir) else []
    
    if len(train_files) > 0 and len(test_files) > 0 and not force_split:
        print(f"Foldery {train_dir} i {test_dir} już zawierają dane.")
        print(f"Liczba plików treningowych: {len(train_files)}")
        print(f"Liczba plików testowych: {len(test_files)}")
        return len(train_files), len(test_files)
    
    # Jeśli wymuszamy podział, usuwamy najpierw stare pliki
    if force_split:
        if os.path.exists(train_dir):
            for file in os.listdir(train_dir):
                os.remove(os.path.join(train_dir, file))
        if os.path.exists(test_dir):
            for file in os.listdir(test_dir):
                os.remove(os.path.join(test_dir, file))
    
    np.random.seed(42)  # Dla powtarzalności wyników
    np.random.shuffle(files)
    train_size = int(len(files) * split_ratio)
    
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

def visualize_data_augmentation():
    """Wizualizuje augmentację danych na przykładowych obrazach."""
    base_dir = "data/skin_moles"
    train_dir = os.path.join(base_dir, "train")
    train_benign_dir = os.path.join(train_dir, "benign")
    train_melanoma_dir = os.path.join(train_dir, "melanoma")
    
    # Generator z augmentacją dla danych treningowych (bazowa wersja)
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

def prepare_data(split_ratio=0.8, force_split=False):
    """Przygotowuje dane - podział na zbiory treningowy i testowy."""
    base_dir = "data/skin_moles"
    benign_dir = os.path.join(base_dir, "benign")
    melanoma_dir = os.path.join(base_dir, "melanoma")

    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")

    train_benign_dir = os.path.join(train_dir, "benign")
    train_melanoma_dir = os.path.join(train_dir, "melanoma")
    test_benign_dir = os.path.join(test_dir, "benign")
    test_melanoma_dir = os.path.join(test_dir, "melanoma")

    if not os.path.exists(benign_dir) or not os.path.exists(melanoma_dir):
        print(f"Błąd: katalogi źródłowe {benign_dir} lub {melanoma_dir} nie istnieją.")
        print("Upewnij się, że struktura folderów jest poprawna i dane są na swoim miejscu.")
        return False

    # Podział danych
    print(f"Dzielenie danych na zbiory treningowy ({split_ratio*100}%) i testowy ({(1-split_ratio)*100}%)...")
    benign_train_count, benign_test_count = split_data(benign_dir, train_benign_dir, test_benign_dir, split_ratio, force_split)
    melanoma_train_count, melanoma_test_count = split_data(melanoma_dir, train_melanoma_dir, test_melanoma_dir, split_ratio, force_split)

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

def run_random_forest_improved():
    """Uruchamia ulepszony model Random Forest."""
    print("Uruchamianie ulepszonego modelu Random Forest...")
    import rf_model_improved
    print("Ulepszony model Random Forest zakończył działanie.")

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

def run_cnn_v3():
    """Uruchamia model CNN v3."""
    print("Uruchamianie modelu CNN v3...")
    import cnn_v3_model
    print("Model CNN v3 zakończył działanie.")

def run_cnn_v4():
    """Uruchamia model CNN v4."""
    print("Uruchamianie modelu CNN v4...")
    import cnn_v4_model
    print("Model CNN v4 zakończył działanie.")

def run_transfer_learning():
    """Uruchamia model Transfer Learning."""
    print("Uruchamianie modelu Transfer Learning...")
    import transfer_learning_model
    print("Model Transfer Learning zakończył działanie.")

def run_xgboost():
    """Uruchamia model XGBoost."""
    print("Uruchamianie modelu XGBoost...")
    import xgboost_model
    print("Model XGBoost zakończył działanie.")

def run_efficientnet():
    """Uruchamia model EfficientNet."""
    print("Uruchamianie modelu EfficientNet...")
    import Projekt_spam.efficientnet_model as efficientnet_model
    print("Model EfficientNet zakończył działanie.")

def run_hybrid():
    """Uruchamia model hybrydowy TL+XGBoost."""
    print("Uruchamianie modelu hybrydowego TL+XGBoost...")
    import hybrid_tl_xgb_model
    print("Model hybrydowy zakończył działanie.")

def main():
    parser = argparse.ArgumentParser(description='Klasyfikacja znamion skórnych: benign vs. melanoma')
    parser.add_argument('--model', type=str, 
                    choices=['rf', 'rf_improved', 'cnn', 'cnn_improved', 'cnn_v3', 'cnn_v4', 'tl', 'xgb', 'effnet', 'hybrid', 'all'], 
                    default='all', help='Wybierz model do uruchomienia')
    parser.add_argument('--split-ratio', type=float, default=0.8,
                        help='Proporcja podziału na zbiór treningowy (domyślnie: 0.8)')
    parser.add_argument('--skip-data-prep', action='store_true',
                        help='Pomiń przygotowanie danych (użyj, gdy dane są już podzielone)')
    parser.add_argument('--force-split', action='store_true',
                        help='Wymuś ponowny podział danych (usunie istniejące pliki w folderach train/test)')
    
    parser.add_argument('--clean-first', action='store_true',
                        help='Wyczyść foldery train/test przed rozpoczęciem')
    
    args = parser.parse_args()
    
    # Tworzenie struktury folderów
    create_folder_structure()
    
    # Przygotowanie danych (podział na zbiory treningowy i testowy)
    if not args.skip_data_prep:
        success = prepare_data(split_ratio=args.split_ratio, force_split=args.force_split)
        if not success:
            print("Błąd podczas przygotowywania danych. Przerwanie programu.")
            return
    
    # Uruchamianie wybranych modeli
    if args.model == 'rf' or args.model == 'all':
        run_random_forest()

    if args.model == 'rf_improved' or args.model == 'all':
        run_random_forest_improved()
    
    if args.model == 'cnn' or args.model == 'all':
        run_cnn()
    
    if args.model == 'cnn_improved' or args.model == 'all':
        run_cnn_improved()
    
    if args.model == 'cnn_v3' or args.model == 'all':
        run_cnn_v3()

    if args.model == 'cnn_v4' or args.model == 'all':
        run_cnn_v4()
    
    if args.model == 'tl' or args.model == 'all':
        run_transfer_learning()
        
    if args.model == 'xgb' or args.model == 'all':
        run_xgboost()
    
    if args.model == 'effnet' or args.model == 'all':
        run_efficientnet()

    if args.model == 'hybrid' or args.model == 'all':
        run_hybrid()

if __name__ == "__main__":
    main()