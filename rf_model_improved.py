# rf_model_improved.py
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tqdm import tqdm
import time
from datetime import datetime
import cv2
from scipy import ndimage
from scipy.stats import entropy
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import gabor
from skimage.segmentation import slic
from skimage.color import rgb2hsv, rgb2lab
import pickle

# Tworzenie struktury folderów dla wyników Random Forest Improved
rf_improved_results_dir = "results/random_forest_improved"
rf_improved_models_dir = os.path.join(rf_improved_results_dir, "models")
rf_improved_plots_dir = os.path.join(rf_improved_results_dir, "plots")

# Tworzymy foldery, jeśli nie istnieją
for dir_path in [rf_improved_results_dir, rf_improved_models_dir, rf_improved_plots_dir]:
    os.makedirs(dir_path, exist_ok=True)

# Ścieżki do danych
base_dir = "data/skin_moles"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

train_benign_dir = os.path.join(train_dir, "benign")
train_melanoma_dir = os.path.join(train_dir, "melanoma")
test_benign_dir = os.path.join(test_dir, "benign")
test_melanoma_dir = os.path.join(test_dir, "melanoma")

print("Przygotowanie danych dla ulepszonego klasyfikatora Random Forest...")

def extract_enhanced_features(img_path):
    """Rozszerzona ekstrakcja cech z obrazu - dodane cechy ABCDE dla melanomy."""
    try:
        # Wczytaj obraz i zmień jego rozmiar do 224x224
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        
        # === PODSTAWOWE CECHY KOLORÓW ===
        # RGB kanały
        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
        
        # Konwersja do innych przestrzeni kolorów
        img_cv = (img_array * 255).astype(np.uint8)
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2HSV) / 255.0
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        
        # LAB space - lepszy dla percepcji kolorów
        lab = rgb2lab(img_array)
        l_lab, a_lab, b_lab = lab[:, :, 0]/100, (lab[:, :, 1]+127)/255, (lab[:, :, 2]+127)/255
        
        # === STATYSTYKI PODSTAWOWE ===
        # RGB
        features = []
        for channel, name in zip([r, g, b], ['r', 'g', 'b']):
            features.extend([
                np.mean(channel), np.std(channel), np.min(channel), np.max(channel),
                np.percentile(channel, 25), np.percentile(channel, 75),
                np.var(channel), entropy(np.histogram(channel, bins=50)[0] + 1e-10)
            ])
        
        # HSV
        for channel, name in zip([h, s, v], ['h', 's', 'v']):
            features.extend([
                np.mean(channel), np.std(channel), np.min(channel), np.max(channel),
                np.percentile(channel, 25), np.percentile(channel, 75)
            ])
        
        # LAB
        for channel, name in zip([l_lab, a_lab, b_lab], ['l', 'a', 'b']):
            features.extend([
                np.mean(channel), np.std(channel), np.var(channel)
            ])
        
        # === CECHY ABCDE DLA MELANOMY ===
        gray = np.mean(img_array, axis=2)
        
        # A - ASYMETRIA
        height, width = gray.shape
        # Asymetria pozioma
        left_half = gray[:, :width//2]
        right_half = np.fliplr(gray[:, width//2:])
        min_width = min(left_half.shape[1], right_half.shape[1])
        asymmetry_h = np.mean(np.abs(left_half[:, :min_width] - right_half[:, :min_width]))
        
        # Asymetria pionowa
        top_half = gray[:height//2, :]
        bottom_half = np.flipud(gray[height//2:, :])
        min_height = min(top_half.shape[0], bottom_half.shape[0])
        asymmetry_v = np.mean(np.abs(top_half[:min_height, :] - bottom_half[:min_height, :]))
        
        features.extend([asymmetry_h, asymmetry_v])
        
        # B - BORDER (granice/brzegi)
        # Wykrywanie brzegów Canny
        gray_uint8 = (gray * 255).astype(np.uint8)
        edges = cv2.Canny(gray_uint8, 50, 150)
        
        # Cechy brzegów
        border_density = np.sum(edges > 0) / (height * width)
        
        # Nieregularność brzegów - używamy kontury
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(largest_contour, True)
            area = cv2.contourArea(largest_contour)
            if perimeter > 0:
                compactness = 4 * np.pi * area / (perimeter ** 2)
            else:
                compactness = 0
        else:
            compactness = 0
        
        features.extend([border_density, compactness])
        
        # C - COLOR (różnorodność kolorów)
        # Liczba unikalnych kolorów (przybliżenie)
        rgb_quantized = (img_array * 16).astype(int)  # Kwantyzacja do 16 poziomów
        unique_colors = len(np.unique(rgb_quantized.reshape(-1, 3), axis=0))
        
        # Dominujący kolor
        hist_r = np.histogram(r, bins=20)[0]
        hist_g = np.histogram(g, bins=20)[0]
        hist_b = np.histogram(b, bins=20)[0]
        
        # Entropia kolorów
        color_entropy = entropy(hist_r + 1e-10) + entropy(hist_g + 1e-10) + entropy(hist_b + 1e-10)
        
        # Różnorodność kolorów w HSV
        hue_spread = np.std(h)
        saturation_mean = np.mean(s)
        
        features.extend([unique_colors, color_entropy, hue_spread, saturation_mean])
        
        # D - DIAMETER/SIZE (wielkość) - symulujemy przez segmentację
        # Segmentacja SLIC do znalezienia głównego obiektu
        segments = slic(img_array, n_segments=10, compactness=10)
        main_segment = np.bincount(segments.flatten()).argmax()
        main_object_pixels = np.sum(segments == main_segment)
        relative_size = main_object_pixels / (height * width)
        
        features.append(relative_size)
        
        # E - EVOLVING/TEXTURE (tekstura)
        # Local Binary Pattern
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        lbp_hist = np.histogram(lbp, bins=n_points + 2)[0]
        lbp_hist = lbp_hist / np.sum(lbp_hist)  # Normalizacja
        
        # Gray Level Co-occurrence Matrix
        gray_glcm = (gray * 255).astype(np.uint8)
        glcm = graycomatrix(gray_glcm, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], 
                          levels=256, symmetric=True, normed=True)
        
        # Właściwości GLCM
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]
        
        features.extend([contrast, dissimilarity, homogeneity, correlation])
        
        # Filtry Gabor dla tekstury
        gabor_responses = []
        for theta in [0, 45, 90, 135]:
            filt_real, _ = gabor(gray, frequency=0.6, theta=np.deg2rad(theta))
            gabor_responses.extend([np.mean(filt_real), np.std(filt_real)])
        
        features.extend(gabor_responses)
        
        # === DODATKOWE CECHY STATYSTYCZNE ===
        # Momenty statystyczne
        for channel in [r, g, b]:
            # Skewness i kurtosis
            mean_ch = np.mean(channel)
            std_ch = np.std(channel)
            if std_ch > 0:
                skewness = np.mean(((channel - mean_ch) / std_ch) ** 3)
                kurtosis = np.mean(((channel - mean_ch) / std_ch) ** 4)
            else:
                skewness = kurtosis = 0
            features.extend([skewness, kurtosis])
        
        # Dodajemy wybrane cechy z LBP histogramu (top 10)
        features.extend(lbp_hist[:10])
        
        return np.array(features)
        
    except Exception as e:
        print(f"Błąd podczas ekstrakcji cech z {img_path}: {e}")
        return np.zeros(200)  # Zwróć wektor zer w przypadku błędu

# Przygotowanie danych treningowych
print("Ekstrakcja cech ze zbioru treningowego...")
X_train = []
y_train = []

# Wczytaj obrazy benign ze zbioru treningowego
print("Wczytywanie obrazów benign (treningowe)...")
benign_files = os.listdir(train_benign_dir)
for img_name in tqdm(benign_files, desc="Benign train"):
    img_path = os.path.join(train_benign_dir, img_name)
    features = extract_enhanced_features(img_path)
    if features is not None:
        X_train.append(features)
        y_train.append(0)  # 0 dla benign

# Wczytaj obrazy melanoma ze zbioru treningowego
print("Wczytywanie obrazów melanoma (treningowe)...")
melanoma_files = os.listdir(train_melanoma_dir)
for img_name in tqdm(melanoma_files, desc="Melanoma train"):
    img_path = os.path.join(train_melanoma_dir, img_name)
    features = extract_enhanced_features(img_path)
    if features is not None:
        X_train.append(features)
        y_train.append(1)  # 1 dla melanoma

# Konwersja list na tablice numpy
X_train = np.array(X_train)
y_train = np.array(y_train)

print(f"Liczba próbek treningowych: {len(X_train)}")
print(f"Liczba cech: {X_train.shape[1]}")
print(f"Rozkład klas: {np.bincount(y_train)}")

# Standaryzacja cech
print("Standaryzacja cech...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Zapisz scaler
scaler_path = os.path.join(rf_improved_models_dir, 'feature_scaler.pkl')
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)

# Optymalizacja hiperparametrów
print("Optymalizacja hiperparametrów...")
param_grid = {
    'n_estimators': [200, 300],
    'max_depth': [15, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2'],
    'class_weight': ['balanced', {0: 1, 1: 2}]
}

# Grid search z cross-validation
rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(
    rf_base, param_grid, cv=5, 
    scoring='f1',  # Optymalizujemy F1 score
    n_jobs=-1, verbose=1
)

print("Rozpoczęcie grid search...")
start_time = time.time()
grid_search.fit(X_train_scaled, y_train)
training_time = time.time() - start_time

print(f"Najlepsze parametry: {grid_search.best_params_}")
print(f"Najlepszy wynik CV: {grid_search.best_score_:.4f}")
print(f"Czas treningu: {training_time:.2f} sekund")

# Najlepszy model
best_rf = grid_search.best_estimator_

# Przygotowanie danych testowych
print("Ekstrakcja cech ze zbioru testowego...")
X_test = []
y_test = []

# Wczytaj obrazy benign ze zbioru testowego
print("Wczytywanie obrazów benign (testowe)...")
benign_test_files = os.listdir(test_benign_dir)
for img_name in tqdm(benign_test_files, desc="Benign test"):
    img_path = os.path.join(test_benign_dir, img_name)
    features = extract_enhanced_features(img_path)
    if features is not None:
        X_test.append(features)
        y_test.append(0)  # 0 dla benign

# Wczytaj obrazy melanoma ze zbioru testowego
print("Wczytywanie obrazów melanoma (testowe)...")
melanoma_test_files = os.listdir(test_melanoma_dir)
for img_name in tqdm(melanoma_test_files, desc="Melanoma test"):
    img_path = os.path.join(test_melanoma_dir, img_name)
    features = extract_enhanced_features(img_path)
    if features is not None:
        X_test.append(features)
        y_test.append(1)  # 1 dla melanoma

# Konwersja i standaryzacja danych testowych
X_test = np.array(X_test)
y_test = np.array(y_test)
X_test_scaled = scaler.transform(X_test)

print(f"Liczba próbek testowych: {len(X_test)}")
print(f"Rozkład klas testowych: {np.bincount(y_test)}")

# Ewaluacja modelu
print("Ewaluacja modelu...")
y_pred = best_rf.predict(X_test_scaled)
y_pred_proba = best_rf.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
print(f"Dokładność ulepszonego Random Forest: {accuracy:.4f}")

# Cross-validation score
cv_scores = cross_val_score(best_rf, X_train_scaled, y_train, cv=5, scoring='f1')
print(f"CV F1 Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Szczegółowy raport klasyfikacji
print("\nRaport klasyfikacji:")
report = classification_report(y_test, y_pred, target_names=['Benign', 'Melanoma'])
print(report)

# Zapisz raport do pliku
with open(os.path.join(rf_improved_results_dir, 'classification_report.txt'), 'w') as f:
    f.write("Ulepszony Random Forest - Raport klasyfikacji\n")
    f.write("=" * 50 + "\n")
    f.write(f"Najlepsze parametry: {grid_search.best_params_}\n")
    f.write(f"CV F1 Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n\n")
    f.write(report)

# Macierz pomyłek
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Benign', 'Melanoma'], 
            yticklabels=['Benign', 'Melanoma'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Ulepszony Random Forest')
plt.savefig(os.path.join(rf_improved_plots_dir, 'confusion_matrix.png'))
plt.show()

# Krzywa ROC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Ulepszony Random Forest')
plt.legend(loc='lower right')
plt.savefig(os.path.join(rf_improved_plots_dir, 'roc_curve.png'))
plt.show()

# Ważność cech z nazwami
feature_names = []
# RGB podstawowe (8 cech x 3 kanały)
for color in ['R', 'G', 'B']:
    feature_names.extend([
        f'{color}_mean', f'{color}_std', f'{color}_min', f'{color}_max',
        f'{color}_q25', f'{color}_q75', f'{color}_var', f'{color}_entropy'
    ])

# HSV (6 cech x 3 kanały)
for color in ['H', 'S', 'V']:
    feature_names.extend([
        f'{color}_mean', f'{color}_std', f'{color}_min', f'{color}_max',
        f'{color}_q25', f'{color}_q75'
    ])

# LAB (3 cechy x 3 kanały)
for color in ['L', 'A', 'B']:
    feature_names.extend([f'{color}_mean', f'{color}_std', f'{color}_var'])

# ABCDE cechy
feature_names.extend(['Asymmetry_H', 'Asymmetry_V', 'Border_Density', 'Compactness'])
feature_names.extend(['Unique_Colors', 'Color_Entropy', 'Hue_Spread', 'Saturation_Mean'])
feature_names.extend(['Relative_Size'])
feature_names.extend(['GLCM_Contrast', 'GLCM_Dissimilarity', 'GLCM_Homogeneity', 'GLCM_Correlation'])

# Gabor responses
for theta in [0, 45, 90, 135]:
    feature_names.extend([f'Gabor_{theta}_mean', f'Gabor_{theta}_std'])

# Momenty statystyczne RGB
for color in ['R', 'G', 'B']:
    feature_names.extend([f'{color}_skewness', f'{color}_kurtosis'])

# LBP features
feature_names.extend([f'LBP_{i}' for i in range(10)])

# Dopasuj długość nazw do rzeczywistej liczby cech
actual_features = X_train.shape[1]
if len(feature_names) != actual_features:
    print(f"Uwaga: Liczba nazw cech ({len(feature_names)}) != liczba cech ({actual_features})")
    # Dodaj brakujące nazwy
    while len(feature_names) < actual_features:
        feature_names.append(f'Feature_{len(feature_names)}')

# Wizualizacja najważniejszych cech
feature_importances = best_rf.feature_importances_
indices = np.argsort(feature_importances)[::-1]

# Top 20 najważniejszych cech
top_k = min(20, len(feature_names))
plt.figure(figsize=(12, 10))
plt.title(f'Top {top_k} najważniejszych cech - Ulepszony Random Forest')
plt.barh(range(top_k), feature_importances[indices][:top_k])
plt.yticks(range(top_k), [feature_names[i] for i in indices][:top_k])
plt.xlabel('Ważność cechy')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(rf_improved_plots_dir, 'feature_importance.png'), bbox_inches='tight')
plt.show()

# Zapisz najlepszy model
model_path = os.path.join(rf_improved_models_dir, 'random_forest_improved_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(best_rf, f)
print(f"Model zapisany do pliku: {model_path}")

# Zapisz podsumowanie wyników
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
with open(os.path.join(rf_improved_results_dir, 'results_summary.txt'), 'w') as f:
    f.write(f"Ulepszony Random Forest Model Results Summary\n")
    f.write(f"==========================================\n")
    f.write(f"Date: {timestamp}\n\n")
    f.write(f"Training time: {training_time:.2f} seconds\n")
    f.write(f"Test accuracy: {accuracy:.4f}\n")
    f.write(f"ROC AUC: {roc_auc:.4f}\n")
    f.write(f"CV F1 Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n\n")
    f.write(f"Best parameters: {grid_search.best_params_}\n\n")
    f.write("Ulepszenia w stosunku do podstawowego RF:\n")
    f.write("1. Cechy ABCDE specyficzne dla melanomy\n")
    f.write("2. Dodatkowe przestrzenie kolorów (HSV, LAB)\n")
    f.write("3. Zaawansowane cechy tekstury (LBP, GLCM, Gabor)\n")
    f.write("4. Standaryzacja cech\n")
    f.write("5. Optymalizacja hiperparametrów\n")
    f.write("6. Cross-validation\n\n")
    f.write(f"Łączna liczba cech: {X_train.shape[1]}\n\n")
    f.write(f"Top 10 najważniejszych cech:\n")
    for i in range(min(10, len(feature_names))):
        idx = indices[i]
        f.write(f"{i+1}. {feature_names[idx]}: {feature_importances[idx]:.6f}\n")

print("Trening i ewaluacja ulepszonego Random Forest zakończone!")

# Funkcja predykcyjna
def predict_skin_lesion_improved(image_path, model, scaler, threshold=0.5):
    """Predykcja dla pojedynczego obrazu z ulepszonym modelem."""
    try:
        features = extract_enhanced_features(image_path)
        if features is None:
            return {"error": "Błąd ekstrakcji cech"}
        
        features_scaled = scaler.transform([features])
        prob = model.predict_proba(features_scaled)[0, 1]
        
        result = "Melanoma" if prob > threshold else "Benign"
        confidence = prob if result == "Melanoma" else 1 - prob
        
        return {
            "prediction": result,
            "confidence": float(confidence),
            "raw_probability": float(prob)
        }
    except Exception as e:
        return {"error": str(e)}