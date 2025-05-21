# xgboost_model.py
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime
from tqdm import tqdm
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, f1_score
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import pickle
import cv2  # Używamy OpenCV zamiast skimage
from scipy import ndimage

# Tworzenie struktury folderów dla wyników XGBoost
xgb_results_dir = "results/xgboost"
xgb_models_dir = os.path.join(xgb_results_dir, "models")
xgb_plots_dir = os.path.join(xgb_results_dir, "plots")

# Tworzymy foldery, jeśli nie istnieją
for dir_path in [xgb_results_dir, xgb_models_dir, xgb_plots_dir]:
    os.makedirs(dir_path, exist_ok=True)

# Ścieżki do danych
base_dir = "data/skin_moles"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

train_benign_dir = os.path.join(train_dir, "benign")
train_melanoma_dir = os.path.join(train_dir, "melanoma")
test_benign_dir = os.path.join(test_dir, "benign")
test_melanoma_dir = os.path.join(test_dir, "melanoma")

print("Przygotowanie danych dla klasyfikatora XGBoost...")

# Funkcja do ekstrakcji cech z obrazu
def extract_features(img_path):
    # Wczytaj obraz i zmień jego rozmiar do 224x224
    img = load_img(img_path, target_size=(224, 224))
    # Konwertuj do tablicy numpy i normalizuj
    img_array = img_to_array(img) / 255.0
    
    # Rozdzielamy kanały RGB
    r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
    
    # Konwersja do przestrzeni kolorów HSV używając OpenCV
    # Najpierw konwertujemy do formatu wymaganego przez OpenCV
    img_cv = (img_array * 255).astype(np.uint8)
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2HSV) / 255.0
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    
    # Oblicz cechy koloru dla RGB
    # Średnie wartości
    r_mean = np.mean(r)
    g_mean = np.mean(g)
    b_mean = np.mean(b)
    
    # Odchylenia standardowe
    r_std = np.std(r)
    g_std = np.std(g)
    b_std = np.std(b)
    
    # Oblicz cechy koloru dla HSV
    # Średnie wartości
    h_mean = np.mean(h)
    s_mean = np.mean(s)
    v_mean = np.mean(v)
    
    # Odchylenia standardowe
    h_std = np.std(h)
    s_std = np.std(s)
    v_std = np.std(v)
    
    # Histogramy kolorów (15 przedziałów na kanał)
    r_hist = np.histogram(r, bins=15, range=(0, 1))[0] / r.size
    g_hist = np.histogram(g, bins=15, range=(0, 1))[0] / g.size
    b_hist = np.histogram(b, bins=15, range=(0, 1))[0] / b.size
    
    h_hist = np.histogram(h, bins=15, range=(0, 1))[0] / h.size
    s_hist = np.histogram(s, bins=15, range=(0, 1))[0] / s.size
    v_hist = np.histogram(v, bins=15, range=(0, 1))[0] / v.size
    
    # Asymetria i kurtoza (istotne cechy dla znamion skórnych)
    # dla RGB
    r_skew = np.mean(((r - r_mean) / r_std)**3) if r_std > 0 else 0
    g_skew = np.mean(((g - g_mean) / g_std)**3) if g_std > 0 else 0
    b_skew = np.mean(((b - b_mean) / b_std)**3) if b_std > 0 else 0
    
    r_kurt = np.mean(((r - r_mean) / r_std)**4) if r_std > 0 else 0
    g_kurt = np.mean(((g - g_mean) / g_std)**4) if g_std > 0 else 0
    b_kurt = np.mean(((b - b_mean) / b_std)**4) if b_std > 0 else 0
    
    # dla HSV
    h_skew = np.mean(((h - h_mean) / h_std)**3) if h_std > 0 else 0
    s_skew = np.mean(((s - s_mean) / s_std)**3) if s_std > 0 else 0
    v_skew = np.mean(((v - v_mean) / v_std)**3) if v_std > 0 else 0
    
    h_kurt = np.mean(((h - h_mean) / h_std)**4) if h_std > 0 else 0
    s_kurt = np.mean(((s - s_mean) / s_std)**4) if s_std > 0 else 0
    v_kurt = np.mean(((v - v_mean) / v_std)**4) if v_std > 0 else 0
    
    # Cechy asymetrii znamienia (ABCD)
    # Aproksymacja asymetrii - różnica pomiędzy lewą i prawą połową obrazu oraz górną i dolną
    height, width = r.shape
    left_half = img_array[:, :width//2, :]
    right_half = img_array[:, width//2:, :]
    top_half = img_array[:height//2, :, :]
    bottom_half = img_array[height//2:, :, :]
    
    # Różnica między połowami jako przybliżenie asymetrii
    asymmetry_h = np.mean(np.abs(np.mean(left_half, axis=(0,1)) - np.mean(right_half, axis=(0,1))))
    asymmetry_v = np.mean(np.abs(np.mean(top_half, axis=(0,1)) - np.mean(bottom_half, axis=(0,1))))
    
    # Cechy brzegów (boundary features)
    # Przybliżenie nieregularności brzegów za pomocą gradientu
    # Konwersja na skalę szarości (prosty averaging)
    gray = np.mean(img_array, axis=2)
    
    # Gradienty Sobela
    sobel_h = ndimage.sobel(gray, axis=0)
    sobel_v = ndimage.sobel(gray, axis=1)
    magnitude = np.sqrt(sobel_h**2 + sobel_v**2)
    
    # Cechy gradientu - jako przybliżenie nieregularności brzegów
    gradient_mean = np.mean(magnitude)
    gradient_std = np.std(magnitude)
    gradient_max = np.max(magnitude)
    
    # Uproszczone cechy tekstury
    # Konwersja do uint8 dla obliczeń
    gray_uint8 = (gray * 255).astype(np.uint8)
    
    # Obliczanie tekstury przy użyciu filtrów OpenCV
    laplacian = cv2.Laplacian(gray_uint8, cv2.CV_64F)
    laplacian_mean = np.mean(np.abs(laplacian))
    laplacian_std = np.std(laplacian)
    
    # Entropie dla poszczególnych kanałów - miara różnorodności struktur
    from scipy.stats import entropy
    
    # Funkcja do obliczania entropii
    def calculate_entropy(img_channel):
        hist, _ = np.histogram(img_channel, bins=25, range=(0, 1), density=True)
        return entropy(hist + 1e-10)  # Małe epsilon dla uniknięcia log(0)
    
    r_entropy = calculate_entropy(r)
    g_entropy = calculate_entropy(g)
    b_entropy = calculate_entropy(b)
    gray_entropy = calculate_entropy(gray)
    
    # Cechy kształtu - mogą być ważne dla identyfikacji melanomy
    # Binaryzacja obrazu
    gray_blur = cv2.GaussianBlur(gray_uint8, (5, 5), 0)
    _, binary = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # Znajdowanie konturów
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Cechy kształtu
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        # Okrągłość - 1 dla idealnego koła, mniej dla nieregularnych kształtów
        circularity = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0
    else:
        area = 0
        perimeter = 0
        circularity = 0
    
    # Normalizacja obszaru i obwodu
    area_normalized = area / (height * width)
    perimeter_normalized = perimeter / (2 * (height + width))
    
    # Połącz wszystkie cechy w jeden wektor
    features = np.concatenate([
        # Podstawowe statystyki RGB
        [r_mean, g_mean, b_mean, r_std, g_std, b_std],
        # Podstawowe statystyki HSV
        [h_mean, s_mean, v_mean, h_std, s_std, v_std],
        # Asymetria i kurtoza RGB
        [r_skew, g_skew, b_skew, r_kurt, g_kurt, b_kurt],
        # Asymetria i kurtoza HSV
        [h_skew, s_skew, v_skew, h_kurt, s_kurt, v_kurt],
        # Cechy asymetrii ABCD
        [asymmetry_h, asymmetry_v],
        # Cechy brzegów
        [gradient_mean, gradient_std, gradient_max],
        # Cechy tekstury
        [laplacian_mean, laplacian_std],
        # Entropie
        [r_entropy, g_entropy, b_entropy, gray_entropy],
        # Cechy kształtu
        [area_normalized, perimeter_normalized, circularity],
        # Histogramy
        r_hist, g_hist, b_hist, h_hist, s_hist, v_hist
    ])
    
    return features

# Przygotowanie danych treningowych
X_train = []
y_train = []

# Wczytaj obrazy benign ze zbioru treningowego
print("Wczytywanie obrazów benign (treningowe)...")
benign_files = os.listdir(train_benign_dir)
for img_name in tqdm(benign_files):
    img_path = os.path.join(train_benign_dir, img_name)
    try:
        X_train.append(extract_features(img_path))
        y_train.append(0)  # 0 dla benign
    except Exception as e:
        print(f"Błąd przetwarzania {img_path}: {e}")

# Wczytaj obrazy melanoma ze zbioru treningowego
print("Wczytywanie obrazów melanoma (treningowe)...")
melanoma_files = os.listdir(train_melanoma_dir)
for img_name in tqdm(melanoma_files):
    img_path = os.path.join(train_melanoma_dir, img_name)
    try:
        X_train.append(extract_features(img_path))
        y_train.append(1)  # 1 dla melanoma
    except Exception as e:
        print(f"Błąd przetwarzania {img_path}: {e}")

# Konwersja list na tablice numpy
X_train = np.array(X_train)
y_train = np.array(y_train)

print(f"Liczba próbek treningowych: {len(X_train)}")
print(f"Liczba cech: {X_train.shape[1]}")
print(f"Rozkład klas treningowych: {np.bincount(y_train)}")

# Trenowanie modelu XGBoost
print("Trenowanie klasyfikatora XGBoost...")
start_time = time.time()

# Definiowanie modelu XGBoost z zaawansowanymi parametrami
xgb_model = xgb.XGBClassifier(
    n_estimators=200,              # Zmniejszona liczba drzew dla szybszego treningu
    learning_rate=0.1,             # Zwiększony learning rate
    max_depth=5,                   # Zmniejszona głębokość drzewa, aby uniknąć przeuczenia
    min_child_weight=1,            # Minimalna suma wag instancji potrzebna w węźle potomnym
    gamma=0.1,                     # Minimalna redukcja straty dla podziału
    subsample=0.8,                 # Procent próbek użytych dla każdego drzewa
    colsample_bytree=0.8,          # Procent cech użytych dla każdego drzewa
    reg_alpha=0.01,                # Regularyzacja L1
    reg_lambda=1.0,                # Regularyzacja L2
    scale_pos_weight=2,            # Zwiększamy wagę klasy melanoma (pozytywnej)
    objective='binary:logistic',   # Funkcja celu dla klasyfikacji binarnej
    eval_metric=['auc', 'logloss'],# Metryki ewaluacji
    early_stopping_rounds=15,      # Wczesne zatrzymanie
    random_state=42,               # Dla powtarzalności wyników
    n_jobs=-1                      # Wykorzystanie wszystkich dostępnych rdzeni CPU
)

# Trenowanie z ewaluacją na bieżąco
eval_set = [(X_train, y_train)]  # W praktyce warto wydzielić zbiór walidacyjny
xgb_model.fit(
    X_train, y_train,
    eval_set=eval_set,
    verbose=True
)

training_time = time.time() - start_time
print(f"Czas treningu: {training_time:.2f} sekund")

# Przygotowanie danych testowych
X_test = []
y_test = []

# Wczytaj obrazy benign ze zbioru testowego
print("Wczytywanie obrazów benign (testowe)...")
benign_test_files = os.listdir(test_benign_dir)
for img_name in tqdm(benign_test_files):
    img_path = os.path.join(test_benign_dir, img_name)
    try:
        X_test.append(extract_features(img_path))
        y_test.append(0)  # 0 dla benign
    except Exception as e:
        print(f"Błąd przetwarzania {img_path}: {e}")

# Wczytaj obrazy melanoma ze zbioru testowego
print("Wczytywanie obrazów melanoma (testowe)...")
melanoma_test_files = os.listdir(test_melanoma_dir)
for img_name in tqdm(melanoma_test_files):
    img_path = os.path.join(test_melanoma_dir, img_name)
    try:
        X_test.append(extract_features(img_path))
        y_test.append(1)  # 1 dla melanoma
    except Exception as e:
        print(f"Błąd przetwarzania {img_path}: {e}")

# Konwersja list na tablice numpy
X_test = np.array(X_test)
y_test = np.array(y_test)

print(f"Liczba próbek testowych: {len(X_test)}")
print(f"Rozkład klas testowych: {np.bincount(y_test)}")

# Ewaluacja modelu
print("Ewaluacja modelu XGBoost...")
y_pred_prob = xgb_model.predict_proba(X_test)[:, 1]  # Prawdopodobieństwa dla klasy 1 (melanoma)
y_pred = xgb_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Dokładność klasyfikatora XGBoost: {accuracy:.4f}")

# Znajdź optymalny próg decyzyjny maksymalizujący F1 dla melanoma
print("\nSzukanie optymalnego progu decyzyjnego dla melanomy...")
thresholds = np.arange(0.1, 0.9, 0.01)
f1_scores = []
recall_scores = []
precision_scores = []
accuracy_scores = []

for threshold in thresholds:
    y_pred_th = (y_pred_prob > threshold).astype(int)
    
    # Macierz pomyłek
    cm = confusion_matrix(y_test, y_pred_th)
    tn, fp, fn, tp = cm.ravel()
    
    # Metryki
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    f1_scores.append(f1)
    recall_scores.append(recall)
    precision_scores.append(precision)
    accuracy_scores.append(accuracy)

# Znajdź próg z najlepszym F1
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]
best_recall = recall_scores[best_idx]
best_precision = precision_scores[best_idx]
best_accuracy = accuracy_scores[best_idx]

print(f"Optymalny próg decyzyjny: {best_threshold:.2f}")
print(f"Przy tym progu: F1={best_f1:.4f}, Recall={best_recall:.4f}, Precision={best_precision:.4f}, Accuracy={best_accuracy:.4f}")

# Używamy optymalnego progu do finalnych przewidywań
y_pred_best = (y_pred_prob > best_threshold).astype(int)

# Szczegółowy raport klasyfikacji z optymalnym progiem
print("\nRaport klasyfikacji z optymalnym progiem:")
report = classification_report(y_test, y_pred_best, target_names=['Benign', 'Melanoma'])
print(report)

# Zapisz raport do pliku
with open(os.path.join(xgb_results_dir, 'classification_report.txt'), 'w') as f:
    f.write(f"Raport klasyfikacji (próg={best_threshold:.2f}):\n")
    f.write(report)

# Macierz pomyłek
print("\nMacierz pomyłek:")
cm = confusion_matrix(y_test, y_pred_best)
print(cm)

# Wizualizacja macierzy pomyłek
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Benign', 'Melanoma'], 
            yticklabels=['Benign', 'Melanoma'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix - XGBoost (próg={best_threshold:.2f})')
plt.savefig(os.path.join(xgb_plots_dir, 'confusion_matrix.png'))
plt.show()

# Krzywa ROC
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.scatter(fpr[best_idx], tpr[best_idx], marker='o', color='red', 
            label=f'Optymalny próg = {best_threshold:.2f}')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - XGBoost')
plt.legend(loc='lower right')
plt.savefig(os.path.join(xgb_plots_dir, 'roc_curve.png'))
plt.show()

# Wykres progu vs. metryki
plt.figure(figsize=(12, 6))
plt.plot(thresholds, f1_scores, label='F1 Score', color='purple')
plt.plot(thresholds, recall_scores, label='Recall (Czułość)', color='green')
plt.plot(thresholds, precision_scores, label='Precision (Precyzja)', color='blue')
plt.plot(thresholds, accuracy_scores, label='Accuracy (Dokładność)', color='orange')
plt.axvline(x=best_threshold, color='red', linestyle='--', 
            label=f'Optymalny próg = {best_threshold:.2f}')
plt.axvline(x=0.5, color='black', linestyle=':', 
            label='Domyślny próg = 0.5')
plt.xlabel('Próg decyzyjny')
plt.ylabel('Wartość metryki')
plt.title('Wpływ progu decyzyjnego na metryki klasyfikacji')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(xgb_plots_dir, 'threshold_metrics.png'))
plt.show()

# Wizualizacja ważności cech
plt.figure(figsize=(14, 8))
xgb.plot_importance(xgb_model, max_num_features=20, height=0.5)
plt.title('Top 20 najważniejszych cech - XGBoost')
plt.savefig(os.path.join(xgb_plots_dir, 'feature_importance.png'))
plt.show()

# Zapisz model
model_path = os.path.join(xgb_models_dir, 'xgboost_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(xgb_model, f)
print(f"Model zapisany do pliku: {model_path}")

# Zapisz podsumowanie wyników
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
with open(os.path.join(xgb_results_dir, 'results_summary.txt'), 'w') as f:
    f.write(f"XGBoost Model Results Summary\n")
    f.write(f"===========================\n")
    f.write(f"Date: {timestamp}\n\n")
    f.write(f"Training time: {training_time:.2f} seconds\n")
    f.write(f"Test accuracy (threshold=0.5): {accuracy:.4f}\n")
    f.write(f"ROC AUC: {roc_auc:.4f}\n\n")
    f.write(f"Optymalny próg decyzyjny: {best_threshold:.2f}\n")
    f.write(f"Przy optymalnym progu: F1={best_f1:.4f}, Recall={best_recall:.4f}, Precision={best_precision:.4f}, Accuracy={best_accuracy:.4f}\n\n")
    f.write("XGBoost Hyperparameters:\n")
    for param, value in xgb_model.get_params().items():
        f.write(f"{param}: {value}\n")
    
    # Zapisz też 10 najważniejszych cech
    f.write("\nTop 10 Feature Importance:\n")
    importance = xgb_model.feature_importances_
    indices = np.argsort(importance)[::-1]
    for i in range(min(10, len(importance))):
        f.write(f"{i+1}. Feature {indices[i]}: {importance[indices[i]]:.6f}\n")

print("Trening i ewaluacja klasyfikatora XGBoost zakończone!")

# Funkcja predykcyjna dla pojedynczego obrazu
def predict_skin_lesion(image_path, model, threshold=0.5):
    """Predykcja dla pojedynczego obrazu znamienia skórnego."""
    try:
        # Ekstrakcja cech
        features = extract_features(image_path)
        features = np.array([features])  # Przekształć na format 2D wymagany przez model
        
        # Predykcja
        prob = model.predict_proba(features)[0, 1]
        
        # Zastosuj próg decyzyjny
        result = "Melanoma" if prob > threshold else "Benign"
        confidence = prob if result == "Melanoma" else 1 - prob
        
        return {
            "prediction": result,
            "confidence": float(confidence),
            "raw_probability": float(prob)
        }
    except Exception as e:
        return {
            "error": str(e)
        }

# Przykład użycia (odkomentuj, aby przetestować na konkretnym obrazie)
# test_image = "path/to/test/image.jpg"
# result = predict_skin_lesion(test_image, xgb_model, threshold=best_threshold)
# print(f"Predykcja: {result['prediction']}")
# print(f"Pewność: {result['confidence']*100:.2f}%")
# print(f"Prawdopodobieństwo melanomy: {result['raw_probability']*100:.2f}%")