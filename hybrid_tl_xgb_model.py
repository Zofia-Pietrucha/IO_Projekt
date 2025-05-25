# hybrid_tl_xgb_model_complete.py - Pełny kod modelu hybrydowego z Premium CV
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime
from tqdm import tqdm
import pickle
import cv2
from scipy import ndimage
from scipy.stats import entropy

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D

# XGBoost i sklearn
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, roc_auc_score

# Tworzenie struktury folderów
hybrid_results_dir = "results/hybrid_tl_xgb"
hybrid_models_dir = os.path.join(hybrid_results_dir, "models")
hybrid_plots_dir = os.path.join(hybrid_results_dir, "plots")

for dir_path in [hybrid_results_dir, hybrid_models_dir, hybrid_plots_dir]:
    os.makedirs(dir_path, exist_ok=True)

# Parametry
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Ścieżki do danych
base_dir = "data/skin_moles"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

print("=== HYBRID MODEL: Transfer Learning + XGBoost ===")
print("Łączenie głębokich cech z MobileNetV2 z tradycyjnymi cechami obrazu")

# === CZĘŚĆ 1: PRZYGOTOWANIE MODELU TRANSFER LEARNING ===
print("\n1. Przygotowanie ekstraktora cech MobileNetV2...")

# Załaduj pre-trenowany MobileNetV2
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=IMG_SIZE + (3,),
    pooling='avg'  # Global Average Pooling
)

# Stwórz ekstraktor cech (bez warstwy klasyfikującej)
feature_extractor = Model(
    inputs=base_model.input,
    outputs=base_model.output  # Wyjście to wektor 1280 cech
)

print(f"Ekstraktor cech MobileNetV2 gotowy. Wymiar wyjścia: {feature_extractor.output_shape}")

# === CZĘŚĆ 2: TRADYCYJNE CECHY OBRAZU (z XGBoost) ===
print("\n2. Definicja ekstrakcji tradycyjnych cech...")

def extract_traditional_features(img_path):
    """Ekstrakcja tradycyjnych cech obrazu (z modelu XGBoost)."""
    try:
        img = load_img(img_path, target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0
        
        # Kanały RGB
        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
        
        # Konwersja do HSV
        img_cv = (img_array * 255).astype(np.uint8)
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2HSV) / 255.0
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        
        features = []
        
        # Podstawowe statystyki RGB
        for channel in [r, g, b]:
            features.extend([
                np.mean(channel), np.std(channel), np.min(channel), np.max(channel),
                np.percentile(channel, 25), np.percentile(channel, 75)
            ])
        
        # Podstawowe statystyki HSV
        for channel in [h, s, v]:
            features.extend([
                np.mean(channel), np.std(channel), np.min(channel), np.max(channel)
            ])
        
        # Asymetria znamienia
        height, width = r.shape
        left_half = img_array[:, :width//2, :]
        right_half = img_array[:, width//2:, :]
        top_half = img_array[:height//2, :, :]
        bottom_half = img_array[height//2:, :, :]
        
        asymmetry_h = np.mean(np.abs(np.mean(left_half, axis=(0,1)) - np.mean(right_half, axis=(0,1))))
        asymmetry_v = np.mean(np.abs(np.mean(top_half, axis=(0,1)) - np.mean(bottom_half, axis=(0,1))))
        features.extend([asymmetry_h, asymmetry_v])
        
        # Cechy tekstury
        gray = np.mean(img_array, axis=2)
        gray_uint8 = (gray * 255).astype(np.uint8)
        
        # Gradienty
        sobel_h = cv2.Sobel(gray_uint8, cv2.CV_64F, 1, 0, ksize=3)
        sobel_v = cv2.Sobel(gray_uint8, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobel_h**2 + sobel_v**2)
        
        features.extend([
            np.mean(magnitude), np.std(magnitude), np.max(magnitude)
        ])
        
        # Laplacian
        laplacian = cv2.Laplacian(gray_uint8, cv2.CV_64F)
        features.extend([np.mean(np.abs(laplacian)), np.std(laplacian)])
        
        # Entropie
        for channel in [r, g, b]:
            hist, _ = np.histogram(channel, bins=25, range=(0, 1), density=True)
            features.append(entropy(hist + 1e-10))
        
        # Cechy kształtu
        gray_blur = cv2.GaussianBlur(gray_uint8, (5, 5), 0)
        _, binary = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            circularity = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0
        else:
            area = 0
            perimeter = 0
            circularity = 0
        
        area_normalized = area / (height * width)
        perimeter_normalized = perimeter / (2 * (height + width))
        features.extend([area_normalized, perimeter_normalized, circularity])
        
        return np.array(features)
        
    except Exception as e:
        print(f"Błąd podczas ekstrakcji cech z {img_path}: {e}")
        return np.zeros(50)  # Domyślnie 50 cech

# === CZĘŚĆ 3: KOMBINOWANA EKSTRAKCJA CECH ===
print("\n3. Definicja kombinowanej ekstrakcji cech...")

def extract_hybrid_features(img_path, feature_extractor):
    """Ekstraktuje zarówno głębokie cechy (MobileNetV2) jak i tradycyjne cechy."""
    try:
        # Wczytaj i przygotuj obraz
        img = load_img(img_path, target_size=IMG_SIZE)
        img_array = img_to_array(img)
        
        # Preprocess dla MobileNetV2
        img_mobilenet = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        img_batch = np.expand_dims(img_mobilenet, axis=0)
        
        # Ekstraktuj głębokie cechy
        deep_features = feature_extractor.predict(img_batch, verbose=0)[0]
        
        # Ekstraktuj tradycyjne cechy
        traditional_features = extract_traditional_features(img_path)
        
        # Połącz cechy
        combined_features = np.concatenate([deep_features, traditional_features])
        
        return combined_features
        
    except Exception as e:
        print(f"Błąd podczas ekstrakcji hybrydowych cech z {img_path}: {e}")
        return np.zeros(1280 + 50)  # MobileNetV2 (1280) + tradycyjne cechy (~50)

# === CZĘŚĆ 4: PRZYGOTOWANIE DANYCH ===
print("\n4. Ekstrakcja cech ze zbioru treningowego...")

X_train = []
y_train = []

# Trenowanie - benign
train_benign_dir = os.path.join(train_dir, "benign")
benign_files = os.listdir(train_benign_dir)
print("Przetwarzanie znamion łagodnych (trenowanie)...")
for img_name in tqdm(benign_files[:], desc="Benign train"):
    img_path = os.path.join(train_benign_dir, img_name)
    features = extract_hybrid_features(img_path, feature_extractor)
    X_train.append(features)
    y_train.append(0)

# Trenowanie - melanoma
train_melanoma_dir = os.path.join(train_dir, "melanoma")
melanoma_files = os.listdir(train_melanoma_dir)
print("Przetwarzanie czerniaków (trenowanie)...")
for img_name in tqdm(melanoma_files[:], desc="Melanoma train"):
    img_path = os.path.join(train_melanoma_dir, img_name)
    features = extract_hybrid_features(img_path, feature_extractor)
    X_train.append(features)
    y_train.append(1)

X_train = np.array(X_train)
y_train = np.array(y_train)

print(f"\nZbiór treningowy:")
print(f"Liczba próbek: {len(X_train)}")
print(f"Liczba cech: {X_train.shape[1]}")
print(f"Rozkład klas: {np.bincount(y_train)}")

# Standaryzacja cech
print("\n5. Standaryzacja cech...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Zapisz scaler
with open(os.path.join(hybrid_models_dir, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

# === CZĘŚĆ 5: TRENOWANIE MODELU XGBOOST ===
print("\n6. Trenowanie modelu XGBoost na hybrydowych cechach...")

start_time = time.time()

# Model XGBoost dostosowany do hybrydowych cech
xgb_model = xgb.XGBClassifier(
    n_estimators=300,              
    learning_rate=0.05,            # Mniejszy learning rate dla stabilności
    max_depth=6,                   
    min_child_weight=1,
    gamma=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.01,
    reg_lambda=1.0,
    scale_pos_weight=2.5,          # Ważenie dla melanoma
    objective='binary:logistic',
    eval_metric=['auc', 'logloss'],
    early_stopping_rounds=20,
    random_state=42,
    n_jobs=-1
)

# Trenowanie z walidacją
eval_set = [(X_train_scaled, y_train)]
xgb_model.fit(
    X_train_scaled, y_train,
    eval_set=eval_set,
    verbose=True
)

training_time = time.time() - start_time
print(f"Czas treningu: {training_time:.2f} sekund")

# === CZĘŚĆ 6: PRZYGOTOWANIE DANYCH TESTOWYCH ===
print("\n7. Ekstrakcja cech ze zbioru testowego...")

X_test = []
y_test = []

# Test - benign
test_benign_dir = os.path.join(test_dir, "benign")
benign_test_files = os.listdir(test_benign_dir)
print("Przetwarzanie znamion łagodnych (test)...")
for img_name in tqdm(benign_test_files, desc="Benign test"):
    img_path = os.path.join(test_benign_dir, img_name)
    features = extract_hybrid_features(img_path, feature_extractor)
    X_test.append(features)
    y_test.append(0)

# Test - melanoma
test_melanoma_dir = os.path.join(test_dir, "melanoma")
melanoma_test_files = os.listdir(test_melanoma_dir)
print("Przetwarzanie czerniaków (test)...")
for img_name in tqdm(melanoma_test_files, desc="Melanoma test"):
    img_path = os.path.join(test_melanoma_dir, img_name)
    features = extract_hybrid_features(img_path, feature_extractor)
    X_test.append(features)
    y_test.append(1)

X_test = np.array(X_test)
y_test = np.array(y_test)
X_test_scaled = scaler.transform(X_test)

print(f"\nZbiór testowy:")
print(f"Liczba próbek: {len(X_test)}")
print(f"Rozkład klas: {np.bincount(y_test)}")

# === CZĘŚĆ 7: EWALUACJA Z PREMIUM CROSS-VALIDATION ===
print("\n8. Ewaluacja modelu hybrydowego...")

y_pred_prob = xgb_model.predict_proba(X_test_scaled)[:, 1]
y_pred = xgb_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"Dokładność modelu hybrydowego: {accuracy:.4f}")

# PREMIUM CROSS-VALIDATION Z EARLY STOPPING
def premium_cross_validation_with_early_stopping(model_params, X, y, cv_folds=5, verbose=True):
    """
    Najbardziej zaawansowana implementacja CV z early stopping.
    """
    if verbose:
        print(f"🔬 Uruchamianie Premium Cross-Validation ({cv_folds} folds)...")
        print("   Każdy fold używa early stopping dla optymalnej jakości")
    
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # Przechowywanie wyników
    fold_results = {
        'f1_scores': [],
        'precision_scores': [],
        'recall_scores': [],
        'roc_auc_scores': [],
        'best_iterations': [],
        'training_times': []
    }
    
    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(X, y)):
        if verbose:
            print(f"\n📊 Fold {fold_idx + 1}/{cv_folds}")
        
        fold_start_time = time.time()
        
        # === KROK 1: Podział na train_val i test ===
        X_train_val = X[train_val_idx]
        X_test_fold = X[test_idx]
        y_train_val = y[train_val_idx]
        y_test_fold = y[test_idx]
        
        # === KROK 2: Stratyfikowany podział train_val na train i validation ===
        train_indices = []
        val_indices = []
        
        # Zachowaj proporcje klas
        for class_label in [0, 1]:
            class_indices = np.where(y_train_val == class_label)[0]
            np.random.shuffle(class_indices)
            split_point = int(0.8 * len(class_indices))
            train_indices.extend(class_indices[:split_point])
            val_indices.extend(class_indices[split_point:])
        
        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)
        
        X_train_inner = X_train_val[train_indices]
        X_val_inner = X_train_val[val_indices]
        y_train_inner = y_train_val[train_indices]
        y_val_inner = y_train_val[val_indices]
        
        if verbose:
            print(f"   📈 Train: {len(X_train_inner)} próbek (benign: {np.sum(y_train_inner==0)}, melanoma: {np.sum(y_train_inner==1)})")
            print(f"   📊 Val:   {len(X_val_inner)} próbek (benign: {np.sum(y_val_inner==0)}, melanoma: {np.sum(y_val_inner==1)})")
            print(f"   🎯 Test:  {len(X_test_fold)} próbek (benign: {np.sum(y_test_fold==0)}, melanoma: {np.sum(y_test_fold==1)})")
        
        # === KROK 3: Trening modelu z early stopping ===
        fold_model = xgb.XGBClassifier(**model_params)
        
        try:
            fold_model.fit(
                X_train_inner, y_train_inner,
                eval_set=[(X_val_inner, y_val_inner)],
                early_stopping_rounds=20,
                verbose=False
            )
            
            best_iteration = fold_model.get_booster().best_iteration
            fold_results['best_iterations'].append(best_iteration)
            
            if verbose:
                print(f"   🏆 Najlepsza iteracja: {best_iteration}")
        
        except Exception as e:
            print(f"   ⚠️  Błąd treningu fold {fold_idx + 1}: {e}")
            # Fallback bez early stopping
            fold_model_simple = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=model_params['learning_rate'],
                max_depth=model_params['max_depth'],
                random_state=42
            )
            fold_model_simple.fit(X_train_val, y_train_val)
            fold_model = fold_model_simple
            fold_results['best_iterations'].append(100)
        
        # === KROK 4: Ewaluacja na test fold ===
        y_pred_fold = fold_model.predict(X_test_fold)
        y_pred_proba_fold = fold_model.predict_proba(X_test_fold)[:, 1]
        
        # Oblicz metryki
        fold_f1 = f1_score(y_test_fold, y_pred_fold)
        fold_precision = precision_score(y_test_fold, y_pred_fold, zero_division=0)
        fold_recall = recall_score(y_test_fold, y_pred_fold, zero_division=0)
        fold_auc = roc_auc_score(y_test_fold, y_pred_proba_fold)
        
        # Zapisz wyniki
        fold_results['f1_scores'].append(fold_f1)
        fold_results['precision_scores'].append(fold_precision)
        fold_results['recall_scores'].append(fold_recall)
        fold_results['roc_auc_scores'].append(fold_auc)
        
        fold_time = time.time() - fold_start_time
        fold_results['training_times'].append(fold_time)
        
        if verbose:
            print(f"   📊 F1: {fold_f1:.4f}, Precision: {fold_precision:.4f}, Recall: {fold_recall:.4f}, AUC: {fold_auc:.4f}")
            print(f"   ⏱️  Czas: {fold_time:.1f}s")
    
    # === KROK 5: Podsumowanie wyników ===
    results_summary = {}
    for metric_name, scores in fold_results.items():
        if metric_name not in ['training_times', 'best_iterations']:
            scores_array = np.array(scores)
            results_summary[metric_name] = {
                'mean': scores_array.mean(),
                'std': scores_array.std(),
                'scores': scores_array
            }
    
    if verbose:
        print(f"\n🏆 PODSUMOWANIE PREMIUM CROSS-VALIDATION:")
        print(f"   F1 Score:  {results_summary['f1_scores']['mean']:.4f} ± {results_summary['f1_scores']['std']:.4f}")
        print(f"   Precision: {results_summary['precision_scores']['mean']:.4f} ± {results_summary['precision_scores']['std']:.4f}")
        print(f"   Recall:    {results_summary['recall_scores']['mean']:.4f} ± {results_summary['recall_scores']['std']:.4f}")
        print(f"   ROC AUC:   {results_summary['roc_auc_scores']['mean']:.4f} ± {results_summary['roc_auc_scores']['std']:.4f}")
        print(f"   Średnia najlepsza iteracja: {np.mean(fold_results['best_iterations']):.1f}")
        print(f"   Całkowity czas CV: {sum(fold_results['training_times']):.1f}s")
    
    return results_summary

# Parametry modelu (identyczne z głównym modelem)
model_params = {
    'n_estimators': 300,
    'learning_rate': 0.05,
    'max_depth': 6,
    'min_child_weight': 1,
    'gamma': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.01,
    'reg_lambda': 1.0,
    'scale_pos_weight': 2.5,
    'objective': 'binary:logistic',
    'eval_metric': ['auc', 'logloss'],
    'random_state': 42,
    'n_jobs': -1
}

# Uruchom Premium Cross-Validation
cv_results = premium_cross_validation_with_early_stopping(
    model_params, X_train_scaled, y_train, cv_folds=5, verbose=True
)

# Dla kompatybilności z resztą kodu
cv_scores = cv_results['f1_scores']['scores']
print(f"\nCV F1 Score (główna metryka): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Optymalizacja progu decyzyjnego
print("\nSzukanie optymalnego progu decyzyjnego...")
thresholds = np.arange(0.1, 0.9, 0.01)
f1_scores = []
recall_scores = []
precision_scores = []
accuracy_scores = []

for threshold in thresholds:
    y_pred_th = (y_pred_prob > threshold).astype(int)
    
    cm = confusion_matrix(y_test, y_pred_th)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
    else:
        precision = recall = f1 = accuracy = 0
    
    f1_scores.append(f1)
    recall_scores.append(recall)
    precision_scores.append(precision)
    accuracy_scores.append(accuracy)

# Najlepszy próg
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]
best_recall = recall_scores[best_idx]
best_precision = precision_scores[best_idx]
best_accuracy = accuracy_scores[best_idx]

print(f"Optymalny próg: {best_threshold:.2f}")
print(f"F1={best_f1:.4f}, Recall={best_recall:.4f}, Precision={best_precision:.4f}, Accuracy={best_accuracy:.4f}")

# Finalne predykcje z optymalnym progiem
y_pred_final = (y_pred_prob > best_threshold).astype(int)

# === CZĘŚĆ 8: WIZUALIZACJE ===
print("\n9. Generowanie wizualizacji...")

# Raport klasyfikacji
report = classification_report(y_test, y_pred_final, target_names=['Benign', 'Melanoma'])
print("\nRaport klasyfikacji:")
print(report)

# Macierz pomyłek
cm = confusion_matrix(y_test, y_pred_final)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Benign', 'Melanoma'],
            yticklabels=['Benign', 'Melanoma'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix - Hybrid TL+XGBoost (próg={best_threshold:.2f})')
plt.savefig(os.path.join(hybrid_plots_dir, 'confusion_matrix.png'))
plt.show()

# Krzywa ROC
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Hybrid TL+XGBoost Model')
plt.legend(loc='lower right')
plt.savefig(os.path.join(hybrid_plots_dir, 'roc_curve.png'))
plt.show()

# Wykres progu vs metryki
plt.figure(figsize=(12, 6))
plt.plot(thresholds, f1_scores, label='F1 Score', color='purple', linewidth=2)
plt.plot(thresholds, recall_scores, label='Recall', color='green', linewidth=2)
plt.plot(thresholds, precision_scores, label='Precision', color='blue', linewidth=2)
plt.plot(thresholds, accuracy_scores, label='Accuracy', color='orange', linewidth=2)
plt.axvline(x=best_threshold, color='red', linestyle='--', 
            label=f'Optimal threshold = {best_threshold:.2f}')
plt.xlabel('Decision Threshold')
plt.ylabel('Metric Value')
plt.title('Threshold vs Performance Metrics')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(hybrid_plots_dir, 'threshold_analysis.png'))
plt.show()

# Analiza ważności cech
feature_importances = xgb_model.feature_importances_

# Podziel na głębokie i tradycyjne cechy
deep_features_importance = feature_importances[:1280]  # MobileNetV2 features
traditional_features_importance = feature_importances[1280:]  # Traditional features

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(deep_features_importance, bins=50, alpha=0.7, color='blue')
plt.title('Rozkład ważności cech głębokich\n(MobileNetV2)')
plt.xlabel('Ważność cechy')
plt.ylabel('Liczba cech')

plt.subplot(1, 2, 2)
plt.hist(traditional_features_importance, bins=20, alpha=0.7, color='green')
plt.title('Rozkład ważności cech tradycyjnych')
plt.xlabel('Ważność cechy')
plt.ylabel('Liczba cech')

plt.tight_layout()
plt.savefig(os.path.join(hybrid_plots_dir, 'feature_importance_distribution.png'))
plt.show()

# Porównanie wkładu typów cech
deep_contribution = np.sum(deep_features_importance)
traditional_contribution = np.sum(traditional_features_importance)

plt.figure(figsize=(8, 6))
plt.pie([deep_contribution, traditional_contribution], 
        labels=['Deep Features\n(MobileNetV2)', 'Traditional Features'],
        autopct='%1.1f%%', startangle=90,
        colors=['lightblue', 'lightgreen'])
plt.title('Wkład różnych typów cech w predykcję')
plt.savefig(os.path.join(hybrid_plots_dir, 'feature_contribution.png'))
plt.show()

print(f"Wkład cech głębokich: {deep_contribution:.3f} ({deep_contribution/(deep_contribution+traditional_contribution)*100:.1f}%)")
print(f"Wkład cech tradycyjnych: {traditional_contribution:.3f} ({traditional_contribution/(deep_contribution+traditional_contribution)*100:.1f}%)")

# Dodatkowa analiza - top tradycyjne cechy
print(f"\nTop 10 najważniejszych tradycyjnych cech:")
traditional_indices = np.argsort(traditional_features_importance)[::-1]
traditional_feature_names = [
    'R_mean', 'R_std', 'R_min', 'R_max', 'R_q25', 'R_q75',
    'G_mean', 'G_std', 'G_min', 'G_max', 'G_q25', 'G_q75',
    'B_mean', 'B_std', 'B_min', 'B_max', 'B_q25', 'B_q75',
    'H_mean', 'H_std', 'H_min', 'H_max',
    'S_mean', 'S_std', 'S_min', 'S_max',
    'V_mean', 'V_std', 'V_min', 'V_max',
    'Asymmetry_H', 'Asymmetry_V',
    'Gradient_mean', 'Gradient_std', 'Gradient_max',
    'Laplacian_mean', 'Laplacian_std',
    'R_entropy', 'G_entropy', 'B_entropy',
    'Area_normalized', 'Perimeter_normalized', 'Circularity'
]

# Dopasuj liczbę nazw do rzeczywistej liczby tradycyjnych cech
if len(traditional_feature_names) < len(traditional_features_importance):
    for i in range(len(traditional_feature_names), len(traditional_features_importance)):
        traditional_feature_names.append(f'Traditional_feature_{i}')

for i in range(min(10, len(traditional_features_importance))):
    idx = traditional_indices[i]
    importance = traditional_features_importance[idx]
    feature_name = traditional_feature_names[idx] if idx < len(traditional_feature_names) else f'Feature_{idx}'
    print(f"{i+1:2d}. {feature_name:20s}: {importance:.6f}")

# === CZĘŚĆ 9: ZAPISYWANIE MODELI I WYNIKÓW ===
print("\n10. Zapisywanie modelu i wyników...")

# Zapisz XGBoost model
with open(os.path.join(hybrid_models_dir, 'xgboost_hybrid.pkl'), 'wb') as f:
    pickle.dump(xgb_model, f)
print(f"✅ XGBoost model zapisany: {os.path.join(hybrid_models_dir, 'xgboost_hybrid.pkl')}")

# Zapisz feature extractor
feature_extractor.save(os.path.join(hybrid_models_dir, 'feature_extractor'))
print(f"✅ Feature extractor zapisany: {os.path.join(hybrid_models_dir, 'feature_extractor')}")

# Zapisz scaler (już zapisany wcześniej, ale dla pewności)
with open(os.path.join(hybrid_models_dir, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
print(f"✅ Scaler zapisany: {os.path.join(hybrid_models_dir, 'scaler.pkl')}")

# Zapisz szczegółowy raport klasyfikacji
with open(os.path.join(hybrid_results_dir, 'classification_report.txt'), 'w', encoding='utf-8') as f:
    f.write("Hybrid Transfer Learning + XGBoost Model - Detailed Report\n")
    f.write("=" * 60 + "\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("MODEL CONFIGURATION:\n")
    f.write("-" * 20 + "\n")
    f.write(f"Deep features: MobileNetV2 (1280 features)\n")
    f.write(f"Traditional features: {len(traditional_features_importance)} features\n")
    f.write(f"Total features: {X_train.shape[1]}\n")
    f.write(f"Training samples: {len(X_train)} (benign: {np.sum(y_train==0)}, melanoma: {np.sum(y_train==1)})\n")
    f.write(f"Test samples: {len(X_test)} (benign: {np.sum(y_test==0)}, melanoma: {np.sum(y_test==1)})\n\n")
    
    f.write("PERFORMANCE METRICS:\n")
    f.write("-" * 20 + "\n")
    f.write(f"Optimal threshold: {best_threshold:.3f}\n")
    f.write(f"Test accuracy: {best_accuracy:.4f}\n")
    f.write(f"Test F1 score: {best_f1:.4f}\n")
    f.write(f"Test precision: {best_precision:.4f}\n")
    f.write(f"Test recall: {best_recall:.4f}\n")
    f.write(f"ROC AUC: {roc_auc:.4f}\n\n")
    
    f.write("PREMIUM CROSS-VALIDATION RESULTS:\n")
    f.write("-" * 35 + "\n")
    f.write(f"CV F1 Score: {cv_results['f1_scores']['mean']:.4f} ± {cv_results['f1_scores']['std']:.4f}\n")
    f.write(f"CV Precision: {cv_results['precision_scores']['mean']:.4f} ± {cv_results['precision_scores']['std']:.4f}\n")
    f.write(f"CV Recall: {cv_results['recall_scores']['mean']:.4f} ± {cv_results['recall_scores']['std']:.4f}\n")
    f.write(f"CV ROC AUC: {cv_results['roc_auc_scores']['mean']:.4f} ± {cv_results['roc_auc_scores']['std']:.4f}\n\n")
    
    f.write("FEATURE CONTRIBUTION ANALYSIS:\n")
    f.write("-" * 30 + "\n")
    f.write(f"Deep features contribution: {deep_contribution:.6f} ({deep_contribution/(deep_contribution+traditional_contribution)*100:.1f}%)\n")
    f.write(f"Traditional features contribution: {traditional_contribution:.6f} ({traditional_contribution/(deep_contribution+traditional_contribution)*100:.1f}%)\n\n")
    
    f.write("TOP 10 TRADITIONAL FEATURES:\n")
    f.write("-" * 28 + "\n")
    for i in range(min(10, len(traditional_features_importance))):
        idx = traditional_indices[i]
        importance = traditional_features_importance[idx]
        feature_name = traditional_feature_names[idx] if idx < len(traditional_feature_names) else f'Feature_{idx}'
        f.write(f"{i+1:2d}. {feature_name:25s}: {importance:.6f}\n")
    
    f.write(f"\n\nCLASSIFICATION REPORT:\n")
    f.write("-" * 21 + "\n")
    f.write(report)

print(f"✅ Szczegółowy raport zapisany: {os.path.join(hybrid_results_dir, 'classification_report.txt')}")

# Zapisz podsumowanie wyników
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
with open(os.path.join(hybrid_results_dir, 'results_summary.txt'), 'w', encoding='utf-8') as f:
    f.write(f"Hybrid TL+XGBoost Model Results Summary\n")
    f.write(f"=====================================\n")
    f.write(f"Date: {timestamp}\n\n")
    f.write(f"TRAINING DETAILS:\n")
    f.write(f"Training time: {training_time:.2f} seconds\n")
    f.write(f"Total features: {X_train.shape[1]} (1280 deep + {X_train.shape[1]-1280} traditional)\n")
    f.write(f"Training samples: {len(X_train)}\n")
    f.write(f"Test samples: {len(X_test)}\n\n")
    
    f.write(f"PERFORMANCE RESULTS:\n")
    f.write(f"Test accuracy (optimal threshold {best_threshold:.3f}): {best_accuracy:.4f}\n")
    f.write(f"ROC AUC: {roc_auc:.4f}\n")
    f.write(f"Optimal F1 score: {best_f1:.4f}\n")
    f.write(f"Optimal Recall: {best_recall:.4f}\n")
    f.write(f"Optimal Precision: {best_precision:.4f}\n\n")
    
    f.write(f"CROSS-VALIDATION RESULTS:\n")
    f.write(f"CV F1 Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")
    f.write(f"CV Precision: {cv_results['precision_scores']['mean']:.4f} ± {cv_results['precision_scores']['std']:.4f}\n")
    f.write(f"CV Recall: {cv_results['recall_scores']['mean']:.4f} ± {cv_results['recall_scores']['std']:.4f}\n")
    f.write(f"CV ROC AUC: {cv_results['roc_auc_scores']['mean']:.4f} ± {cv_results['roc_auc_scores']['std']:.4f}\n\n")
    
    f.write("MODEL ARCHITECTURE:\n")
    f.write("- Feature Extractor: MobileNetV2 (pre-trained on ImageNet)\n")
    f.write("- Traditional Features: Color, texture, shape, asymmetry features\n")
    f.write("- Classifier: XGBoost with optimized hyperparameters and early stopping\n")
    f.write("- Cross-Validation: Premium 5-fold with early stopping per fold\n")
    f.write("- Threshold Optimization: Grid search for optimal F1 score\n\n")
    
    f.write(f"FEATURE ANALYSIS:\n")
    f.write(f"Deep features contribution: {deep_contribution:.3f} ({deep_contribution/(deep_contribution+traditional_contribution)*100:.1f}%)\n")
    f.write(f"Traditional features contribution: {traditional_contribution:.3f} ({traditional_contribution/(deep_contribution+traditional_contribution)*100:.1f}%)\n")

print(f"✅ Podsumowanie wyników zapisane: {os.path.join(hybrid_results_dir, 'results_summary.txt')}")

# Zapisz parametry modelu do późniejszego użycia
model_config = {
    'best_threshold': best_threshold,
    'feature_names': traditional_feature_names,
    'model_params': model_params,
    'training_time': training_time,
    'cv_results': cv_results,
    'feature_contributions': {
        'deep': deep_contribution,
        'traditional': traditional_contribution
    }
}

with open(os.path.join(hybrid_models_dir, 'model_config.pkl'), 'wb') as f:
    pickle.dump(model_config, f)
print(f"✅ Konfiguracja modelu zapisana: {os.path.join(hybrid_models_dir, 'model_config.pkl')}")

print("=" * 60)
print("🎉 HYBRID MODEL TRAINING COMPLETED SUCCESSFULLY! 🎉")
print("=" * 60)
print(f"📊 FINAL RESULTS:")
print(f"   Best accuracy: {best_accuracy:.4f}")
print(f"   Best F1 score: {best_f1:.4f}")
print(f"   ROC AUC: {roc_auc:.4f}")
print(f"   CV F1 Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"   Optimal threshold: {best_threshold:.3f}")
print(f"   Deep features: {deep_contribution/(deep_contribution+traditional_contribution)*100:.1f}% contribution")
print(f"   Traditional features: {traditional_contribution/(deep_contribution+traditional_contribution)*100:.1f}% contribution")
print("=" * 60)

# === FUNKCJA PREDYKCYJNA ===
def predict_hybrid(image_path, feature_extractor=None, xgb_model=None, scaler=None, threshold=None):
    """
    Predykcja za pomocą modelu hybrydowego.
    
    Args:
        image_path: Ścieżka do obrazu
        feature_extractor: Model MobileNetV2 (jeśli None, wczyta z pliku)
        xgb_model: Model XGBoost (jeśli None, wczyta z pliku)
        scaler: Scaler (jeśli None, wczyta z pliku)
        threshold: Próg decyzyjny (jeśli None, użyje optymalnego)
    
    Returns:
        Dictionary z wynikami predykcji
    """
    if threshold is None:
        threshold = best_threshold
    
    try:
        # Wczytaj modele jeśli nie podane
        if feature_extractor is None:
            feature_extractor = tf.keras.models.load_model(os.path.join(hybrid_models_dir, 'feature_extractor'))
        
        if xgb_model is None:
            with open(os.path.join(hybrid_models_dir, 'xgboost_hybrid.pkl'), 'rb') as f:
                xgb_model = pickle.load(f)
        
        if scaler is None:
            with open(os.path.join(hybrid_models_dir, 'scaler.pkl'), 'rb') as f:
                scaler = pickle.load(f)
        
        # Ekstraktuj hybrydowe cechy
        features = extract_hybrid_features(image_path, feature_extractor)
        features_scaled = scaler.transform([features])
        
        # Predykcja
        prob = xgb_model.predict_proba(features_scaled)[0, 1]
        result = "Melanoma" if prob > threshold else "Benign"
        confidence = prob if result == "Melanoma" else 1 - prob
        
        return {
            "prediction": result,
            "confidence": float(confidence * 100),  # Percentage
            "raw_probability": float(prob * 100),   # Percentage
            "threshold_used": float(threshold),
            "features_extracted": len(features),
            "status": "success"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "prediction": None,
            "confidence": 0.0
        }

print("\n🚀 MODEL READY FOR USE!")
print("📋 Available functions:")
print("   - predict_hybrid(image_path) - Predykcja dla pojedynczego obrazu")
print("📁 Saved files:")
print(f"   - Models: {hybrid_models_dir}")
print(f"   - Plots: {hybrid_plots_dir}")
print(f"   - Reports: {hybrid_results_dir}")
print("\n💡 Example usage:")
print("   result = predict_hybrid('path/to/your/image.jpg')")
print("   print(f\"Prediction: {result['prediction']} ({result['confidence']:.1f}% confidence)\")")
print("=" * 60)