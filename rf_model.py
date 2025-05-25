import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tqdm import tqdm
import time
from datetime import datetime

rf_results_dir = "results/random_forest"
rf_models_dir = os.path.join(rf_results_dir, "models")
rf_plots_dir = os.path.join(rf_results_dir, "plots")

for dir_path in [rf_results_dir, rf_models_dir, rf_plots_dir]:
    os.makedirs(dir_path, exist_ok=True)

base_dir = "data/skin_moles"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

train_benign_dir = os.path.join(train_dir, "benign")
train_melanoma_dir = os.path.join(train_dir, "melanoma")
test_benign_dir = os.path.join(test_dir, "benign")
test_melanoma_dir = os.path.join(test_dir, "melanoma")

print("Przygotowanie danych dla klasyfikatora Random Forest...")

# Funkcja do ekstrakcji cech z obrazu
def extract_features(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    
    # Oblicz cechy koloru
    # Średnie wartości RGB
    r_mean = np.mean(img_array[:, :, 0])
    g_mean = np.mean(img_array[:, :, 1])
    b_mean = np.mean(img_array[:, :, 2])
    
    # Odchylenia standardowe RGB
    r_std = np.std(img_array[:, :, 0])
    g_std = np.std(img_array[:, :, 1])
    b_std = np.std(img_array[:, :, 2])
    
    # Histogramy kolorów (20 przedziałów na kanał dla lepszej dokładności)
    r_hist = np.histogram(img_array[:, :, 0], bins=20, range=(0, 1))[0]
    g_hist = np.histogram(img_array[:, :, 1], bins=20, range=(0, 1))[0]
    b_hist = np.histogram(img_array[:, :, 2], bins=20, range=(0, 1))[0]
    
    # Asymetria i kurtoza (istotne cechy dla znamion skórnych)
    r_skew = np.mean(((img_array[:, :, 0] - r_mean) / r_std)**3) if r_std > 0 else 0
    g_skew = np.mean(((img_array[:, :, 1] - g_mean) / g_std)**3) if g_std > 0 else 0
    b_skew = np.mean(((img_array[:, :, 2] - b_mean) / b_std)**3) if b_std > 0 else 0
    
    r_kurt = np.mean(((img_array[:, :, 0] - r_mean) / r_std)**4) if r_std > 0 else 0
    g_kurt = np.mean(((img_array[:, :, 1] - g_mean) / g_std)**4) if g_std > 0 else 0
    b_kurt = np.mean(((img_array[:, :, 2] - b_mean) / b_std)**4) if b_std > 0 else 0
    
    features = np.concatenate([
        [r_mean, g_mean, b_mean, r_std, g_std, b_std, r_skew, g_skew, b_skew, r_kurt, g_kurt, b_kurt], 
        r_hist, g_hist, b_hist
    ])
    
    return features

X_train = []
y_train = []

print("Wczytywanie obrazów benign (treningowe)...")
benign_files = os.listdir(train_benign_dir)
for img_name in tqdm(benign_files):
    img_path = os.path.join(train_benign_dir, img_name)
    X_train.append(extract_features(img_path))
    y_train.append(0)  # 0 dla benign

print("Wczytywanie obrazów melanoma (treningowe)...")
melanoma_files = os.listdir(train_melanoma_dir)
for img_name in tqdm(melanoma_files):
    img_path = os.path.join(train_melanoma_dir, img_name)
    X_train.append(extract_features(img_path))
    y_train.append(1)  # 1 dla melanoma

X_train = np.array(X_train)
y_train = np.array(y_train)

print(f"Liczba próbek treningowych: {len(X_train)}")
print(f"Liczba cech: {X_train.shape[1]}")

print("Trenowanie klasyfikatora Random Forest...")
start_time = time.time()

rf_classifier = RandomForestClassifier(
    n_estimators=200,      # Zwiększona liczba drzew
    max_depth=20,          # Kontrola głębokości drzew
    min_samples_split=5,   # Minimalna liczba próbek do podziału węzła
    min_samples_leaf=2,    # Minimalna liczba próbek w liściu
    random_state=42,       # Dla powtarzalności wyników
    n_jobs=-1              # Wykorzystanie wszystkich dostępnych rdzeni CPU
)

rf_classifier.fit(X_train, y_train)
training_time = time.time() - start_time
print(f"Czas treningu: {training_time:.2f} sekund")

X_test = []
y_test = []

print("Wczytywanie obrazów benign (testowe)...")
benign_test_files = os.listdir(test_benign_dir)
for img_name in tqdm(benign_test_files):
    img_path = os.path.join(test_benign_dir, img_name)
    X_test.append(extract_features(img_path))
    y_test.append(0)  # 0 dla benign

print("Wczytywanie obrazów melanoma (testowe)...")
melanoma_test_files = os.listdir(test_melanoma_dir)
for img_name in tqdm(melanoma_test_files):
    img_path = os.path.join(test_melanoma_dir, img_name)
    X_test.append(extract_features(img_path))
    y_test.append(1)  # 1 dla melanoma

X_test = np.array(X_test)
y_test = np.array(y_test)

print(f"Liczba próbek testowych: {len(X_test)}")

print("Ewaluacja modelu...")
y_pred = rf_classifier.predict(X_test)
y_pred_proba = rf_classifier.predict_proba(X_test)[:, 1]  # Prawdopodobieństwa dla klasy 1 (melanoma)

accuracy = accuracy_score(y_test, y_pred)
print(f"Dokładność klasyfikatora Random Forest: {accuracy:.4f}")

print("\nRaport klasyfikacji:")
report = classification_report(y_test, y_pred, target_names=['Benign', 'Melanoma'])
print(report)

with open(os.path.join(rf_results_dir, 'classification_report.txt'), 'w') as f:
    f.write(report)

# Macierz pomyłek
print("\nMacierz pomyłek:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Wizualizacja macierzy pomyłek
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Benign', 'Melanoma'], 
            yticklabels=['Benign', 'Melanoma'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Random Forest')
plt.savefig(os.path.join(rf_plots_dir, 'confusion_matrix.png'))
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
plt.title('Receiver Operating Characteristic - Random Forest')
plt.legend(loc='lower right')
plt.savefig(os.path.join(rf_plots_dir, 'roc_curve.png'))
plt.show()

# Ważność cech
feature_importances = rf_classifier.feature_importances_
feature_names = ['R Mean', 'G Mean', 'B Mean', 'R Std', 'G Std', 'B Std', 
                'R Skew', 'G Skew', 'B Skew', 'R Kurt', 'G Kurt', 'B Kurt']
feature_names.extend([f'R Hist {i}' for i in range(20)])
feature_names.extend([f'G Hist {i}' for i in range(20)])
feature_names.extend([f'B Hist {i}' for i in range(20)])

# Sortuj ważności cech i pokaż 15 najważniejszych
indices = np.argsort(feature_importances)[::-1]
top_k = 15
plt.figure(figsize=(12, 8))
plt.title('Top 15 najważniejszych cech - Random Forest')
plt.bar(range(top_k), feature_importances[indices][:top_k], align='center')
plt.xticks(range(top_k), [feature_names[i] for i in indices][:top_k], rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(rf_plots_dir, 'feature_importance.png'))
plt.show()

# Zapisz model
import pickle
model_path = os.path.join(rf_models_dir, 'random_forest_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(rf_classifier, f)
print(f"Model zapisany do pliku: {model_path}")

# Zapisz podsumowanie wyników
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
with open(os.path.join(rf_results_dir, 'results_summary.txt'), 'w') as f:
    f.write(f"Random Forest Model Results Summary\n")
    f.write(f"=================================\n")
    f.write(f"Date: {timestamp}\n\n")
    f.write(f"Training time: {training_time:.2f} seconds\n")
    f.write(f"Test accuracy: {accuracy:.4f}\n")
    f.write(f"ROC AUC: {roc_auc:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\n\nTop 15 most important features:\n")
    for i in range(top_k):
        idx = indices[i]
        f.write(f"{i+1}. {feature_names[idx]}: {feature_importances[idx]:.4f}\n")

print("Trening i ewaluacja klasyfikatora Random Forest zakończone!")