# cnn_v3_model.py
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input, Add, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, f1_score
import tensorflow as tf
from tensorflow.keras import backend as K

# Tworzenie struktury folderów dla wyników CNN v3
cnn_v3_results_dir = "results/cnn_v3"
cnn_v3_models_dir = os.path.join(cnn_v3_results_dir, "models")
cnn_v3_plots_dir = os.path.join(cnn_v3_results_dir, "plots")

# Tworzymy foldery, jeśli nie istnieją
for dir_path in [cnn_v3_results_dir, cnn_v3_models_dir, cnn_v3_plots_dir]:
    os.makedirs(dir_path, exist_ok=True)

# Parametry modelu
img_width, img_height = 224, 224
batch_size = 32
epochs = 30  # Zwiększona liczba epok

# Ścieżki do danych
base_dir = "data/skin_moles"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

# Policzmy liczbę próbek w każdej klasie, aby określić ważenie klas
train_benign_dir = os.path.join(train_dir, "benign")
train_melanoma_dir = os.path.join(train_dir, "melanoma")
n_benign = len(os.listdir(train_benign_dir))
n_melanoma = len(os.listdir(train_melanoma_dir))
total = n_benign + n_melanoma

# Obliczenie wag klas (odwrotnie proporcjonalne do liczby próbek)
# Dajemy znacznie większą wagę klasie melanoma (3x), aby zwiększyć jej czułość
weight_for_benign = (1 / n_benign) * total / 2.0
weight_for_melanoma = (1 / n_melanoma) * total / 2.0 * 3.0  # Zwiększono 3x zamiast 1.5x

class_weights = {0: weight_for_benign, 1: weight_for_melanoma}
print(f"Wagi klas: Benign={weight_for_benign:.3f}, Melanoma={weight_for_melanoma:.3f}")

# Focal Loss - funkcja straty która skupia się na trudnych przypadkach
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        # Clip bardzo małych wartości, aby uniknąć problemów numerycznych
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        # Obliczanie focal loss dla 0 i 1
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        
        # Obliczanie focal loss
        loss = -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - \
               K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
        return loss
    return focal_loss_fixed

# Zaawansowana augmentacja danych dla zbioru treningowego
train_datagen = ImageDataGenerator(
    rescale=1./255,                  # Normalizacja
    rotation_range=180,              # Pełny zakres obrotów
    width_shift_range=0.15,          # Przesunięcia poziome
    height_shift_range=0.15,         # Przesunięcia pionowe
    brightness_range=[0.7, 1.3],     # Większa różnorodność jasności
    shear_range=15,                  # Ścinanie/deformacja
    zoom_range=[0.85, 1.15],         # Przybliżenie/oddalenie
    horizontal_flip=True,            # Odbicie w poziomie
    vertical_flip=True,              # Odbicie w pionie
    fill_mode='reflect',             # Lepsze wypełnianie
    channel_shift_range=20.0,        # Zmiany w kanałach kolorów
)

# Generator dla danych testowych (tylko normalizacja)
test_datagen = ImageDataGenerator(rescale=1./255)

# Przygotowanie generatorów
print("Przygotowanie generatorów danych...")
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False  # Ważne dla ewaluacji!
)

# Funkcja tworząca blok residualny
def residual_block(x, filters, kernel_size=3, strides=1):
    shortcut = x
    
    # Pierwszy zestaw konwolucji
    x = Conv2D(filters, kernel_size, strides=strides, padding='same', kernel_regularizer=l2(0.0005))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Drugi zestaw konwolucji
    x = Conv2D(filters, kernel_size, padding='same', kernel_regularizer=l2(0.0005))(x)
    x = BatchNormalization()(x)
    
    # Jeśli wymiary się zmieniły (strides>1), dopasuj shortcut
    if strides > 1 or shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, 1, strides=strides, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
    
    # Dodaj shortcut connection
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    
    return x

# Definicja modelu CNN v3 z połączeniami rezydualnymi
def create_resnet_model():
    inputs = Input(shape=(img_width, img_height, 3))
    
    # Początkowa konwolucja
    x = Conv2D(64, 7, strides=2, padding='same', kernel_regularizer=l2(0.0005))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(3, strides=2, padding='same')(x)
    
    # Bloki rezydualne
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = residual_block(x, 128, strides=2)
    x = residual_block(x, 128)
    x = residual_block(x, 256, strides=2)
    x = residual_block(x, 256)
    x = residual_block(x, 512, strides=2)
    x = residual_block(x, 512)
    
    # Global Average Pooling zamiast Flatten
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # Warstwa w pełni połączona z dropout
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.0005))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)  # Mniejszy dropout
    
    # Warstwa wyjściowa z sigmoid dla klasyfikacji binarnej
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Utwórz model
print("Tworzenie modelu CNN v3...")
model = create_resnet_model()

# Kompilacja modelu z focal loss
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss=focal_loss(gamma=2.0, alpha=0.25),  # Focal loss zamiast binary crossentropy
    metrics=[
        'accuracy',
        tf.keras.metrics.Recall(name='recall'),  
        tf.keras.metrics.Precision(name='precision'),  
        tf.keras.metrics.AUC(name='auc')
    ]
)

# Podsumowanie architektury modelu
model.summary()

# Zapisz podsumowanie architektury do pliku
with open(os.path.join(cnn_v3_results_dir, 'model_architecture.txt'), 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

# Przygotowanie callbacków
checkpoint = ModelCheckpoint(
    os.path.join(cnn_v3_models_dir, 'best_model.h5'),
    monitor='val_recall',  # Monitorujemy czułość (recall)
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_recall',
    patience=10,  # Większa cierpliwość
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

callbacks = [checkpoint, early_stopping, reduce_lr]

# Trening modelu
print("Rozpoczęcie treningu modelu CNN v3...")
start_time = time.time()

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size,
    callbacks=callbacks,
    class_weight=class_weights  # Zastosowanie wag klas
)

training_time = time.time() - start_time
print(f"Czas treningu CNN v3: {training_time:.2f} sekund")

# Zapisz historię treningu
np.save(os.path.join(cnn_v3_results_dir, 'training_history.npy'), history.history)

# Wczytaj najlepszy model (zapisany przez callback)
model.load_weights(os.path.join(cnn_v3_models_dir, 'best_model.h5'))

# Wizualizacja historii treningu (3 wykresy: dokładność, funkcja straty, recall)
plt.figure(figsize=(18, 6))

# Wykres dokładności
plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Trening')
plt.plot(history.history['val_accuracy'], label='Walidacja')
plt.title('Dokładność modelu CNN v3')
plt.xlabel('Epoka')
plt.ylabel('Dokładność')
plt.legend(loc='lower right')

# Wykres funkcji straty
plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label='Trening')
plt.plot(history.history['val_loss'], label='Walidacja')
plt.title('Funkcja straty modelu CNN v3')
plt.xlabel('Epoka')
plt.ylabel('Strata')
plt.legend(loc='upper right')

# Wykres czułości (recall)
plt.subplot(1, 3, 3)
plt.plot(history.history['recall'], label='Trening')
plt.plot(history.history['val_recall'], label='Walidacja')
plt.title('Czułość (Recall) modelu CNN v3')
plt.xlabel('Epoka')
plt.ylabel('Czułość')
plt.legend(loc='lower right')

plt.tight_layout()
plt.savefig(os.path.join(cnn_v3_plots_dir, 'training_history.png'))
plt.show()

# Ewaluacja modelu na zbiorze testowym z domyślnym progiem 0.5
print("Ewaluacja modelu CNN v3 z progiem 0.5...")
test_generator.reset()
results = model.evaluate(test_generator)
print(f"Test loss: {results[0]:.4f}")
print(f"Test accuracy: {results[1]:.4f}")
print(f"Test recall: {results[2]:.4f}")
print(f"Test precision: {results[3]:.4f}")
print(f"Test AUC: {results[4]:.4f}")

# Przewidywania na zbiorze testowym
test_generator.reset()
y_pred_prob = model.predict(test_generator)
y_true = test_generator.classes

# Znajdź optymalny próg decyzyjny maksymalizujący F1
print("\nSzukanie optymalnego progu decyzyjnego...")
thresholds = np.arange(0.1, 0.9, 0.01)
f1_scores = []
recall_scores = []
precision_scores = []

for threshold in thresholds:
    y_pred = (y_pred_prob > threshold).astype(int).flatten()
    # Obliczamy F1 dla klasy melanoma (pozytywnej)
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_positives = np.sum((y_true == 0) & (y_pred == 1))
    false_negatives = np.sum((y_true == 1) & (y_pred == 0))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    f1_scores.append(f1)
    recall_scores.append(recall)
    precision_scores.append(precision)

# Znajdź próg z najlepszym F1
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]
best_recall = recall_scores[best_idx]
best_precision = precision_scores[best_idx]

print(f"Optymalny próg decyzyjny: {best_threshold:.2f}")
print(f"Przy tym progu: F1={best_f1:.4f}, Recall={best_recall:.4f}, Precision={best_precision:.4f}")

# Użyj optymalnego progu do finalnych przewidywań
y_pred = (y_pred_prob > best_threshold).astype(int).flatten()

# Macierz pomyłek z optymalnym progiem
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Benign', 'Melanoma'],
            yticklabels=['Benign', 'Melanoma'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix - CNN v3 (próg={best_threshold:.2f})')
plt.savefig(os.path.join(cnn_v3_plots_dir, 'confusion_matrix.png'))
plt.show()

# Raport klasyfikacji z optymalnym progiem
print("\nRaport klasyfikacji z optymalnym progiem:")
report = classification_report(y_true, y_pred, target_names=['Benign', 'Melanoma'])
print(report)

# Zapisz raport do pliku
with open(os.path.join(cnn_v3_results_dir, 'classification_report.txt'), 'w') as f:
    f.write(f"Raport klasyfikacji (próg={best_threshold:.2f}):\n")
    f.write(report)

# Krzywa ROC
fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
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
plt.title('Receiver Operating Characteristic - CNN v3')
plt.legend(loc='lower right')
plt.savefig(os.path.join(cnn_v3_plots_dir, 'roc_curve.png'))
plt.show()

# Wykres progu vs. metryki
plt.figure(figsize=(12, 6))
plt.plot(thresholds, f1_scores, label='F1 Score', color='purple')
plt.plot(thresholds, recall_scores, label='Recall (Czułość)', color='green')
plt.plot(thresholds, precision_scores, label='Precision (Precyzja)', color='blue')
plt.axvline(x=best_threshold, color='red', linestyle='--', 
            label=f'Optymalny próg = {best_threshold:.2f}')
plt.axvline(x=0.5, color='black', linestyle=':', 
            label='Domyślny próg = 0.5')
plt.xlabel('Próg decyzyjny')
plt.ylabel('Wartość metryki')
plt.title('Wpływ progu decyzyjnego na metryki klasyfikacji')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(cnn_v3_plots_dir, 'threshold_metrics.png'))
plt.show()

# Zapisz model w formacie TensorFlow SavedModel (dla aplikacji webowej)
model.save(os.path.join(cnn_v3_models_dir, 'cnn_v3_model'))
print(f"Model zapisany w: {os.path.join(cnn_v3_models_dir, 'cnn_v3_model')}")

# Zapisz podsumowanie wyników
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
with open(os.path.join(cnn_v3_results_dir, 'results_summary.txt'), 'w') as f:
    f.write(f"CNN v3 Model Results Summary\n")
    f.write(f"=========================\n")
    f.write(f"Date: {timestamp}\n\n")
    f.write(f"Training time: {training_time:.2f} seconds\n")
    f.write(f"Test accuracy: {results[1]:.4f}\n")
    f.write(f"Test recall: {results[2]:.4f}\n")
    f.write(f"Test precision: {results[3]:.4f}\n")
    f.write(f"Test AUC: {results[4]:.4f}\n\n")
    f.write(f"Optymalny próg decyzyjny: {best_threshold:.2f}\n")
    f.write(f"Przy optymalnym progu: F1={best_f1:.4f}, Recall={best_recall:.4f}, Precision={best_precision:.4f}\n\n")
    f.write("Główne ulepszenia w CNN v3:\n")
    f.write("1. Architektura rezydualna (ResNet-like) zamiast standardowej CNN\n")
    f.write("2. Focal Loss zamiast Binary Crossentropy\n")
    f.write("3. Znacznie silniejsze ważenie klasy melanoma (3x)\n")
    f.write("4. Rozszerzona i ulepszona augmentacja danych\n")
    f.write("5. GlobalAveragePooling zamiast Flatten\n")
    f.write("6. Mniejszy Dropout (0.4 zamiast 0.5)\n")
    f.write("7. Więcej epok i większa cierpliwość early stopping\n\n")
    f.write("Raport klasyfikacji przy optymalnym progu:\n")
    f.write(report)

print("Trening i ewaluacja modelu CNN v3 zakończone!")

# Dodajmy funkcję do predykcji na pojedynczym obrazie
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def predict_skin_lesion(image_path, model, threshold=0.5):
    """Predykcja dla pojedynczego obrazu znamienia skórnego."""
    # Wczytaj i przygotuj obraz
    img = load_img(image_path, target_size=(img_width, img_height))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Dodaj wymiar batcha
    
    # Predykcja
    prediction = model.predict(img_array)[0][0]
    
    # Zastosuj próg decyzyjny
    result = "Melanoma" if prediction > threshold else "Benign"
    confidence = prediction if result == "Melanoma" else 1 - prediction
    
    return {
        "prediction": result,
        "confidence": float(confidence),
        "raw_probability": float(prediction)
    }

# Przykład użycia (odkomentuj, aby przetestować na konkretnym obrazie)
# test_image = "path/to/test/image.jpg"
# result = predict_skin_lesion(test_image, model, threshold=best_threshold)
# print(f"Predykcja: {result['prediction']}")
# print(f"Pewność: {result['confidence']*100:.2f}%")
# print(f"Prawdopodobieństwo melanomy: {result['raw_probability']*100:.2f}%")