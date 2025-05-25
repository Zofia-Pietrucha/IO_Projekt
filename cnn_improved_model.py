import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, f1_score
import tensorflow as tf

cnn_improved_results_dir = "results/cnn_improved"
cnn_improved_models_dir = os.path.join(cnn_improved_results_dir, "models")
cnn_improved_plots_dir = os.path.join(cnn_improved_results_dir, "plots")

for dir_path in [cnn_improved_results_dir, cnn_improved_models_dir, cnn_improved_plots_dir]:
    os.makedirs(dir_path, exist_ok=True)

# Parametry modelu
img_width, img_height = 224, 224
batch_size = 32
epochs = 25  # Zwiększona liczba epok z marginesem na early stopping

base_dir = "data/skin_moles"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

# Liczba próbek w każdej klasie, aby określić ważenie klas
train_benign_dir = os.path.join(train_dir, "benign")
train_melanoma_dir = os.path.join(train_dir, "melanoma")
n_benign = len(os.listdir(train_benign_dir))
n_melanoma = len(os.listdir(train_melanoma_dir))
total = n_benign + n_melanoma

# Obliczenie wag klas (odwrotnie proporcjonalne do liczby próbek)
# Większq waga klasie melanoma, aby zwiększyć jej czułość
weight_for_benign = (1 / n_benign) * total / 2.0
weight_for_melanoma = (1 / n_melanoma) * total / 2.0 * 1.5  # Dodatkowe 50% wagi dla melanoma

class_weights = {0: weight_for_benign, 1: weight_for_melanoma}
print(f"Wagi klas: Benign={weight_for_benign:.3f}, Melanoma={weight_for_melanoma:.3f}")

# Ulepszony generator danych z rozszerzoną augmentacją dla zbioru treningowego
train_datagen = ImageDataGenerator(
    rescale=1./255,             # normalizacja
    rotation_range=90,          # pełny zakres obrotów - znamiona mogą być w dowolnej orientacji
    brightness_range=[0.8, 1.2],# zmiana jasności - symulacja różnych warunków oświetlenia
    horizontal_flip=True,       # losowe odbicie w poziomie
    vertical_flip=True,         # losowe odbicie w pionie
    zoom_range=[0.9, 1.1],      # niewielki zoom in/out
    fill_mode='reflect',        # lepsze wypełnianie niż 'nearest'
    width_shift_range=0.1,      # niewielkie przesunięcia poziome
    height_shift_range=0.1      # niewielkie przesunięcia pionowe
)

# Generator dla danych testowych (tylko normalizacja)
test_datagen = ImageDataGenerator(rescale=1./255)

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
    class_mode='binary',
    shuffle=False
)

# Funkcja harmonogramu cyklicznej stopy uczenia
def cyclic_learning_rate(epochs_per_step=5, min_lr=1e-6, max_lr=1e-4):
    """Cykliczna stopa uczenia dla bardziej stabilnego uczenia."""
    def schedule(epoch):
        cycle = np.floor(1 + epoch / (2 * epochs_per_step))
        x = np.abs(epoch / epochs_per_step - 2 * cycle + 1)
        lr = min_lr + (max_lr - min_lr) * max(0, (1 - x))
        return lr
    return LearningRateScheduler(schedule)

model = Sequential([
    # Pierwszy blok konwolucyjny
    Conv2D(32, (3, 3), activation='relu', padding='same', 
           kernel_regularizer=l2(0.001),  # Dodana regularyzacja L2
           input_shape=(img_width, img_height, 3)),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same',
           kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),  # Zwiększony dropout
    
    # Drugi blok konwolucyjny
    Conv2D(64, (3, 3), activation='relu', padding='same',
           kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same',
           kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),
    
    # Trzeci blok konwolucyjny
    Conv2D(128, (3, 3), activation='relu', padding='same',
           kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu', padding='same',
           kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),
    
    # Czwarty blok konwolucyjny (dodatkowy)
    Conv2D(256, (3, 3), activation='relu', padding='same',
           kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Conv2D(256, (3, 3), activation='relu', padding='same',
           kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),
    
    # Spłaszczenie i warstwy w pełni połączone
    Flatten(),
    Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),  # Dodatkowa warstwa
    BatchNormalization(),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Wyjście binarne: 0 - benign, 1 - melanoma
])

# Kompilacja modelu ze zmodyfikowaną funkcją straty i dodatkowymi metrykami
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.1),  # Dodane label smoothing
    metrics=[
        'accuracy',
        tf.keras.metrics.Recall(name='recall'),  # Czułość (recall) - szczególnie ważna dla melanoma
        tf.keras.metrics.Precision(name='precision'),  # Precyzja
        tf.keras.metrics.AUC(name='auc')  # Obszar pod krzywą ROC
    ]
)

model.summary()

with open(os.path.join(cnn_improved_results_dir, 'model_architecture.txt'), 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

checkpoint = ModelCheckpoint(
    os.path.join(cnn_improved_models_dir, 'best_model.h5'),
    monitor='val_recall',  # Monitorujemy czułość (recall) zamiast dokładności
    save_best_only=True,
    mode='max',  # Maksymalizujemy recall
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_recall',
    patience=8,         # Zwiększona cierpliwość
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

# cykliczny harmonogram stopy uczenia
cyclic_lr = cyclic_learning_rate(epochs_per_step=5, min_lr=1e-6, max_lr=1e-4)

callbacks = [checkpoint, early_stopping, reduce_lr, cyclic_lr]

# Trening modelu
print("Rozpoczęcie treningu ulepszonego modelu CNN...")
start_time = time.time()

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size,
    callbacks=callbacks,
    class_weight=class_weights  # Dodane ważenie klas
)

training_time = time.time() - start_time
print(f"Czas treningu ulepszonego CNN: {training_time:.2f} sekund")

np.save(os.path.join(cnn_improved_results_dir, 'training_history.npy'), history.history)

model.load_weights(os.path.join(cnn_improved_models_dir, 'best_model.h5'))

plt.figure(figsize=(16, 6))
plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Trening')
plt.plot(history.history['val_accuracy'], label='Walidacja')
plt.title('Dokładność modelu CNN')
plt.xlabel('Epoka')
plt.ylabel('Dokładność')
plt.legend(loc='lower right')

# Wizualizacja historii treningu (funkcja straty)
plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label='Trening')
plt.plot(history.history['val_loss'], label='Walidacja')
plt.title('Funkcja straty modelu CNN')
plt.xlabel('Epoka')
plt.ylabel('Strata')
plt.legend(loc='upper right')

# Wizualizacja historii treningu (czułość - recall)
plt.subplot(1, 3, 3)
plt.plot(history.history['recall'], label='Trening')
plt.plot(history.history['val_recall'], label='Walidacja')
plt.title('Czułość (Recall) modelu CNN')
plt.xlabel('Epoka')
plt.ylabel('Czułość')
plt.legend(loc='lower right')

plt.tight_layout()
plt.savefig(os.path.join(cnn_improved_plots_dir, 'training_history.png'))
plt.show()

# Ewaluacja modelu na zbiorze testowym z domyślnym progiem 0.5
print("Ewaluacja modelu CNN z progiem 0.5...")
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

# optymalny próg decyzyjny maksymalizujący F1 dla melanoma
print("\nSzukanie optymalnego progu decyzyjnego dla melanomy...")
thresholds = np.arange(0.3, 0.7, 0.01)
f1_scores = []
recall_scores = []
precision_scores = []

for threshold in thresholds:
    y_pred = (y_pred_prob > threshold).astype(int).flatten()
    # F1 dla klasy melanoma (pozytywnej)
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_positives = np.sum((y_true == 0) & (y_pred == 1))
    false_negatives = np.sum((y_true == 1) & (y_pred == 0))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    f1_scores.append(f1)
    recall_scores.append(recall)
    precision_scores.append(precision)

best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]
best_recall = recall_scores[best_idx]
best_precision = precision_scores[best_idx]

print(f"Optymalny próg decyzyjny: {best_threshold:.2f}")
print(f"Przy tym progu: F1={best_f1:.4f}, Recall={best_recall:.4f}, Precision={best_precision:.4f}")

y_pred = (y_pred_prob > best_threshold).astype(int).flatten()

# Macierz pomyłek z optymalnym progiem
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Benign', 'Melanoma'],
            yticklabels=['Benign', 'Melanoma'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix - Ulepszony CNN (próg={best_threshold:.2f})')
plt.savefig(os.path.join(cnn_improved_plots_dir, 'confusion_matrix.png'))
plt.show()

# Raport klasyfikacji z optymalnym progiem
print("\nRaport klasyfikacji z optymalnym progiem:")
report = classification_report(y_true, y_pred, target_names=['Benign', 'Melanoma'])
print(report)

with open(os.path.join(cnn_improved_results_dir, 'classification_report.txt'), 'w') as f:
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
plt.title('Receiver Operating Characteristic - Ulepszony CNN')
plt.legend(loc='lower right')
plt.savefig(os.path.join(cnn_improved_plots_dir, 'roc_curve.png'))
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
plt.savefig(os.path.join(cnn_improved_plots_dir, 'threshold_metrics.png'))
plt.show()

model.save(os.path.join(cnn_improved_models_dir, 'cnn_improved_model'))
print(f"Model zapisany w: {os.path.join(cnn_improved_models_dir, 'cnn_improved_model')}")

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
with open(os.path.join(cnn_improved_results_dir, 'results_summary.txt'), 'w') as f:
    f.write(f"Ulepszony CNN Model Results Summary\n")
    f.write(f"================================\n")
    f.write(f"Date: {timestamp}\n\n")
    f.write(f"Training time: {training_time:.2f} seconds\n")
    f.write(f"Test accuracy: {results[1]:.4f}\n")
    f.write(f"Test recall: {results[2]:.4f}\n")
    f.write(f"Test precision: {results[3]:.4f}\n")
    f.write(f"Test AUC: {results[4]:.4f}\n\n")
    f.write(f"Optymalny próg decyzyjny: {best_threshold:.2f}\n")
    f.write(f"Przy optymalnym progu: F1={best_f1:.4f}, Recall={best_recall:.4f}, Precision={best_precision:.4f}\n\n")
    f.write("Ulepszenia w stosunku do podstawowego modelu CNN:\n")
    f.write("1. Ważenie klas dla lepszego wykrywania melanomy\n")
    f.write("2. Rozszerzona augmentacja danych\n")
    f.write("3. Dodana regularyzacja L2\n")
    f.write("4. Label smoothing w funkcji straty\n")
    f.write("5. Cykliczna stopa uczenia\n")
    f.write("6. Optymalizacja progu decyzyjnego\n")
    f.write("7. Monitorowanie recall zamiast accuracy\n")
    f.write("8. Głębsza architektura z dodatkowym blokiem konwolucyjnym\n\n")
    f.write("Raport klasyfikacji przy optymalnym progu:\n")
    f.write(report)

print("Trening i ewaluacja ulepszonego modelu CNN zakończone!")

from tensorflow.keras.preprocessing.image import load_img, img_to_array

def predict_skin_lesion(image_path, model, threshold=0.5):
    """Predykcja dla pojedynczego obrazu znamienia skórnego."""
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
