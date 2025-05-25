import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

cnn_results_dir = "results/cnn"
cnn_models_dir = os.path.join(cnn_results_dir, "models")
cnn_plots_dir = os.path.join(cnn_results_dir, "plots")

for dir_path in [cnn_results_dir, cnn_models_dir, cnn_plots_dir]:
    os.makedirs(dir_path, exist_ok=True)

# Parametry modelu
img_width, img_height = 224, 224
batch_size = 32
epochs = 20

base_dir = "data/skin_moles"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

# Generatory danych z augmentacją dla zbioru treningowego
train_datagen = ImageDataGenerator(
    rescale=1./255,             # normalizacja
    rotation_range=20,          # ostrożny losowy obrót
    horizontal_flip=True,       # losowe odbicie w poziomie
    vertical_flip=True,         # losowe odbicie w pionie
    zoom_range=0.1,             # delikatne przybliżenie
    fill_mode='nearest'         # strategia wypełniania nowych pikseli
)

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

model = Sequential([
    # Pierwszy blok konwolucyjny, proste wzorce (brzegi, linie)
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(img_width, img_height, 3)),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    # Drugi blok konwolucyjny, średniej złożoności (zakrzywienia, tekstury)  
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    # Trzeci blok konwolucyjny, złożone wzorce (kształty znamion)
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    # Spłaszczenie i warstwy w pełni połączone
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Wyjście binarne: 0 - benign, 1 - melanoma
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

with open(os.path.join(cnn_results_dir, 'model_architecture.txt'), 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

checkpoint = ModelCheckpoint(
    os.path.join(cnn_models_dir, 'best_model.h5'),
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

callbacks = [checkpoint, early_stopping, reduce_lr]

print("Rozpoczęcie treningu modelu CNN...")
start_time = time.time()

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size,
    callbacks=callbacks
)

training_time = time.time() - start_time
print(f"Czas treningu CNN: {training_time:.2f} sekund")

np.save(os.path.join(cnn_results_dir, 'training_history.npy'), history.history)

model.load_weights(os.path.join(cnn_models_dir, 'best_model.h5'))

# Wizualizacja historii treningu (dokładność)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Trening')
plt.plot(history.history['val_accuracy'], label='Walidacja')
plt.title('Dokładność modelu CNN')
plt.xlabel('Epoka')
plt.ylabel('Dokładność')
plt.legend(loc='lower right')

# Wizualizacja historii treningu (funkcja straty)
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Trening')
plt.plot(history.history['val_loss'], label='Walidacja')
plt.title('Funkcja straty modelu CNN')
plt.xlabel('Epoka')
plt.ylabel('Strata')
plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(cnn_plots_dir, 'training_history.png'))
plt.show()

# Ewaluacja modelu na zbiorze testowym
print("Ewaluacja modelu CNN...")
test_generator.reset()
results = model.evaluate(test_generator)
print(f"Test loss: {results[0]:.4f}")
print(f"Test accuracy: {results[1]:.4f}")

# Przewidywania na zbiorze testowym
test_generator.reset()
y_pred_prob = model.predict(test_generator)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# rzeczywiste etykiety
y_true = test_generator.classes

# Macierz pomyłek
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Benign', 'Melanoma'],
            yticklabels=['Benign', 'Melanoma'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - CNN')
plt.savefig(os.path.join(cnn_plots_dir, 'confusion_matrix.png'))
plt.show()

print("\nRaport klasyfikacji:")
report = classification_report(y_true, y_pred, target_names=['Benign', 'Melanoma'])
print(report)

with open(os.path.join(cnn_results_dir, 'classification_report.txt'), 'w') as f:
    f.write(report)

# Krzywa ROC
fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - CNN')
plt.legend(loc='lower right')
plt.savefig(os.path.join(cnn_plots_dir, 'roc_curve.png'))
plt.show()

model.save(os.path.join(cnn_models_dir, 'cnn_model'))
print(f"Model CNN zapisany w: {os.path.join(cnn_models_dir, 'cnn_model')}")

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
with open(os.path.join(cnn_results_dir, 'results_summary.txt'), 'w') as f:
    f.write(f"CNN Model Results Summary\n")
    f.write(f"======================\n")
    f.write(f"Date: {timestamp}\n\n")
    f.write(f"Training time: {training_time:.2f} seconds\n")
    f.write(f"Test accuracy: {results[1]:.4f}\n")
    f.write(f"Test loss: {results[0]:.4f}\n")
    f.write(f"ROC AUC: {roc_auc:.4f}\n\n")
    f.write("Architecture:\n")
    model.summary(print_fn=lambda x: f.write(x + '\n'))
    f.write("\nClassification Report:\n")
    f.write(report)

print("Trening i ewaluacja modelu CNN zakończone!")