import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

tl_results_dir = "results/transfer_learning"
tl_models_dir = os.path.join(tl_results_dir, "models")
tl_plots_dir = os.path.join(tl_results_dir, "plots")

for dir_path in [tl_results_dir, tl_models_dir, tl_plots_dir]:
    os.makedirs(dir_path, exist_ok=True)

# Parametry modelu
img_width, img_height = 224, 224  # MobileNetV2 wymaga tego rozmiaru
batch_size = 32
epochs = 20

base_dir = "data/skin_moles"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

# Generatory danych z augmentacją dla zbioru treningowego
# Używamy zmodyfikowanej augmentacji, która nie przycina obrazów
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

print("Wczytywanie modelu bazowego MobileNetV2...")
base_model = MobileNetV2(
    weights='imagenet',        # Wczytanie wag z pre-treningu na ImageNet
    include_top=False,         # Bez warstw w pełni połączonych na wierzchu
    input_shape=(img_width, img_height, 3)
)

# Zamrożenie warstw bazowego modelu
for layer in base_model.layers:
    layer.trainable = False

# Dodanie nowych warstw na wierzchu modelu
x = base_model.output
x = GlobalAveragePooling2D()(x)      # Globalne pooling cech
x = Dense(256, activation='relu')(x) # Warstwa w pełni połączona
x = Dropout(0.5)(x)                  # Dropout dla zapobiegania przeuczeniu
predictions = Dense(1, activation='sigmoid')(x) # Wyjściowa warstwa dla klasyfikacji binarnej

# Stworzenie nowego modelu
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

with open(os.path.join(tl_results_dir, 'model_architecture.txt'), 'w', encoding='utf-8') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

checkpoint = ModelCheckpoint(
    os.path.join(tl_models_dir, 'best_model_stage1.h5'),
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

# Trening modelu (tylko nowe warstwy)
print("Rozpoczęcie treningu modelu (Etap 1 - tylko nowe warstwy)...")
start_time = time.time()

history_stage1 = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size,
    callbacks=callbacks
)

# Zapisz historię treningu etapu 1
np.save(os.path.join(tl_results_dir, 'training_history_stage1.npy'), history_stage1.history)

# Etap 2: Fine-tuning - odmrożenie kilku ostatnich warstw modelu bazowego
print("\nRozpoczęcie etapu 2 - fine-tuning...")

# Wczytanie najlepszego modelu z etapu 1
model.load_weights(os.path.join(tl_models_dir, 'best_model_stage1.h5'))

# Odmrożenie kilku ostatnich warstw
for layer in base_model.layers[-20:]:  # 20 ostatnich warstw
    layer.trainable = True

# Rekompilacja modelu z mniejszą stopą uczenia
model.compile(
    optimizer=Adam(learning_rate=0.00001),  # 10x mniejsza stopa uczenia
    loss='binary_crossentropy',
    metrics=['accuracy']
)

checkpoint = ModelCheckpoint(
    os.path.join(tl_models_dir, 'best_model_stage2.h5'),
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

callbacks = [checkpoint, early_stopping, reduce_lr]

# Trening modelu (fine-tuning)
history_stage2 = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs // 2,  # Mniej epok dla fine-tuningu
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size,
    callbacks=callbacks
)

training_time = time.time() - start_time
print(f"Całkowity czas treningu: {training_time:.2f} sekund")

np.save(os.path.join(tl_results_dir, 'training_history_stage2.npy'), history_stage2.history)

model.load_weights(os.path.join(tl_models_dir, 'best_model_stage2.h5'))

combined_accuracy = history_stage1.history['accuracy'] + history_stage2.history['accuracy']
combined_val_accuracy = history_stage1.history['val_accuracy'] + history_stage2.history['val_accuracy']
combined_loss = history_stage1.history['loss'] + history_stage2.history['loss']
combined_val_loss = history_stage1.history['val_loss'] + history_stage2.history['val_loss']

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(combined_accuracy, label='Trening')
plt.plot(combined_val_accuracy, label='Walidacja')
plt.axvline(x=len(history_stage1.history['accuracy']), color='r', linestyle='--', label='Etap 2')
plt.title('Dokładność modelu - Transfer Learning')
plt.xlabel('Epoka')
plt.ylabel('Dokładność')
plt.legend(loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(combined_loss, label='Trening')
plt.plot(combined_val_loss, label='Walidacja')
plt.axvline(x=len(history_stage1.history['loss']), color='r', linestyle='--', label='Etap 2')
plt.title('Funkcja straty - Transfer Learning')
plt.xlabel('Epoka')
plt.ylabel('Strata')
plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(tl_plots_dir, 'training_history.png'))
plt.show()

print("Ewaluacja modelu...")
test_generator.reset()
results = model.evaluate(test_generator)
print(f"Test loss: {results[0]:.4f}")
print(f"Test accuracy: {results[1]:.4f}")

test_generator.reset()
y_pred_prob = model.predict(test_generator)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

y_true = test_generator.classes

# Macierz pomyłek
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Benign', 'Melanoma'],
            yticklabels=['Benign', 'Melanoma'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Transfer Learning (MobileNetV2)')
plt.savefig(os.path.join(tl_plots_dir, 'confusion_matrix.png'))
plt.show()

print("\nRaport klasyfikacji:")
report = classification_report(y_true, y_pred, target_names=['Benign', 'Melanoma'])
print(report)

with open(os.path.join(tl_results_dir, 'classification_report.txt'), 'w', encoding='utf-8') as f:
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
plt.title('Receiver Operating Characteristic - Transfer Learning (MobileNetV2)')
plt.legend(loc='lower right')
plt.savefig(os.path.join(tl_plots_dir, 'roc_curve.png'))
plt.show()

model.save(os.path.join(tl_models_dir, 'mobilenetv2_model'))
print(f"Model zapisany w: {os.path.join(tl_models_dir, 'mobilenetv2_model')}")

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
with open(os.path.join(tl_results_dir, 'results_summary.txt'), 'w', encoding='utf-8') as f:
    f.write(f"Transfer Learning (MobileNetV2) Results Summary\n")
    f.write(f"==========================================\n")
    f.write(f"Date: {timestamp}\n\n")
    f.write(f"Training time: {training_time:.2f} seconds\n")
    f.write(f"Test accuracy: {results[1]:.4f}\n")
    f.write(f"Test loss: {results[0]:.4f}\n")
    f.write(f"ROC AUC: {roc_auc:.4f}\n\n")
    f.write("Architecture:\n")
    f.write("Base model: MobileNetV2 (pre-trained on ImageNet)\n")
    f.write("Custom top layers: GlobalAveragePooling2D -> Dense(256) -> Dropout(0.5) -> Dense(1, sigmoid)\n")
    f.write("\nTraining strategy:\n")
    f.write("Stage 1: Train only custom top layers\n")
    f.write("Stage 2: Fine-tune last 20 layers of MobileNetV2\n")
    f.write("\nClassification Report:\n")
    f.write(report)

print("Trening i ewaluacja modelu Transfer Learning zakończone!")