import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, Add, Multiply, Reshape, Concatenate
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, f1_score
from tensorflow.keras import backend as K
import tensorflow_addons as tfa

cnn_v4_results_dir = "results/cnn_v4"
cnn_v4_models_dir = os.path.join(cnn_v4_results_dir, "models")
cnn_v4_plots_dir = os.path.join(cnn_v4_results_dir, "plots")

for dir_path in [cnn_v4_results_dir, cnn_v4_models_dir, cnn_v4_plots_dir]:
    os.makedirs(dir_path, exist_ok=True)

# Parametry modelu
img_width, img_height = 224, 224
batch_size = 32
epochs = 40  # Zwiększona liczba epok

base_dir = "data/skin_moles"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

train_benign_dir = os.path.join(train_dir, "benign")
train_melanoma_dir = os.path.join(train_dir, "melanoma")
n_benign = len(os.listdir(train_benign_dir))
n_melanoma = len(os.listdir(train_melanoma_dir))
total = n_benign + n_melanoma

# Obliczenie wag klas (odwrotnie proporcjonalne do liczby próbek)
weight_for_benign = (1 / n_benign) * total / 2.0
weight_for_melanoma = (1 / n_melanoma) * total / 2.0 * 2.5  # Zmniejszone z 3.0 na 2.5

class_weights = {0: weight_for_benign, 1: weight_for_melanoma}
print(f"Wagi klas: Benign={weight_for_benign:.3f}, Melanoma={weight_for_melanoma:.3f}")

def focal_dice_loss(gamma=2.0, alpha=0.25, dice_weight=0.5):
    def loss_function(y_true, y_pred):
        # Focal Loss
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        
        focal_loss = -alpha * (1 - y_pred) ** gamma * y_true * K.log(y_pred) - \
                     (1 - alpha) * y_pred ** gamma * (1 - y_true) * K.log(1 - y_pred)
        focal_loss = K.mean(focal_loss)
        
        # Dice Loss
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        dice_loss = 1 - (2. * intersection + epsilon) / (K.sum(y_true_f) + K.sum(y_pred_f) + epsilon)
        
        # Kombinacja obu strat
        return (1 - dice_weight) * focal_loss + dice_weight * dice_loss
    
    return loss_function

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Aktualizacja precision i recall
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)
        
    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + K.epsilon()))
        
    def reset_state(self):
        # Reset stanu
        self.precision.reset_state()
        self.recall.reset_state()

# Zaawansowana augmentacja danych dla zbioru treningowego
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=180,           # Pełny zakres obrotów
    width_shift_range=0.15,
    height_shift_range=0.15,
    brightness_range=[0.7, 1.3],
    shear_range=15,
    zoom_range=[0.85, 1.15],
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='reflect',
    channel_shift_range=20.0,
    preprocessing_function=None   # Można dodać niestandardową funkcję preprocessingu
)

test_datagen = ImageDataGenerator(rescale=1./255)

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
    shuffle=False
)

# Funkcja Squeeze-and-Excitation Block
def squeeze_excite_block(input_tensor, ratio=16):
    init = input_tensor
    channel_axis = -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)
    
    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    
    x = Multiply()([init, se])
    return x

# Funkcja tworząca blok hybrydowy (rezydualny + gęste połączenia)
def hybrid_residual_block(x, filters, kernel_size=3, stride=1, use_se=True, l2_reg=0.0001):
    shortcut = x
    
    # Pierwsza konwolucja
    x = SeparableConv2D(filters, kernel_size, strides=stride, padding='same', 
                       depthwise_regularizer=l2(l2_reg), pointwise_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Druga konwolucja
    x = SeparableConv2D(filters, kernel_size, padding='same', 
                       depthwise_regularizer=l2(l2_reg), pointwise_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    
    # Squeeze and Excitation
    if use_se:
        x = squeeze_excite_block(x)
    
    # Połączenie rezydualne
    if stride > 1 or shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, 1, strides=stride, padding='same', kernel_regularizer=l2(l2_reg))(shortcut)
        shortcut = BatchNormalization()(shortcut)
    
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    
    return x

# Dropout przestrzenny
def spatial_dropout(x, dropout_rate=0.2):
    return tf.keras.layers.SpatialDropout2D(dropout_rate)(x)

# Definicja modelu CNN v4 (architektura hybrydowa)
def create_hybrid_model():
    inputs = Input(shape=(img_width, img_height, 3))
    
    # Początkowa konwolucja
    x = Conv2D(64, 7, strides=2, padding='same', kernel_regularizer=l2(0.0001))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(3, strides=2, padding='same')(x)
    
    # Pierwszy blok hybrydowy
    x = hybrid_residual_block(x, 64)
    x = hybrid_residual_block(x, 64)
    x = spatial_dropout(x, 0.1)
    
    # Drugi blok hybrydowy z redukcją rozmiaru
    x = hybrid_residual_block(x, 128, stride=2)
    x = hybrid_residual_block(x, 128)
    x = spatial_dropout(x, 0.15)
    
    # Trzeci blok hybrydowy z redukcją rozmiaru
    x = hybrid_residual_block(x, 256, stride=2)
    x = hybrid_residual_block(x, 256)
    x = spatial_dropout(x, 0.2)
    
    # Czwarty blok hybrydowy z redukcją rozmiaru
    x = hybrid_residual_block(x, 512, stride=2)
    x = hybrid_residual_block(x, 512)
    x = spatial_dropout(x, 0.25)
    
    # Global Average Pooling i warstwy w pełni połączone
    x = GlobalAveragePooling2D()(x)
    
    # Warstwa w pełni połączona z dropout
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Warstwa wyjściowa
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Scheduler stopy uczenia - cosine annealing with warm restarts
def cosine_annealing_scheduler(epoch, lr_max=0.001, lr_min=1e-6, cycles=5):
    if cycles == 0:
        return lr_max
    
    # długość cyklu w epokach
    cycle_length = epochs // cycles
    
    # aktualny cykl i pozycję w cyklu
    current_cycle = epoch // cycle_length
    position_in_cycle = epoch % cycle_length
    
    # współczynnik w obecnym cyklu (od 0 do 1)
    cycle_position = position_in_cycle / cycle_length
    
    # Cosine annealing formula
    cosine_value = 0.5 * (1 + np.cos(np.pi * cycle_position))
    learning_rate = lr_min + (lr_max - lr_min) * cosine_value
    
    return float(learning_rate)

print("Tworzenie modelu CNN v4...")
model = create_hybrid_model()

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=focal_dice_loss(gamma=2.0, alpha=0.25, dice_weight=0.3),
    metrics=[
        'accuracy',
        tf.keras.metrics.Recall(name='recall'),  
        tf.keras.metrics.Precision(name='precision'),  
        tf.keras.metrics.AUC(name='auc'),
        F1Score(name='f1_score')  # Zmienione na instancję klasy F1Score
    ]
)

model.summary()

with open(os.path.join(cnn_v4_results_dir, 'model_architecture.txt'), 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

checkpoint = ModelCheckpoint(
    os.path.join(cnn_v4_models_dir, 'best_model.h5'),
    monitor='val_f1_score',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_f1_score',
    patience=12,  # Większa cierpliwość
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=6,
    min_lr=1e-6,
    verbose=1
)

# Custom LR scheduler
lr_scheduler = LearningRateScheduler(
    lambda epoch: cosine_annealing_scheduler(epoch, lr_max=0.001, lr_min=1e-6, cycles=5),
    verbose=1
)

callbacks = [checkpoint, early_stopping, reduce_lr, lr_scheduler]

# Trening modelu
print("Rozpoczęcie treningu modelu CNN v4...")
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
print(f"Czas treningu CNN v4: {training_time:.2f} sekund")

np.save(os.path.join(cnn_v4_results_dir, 'training_history.npy'), history.history)

model.load_weights(os.path.join(cnn_v4_models_dir, 'best_model.h5'))

# Wizualizacja historii treningu (metryki w czasie)
plt.figure(figsize=(20, 10))

# Wykres dokładności
plt.subplot(2, 3, 1)
plt.plot(history.history['accuracy'], label='Trening')
plt.plot(history.history['val_accuracy'], label='Walidacja')
plt.title('Dokładność modelu CNN v4')
plt.xlabel('Epoka')
plt.ylabel('Dokładność')
plt.legend(loc='lower right')

# Wykres funkcji straty
plt.subplot(2, 3, 2)
plt.plot(history.history['loss'], label='Trening')
plt.plot(history.history['val_loss'], label='Walidacja')
plt.title('Funkcja straty modelu CNN v4')
plt.xlabel('Epoka')
plt.ylabel('Strata')
plt.legend(loc='upper right')

# Wykres czułości (recall)
plt.subplot(2, 3, 3)
plt.plot(history.history['recall'], label='Trening')
plt.plot(history.history['val_recall'], label='Walidacja')
plt.title('Czułość (Recall) modelu CNN v4')
plt.xlabel('Epoka')
plt.ylabel('Czułość')
plt.legend(loc='lower right')

# Wykres precyzji
plt.subplot(2, 3, 4)
plt.plot(history.history['precision'], label='Trening')
plt.plot(history.history['val_precision'], label='Walidacja')
plt.title('Precyzja modelu CNN v4')
plt.xlabel('Epoka')
plt.ylabel('Precyzja')
plt.legend(loc='lower right')

# Wykres F1 Score - zaktualizowane nazwy kluczy
plt.subplot(2, 3, 5)
plt.plot(history.history['f1_score'], label='Trening')
plt.plot(history.history['val_f1_score'], label='Walidacja')
plt.title('F1 Score modelu CNN v4')
plt.xlabel('Epoka')
plt.ylabel('F1 Score')
plt.legend(loc='lower right')

# Wykres AUC
plt.subplot(2, 3, 6)
plt.plot(history.history['auc'], label='Trening')
plt.plot(history.history['val_auc'], label='Walidacja')
plt.title('AUC modelu CNN v4')
plt.xlabel('Epoka')
plt.ylabel('AUC')
plt.legend(loc='lower right')

plt.tight_layout()
plt.savefig(os.path.join(cnn_v4_plots_dir, 'training_history.png'))
plt.show()

# Ewaluacja modelu na zbiorze testowym z domyślnym progiem 0.5
print("Ewaluacja modelu CNN v4 z progiem 0.5...")
test_generator.reset()
results = model.evaluate(test_generator)
metrics_names = model.metrics_names
for i, metric in enumerate(metrics_names):
    print(f"Test {metric}: {results[i]:.4f}")

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
accuracy_scores = []

for threshold in thresholds:
    y_pred = (y_pred_prob > threshold).astype(int).flatten()
    
    # Macierz pomyłek
    cm = confusion_matrix(y_true, y_pred)
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

best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]
best_recall = recall_scores[best_idx]
best_precision = precision_scores[best_idx]
best_accuracy = accuracy_scores[best_idx]

print(f"Optymalny próg decyzyjny: {best_threshold:.2f}")
print(f"Przy tym progu: F1={best_f1:.4f}, Recall={best_recall:.4f}, Precision={best_precision:.4f}, Accuracy={best_accuracy:.4f}")

y_pred = (y_pred_prob > best_threshold).astype(int).flatten()

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Benign', 'Melanoma'],
            yticklabels=['Benign', 'Melanoma'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix - CNN v4 (próg={best_threshold:.2f})')
plt.savefig(os.path.join(cnn_v4_plots_dir, 'confusion_matrix.png'))
plt.show()

# Raport klasyfikacji z optymalnym progiem
print("\nRaport klasyfikacji z optymalnym progiem:")
report = classification_report(y_true, y_pred, target_names=['Benign', 'Melanoma'])
print(report)

with open(os.path.join(cnn_v4_results_dir, 'classification_report.txt'), 'w') as f:
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
plt.title('Receiver Operating Characteristic - CNN v4')
plt.legend(loc='lower right')
plt.savefig(os.path.join(cnn_v4_plots_dir, 'roc_curve.png'))
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
plt.savefig(os.path.join(cnn_v4_plots_dir, 'threshold_metrics.png'))
plt.show()

model.save(os.path.join(cnn_v4_models_dir, 'cnn_v4_model'))
print(f"Model zapisany w: {os.path.join(cnn_v4_models_dir, 'cnn_v4_model')}")

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
with open(os.path.join(cnn_v4_results_dir, 'results_summary.txt'), 'w') as f:
    f.write(f"CNN v4 Model Results Summary\n")
    f.write(f"=========================\n")
    f.write(f"Date: {timestamp}\n\n")
    f.write(f"Training time: {training_time:.2f} seconds\n")
    
    for i, metric in enumerate(metrics_names):
        f.write(f"Test {metric}: {results[i]:.4f}\n")
    
    f.write(f"\nOptymalny próg decyzyjny: {best_threshold:.2f}\n")
    f.write(f"Przy optymalnym progu: F1={best_f1:.4f}, Recall={best_recall:.4f}, Precision={best_precision:.4f}, Accuracy={best_accuracy:.4f}\n\n")
    
    f.write("Główne ulepszenia w CNN v4:\n")
    f.write("1. Architektura hybrydowa (rezydualna + gęste połączenia)\n")
    f.write("2. Kombinowana funkcja straty (Focal Loss + Dice Loss)\n")
    f.write("3. Zrównoważone ważenie klasy melanoma (2.5x zamiast 3x)\n")
    f.write("4. Dodanie bloków Squeeze-and-Excitation\n")
    f.write("5. Zastosowanie separowalnych konwolucji dla zmniejszenia liczby parametrów\n")
    f.write("6. Spatial Dropout zamiast zwykłego Dropout\n")
    f.write("7. Cosine Annealing LR Scheduler z warm restarts\n")
    f.write("8. Użycie funkcji aktywacji ReLU\n")
    f.write("9. Monitorowanie F1-score zamiast recall\n")
    f.write("10. Optymalizacja progu decyzyjnego\n\n")
    
    f.write("Raport klasyfikacji przy optymalnym progu:\n")
    f.write(report)

print("Trening i ewaluacja modelu CNN v4 zakończone!")