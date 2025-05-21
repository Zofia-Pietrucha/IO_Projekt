# efficientnet_model.py
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import AdamW

# Struktura folderów
effnet_results_dir = "results/efficientnet"
effnet_models_dir = os.path.join(effnet_results_dir, "models")
effnet_plots_dir = os.path.join(effnet_results_dir, "plots")

# Tworzenie folderów wynikowych
os.makedirs(effnet_models_dir, exist_ok=True)
os.makedirs(effnet_plots_dir, exist_ok=True)

# Parametry modelu
IMG_SIZE = (384, 384)  # Większy rozmiar dla EfficientNet
BATCH_SIZE = 32
EPOCHS = 30
INIT_LR = 1e-4
WEIGHT_DECAY = 1e-5

# Ścieżki do danych
base_dir = "data/skin_moles"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

# Obliczanie wag klas
def calculate_class_weights():
    train_benign = len(os.listdir(os.path.join(train_dir, "benign")))
    train_melanoma = len(os.listdir(os.path.join(train_dir, "melanoma")))
    total = train_benign + train_melanoma
    weight_for_benign = (1 / train_benign) * total / 2.0
    weight_for_melanoma = (1 / train_melanoma) * total / 2.0 * 3.0
    return {0: weight_for_benign, 1: weight_for_melanoma}

class_weights = calculate_class_weights()

# Zaawansowana augmentacja
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='reflect',
    brightness_range=[0.8, 1.2],
    channel_shift_range=30.0
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Generatory danych
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# Metryka F1Score
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)
        
    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + K.epsilon()))
        
    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

# Model EfficientNetV2 z dostosowaną głową klasyfikacyjną
def build_effnet_model():
    base_model = applications.EfficientNetV2L(
        include_top=False,
        weights='imagenet',
        input_shape=IMG_SIZE + (3,),
        pooling='avg'
    )
    
    # Zamrożenie warstw bazowych
    base_model.trainable = False
    
    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = applications.efficientnet_v2.preprocess_input(inputs)
    x = base_model(x)
    
    # Głowa klasyfikacyjna
    x = layers.Dense(512, activation='relu', kernel_regularizer=l2(WEIGHT_DECAY))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(128, activation='relu', kernel_regularizer=l2(WEIGHT_DECAY))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs, outputs)
    
    optimizer = AdamW(
    learning_rate=INIT_LR,
    weight_decay=WEIGHT_DECAY
)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_focal_crossentropy',
        metrics=[
            'accuracy',
            F1Score(),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    return model

# Callbacki
callbacks = [
    EarlyStopping(patience=8, monitor='val_f1_score', mode='max', restore_best_weights=True),
    ModelCheckpoint(os.path.join(effnet_models_dir, 'best_model.h5'), save_best_only=True, monitor='val_f1_score'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=4, min_lr=1e-6)
]

# Inicjalizacja modelu
model = build_effnet_model()
model.summary()

# Trening
start_time = time.time()
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    callbacks=callbacks,
    class_weight=class_weights
)
training_time = time.time() - start_time

# Zapisywanie wyników
model.save(os.path.join(effnet_models_dir, 'efficientnet_model'))
np.save(os.path.join(effnet_results_dir, 'history.npy'), history.history)

# Ewaluacja
test_generator.reset()
y_pred = model.predict(test_generator)
y_probs = y_pred.flatten()
y_true = test_generator.classes
y_pred = (y_probs > 0.5).astype(int)

# Raport klasyfikacji
print(classification_report(y_true, y_pred, target_names=['Benign', 'Melanoma']))

# Wizualizacja wyników
plt.figure(figsize=(20, 10))
metrics = ['loss', 'accuracy', 'f1_score', 'auc', 'precision', 'recall']

for i, metric in enumerate(metrics, 1):
    plt.subplot(2, 3, i)
    plt.plot(history.history[metric], label='Train')
    plt.plot(history.history[f'val_{metric}'], label='Validation')
    plt.title(f'{metric.upper()} Evolution')
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(effnet_plots_dir, 'training_metrics.png'))
plt.show()

# Krzywa ROC
fpr, tpr, _ = roc_curve(y_true, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - EfficientNetV2')
plt.legend()
plt.savefig(os.path.join(effnet_plots_dir, 'roc_curve.png'))
plt.show()

print(f"Training completed in {training_time:.2f} seconds")