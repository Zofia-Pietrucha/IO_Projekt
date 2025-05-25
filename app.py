import os
import numpy as np
import tensorflow as tf
import pickle
import cv2
from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from scipy.stats import entropy

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Maksymalny rozmiar: 16MB

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Ścieżki do modelu hybrydowego
HYBRID_MODELS_DIR = 'results/hybrid_tl_xgb/models'
FEATURE_EXTRACTOR_PATH = os.path.join(HYBRID_MODELS_DIR, 'feature_extractor')
XGBOOST_MODEL_PATH = os.path.join(HYBRID_MODELS_DIR, 'xgboost_hybrid.pkl')
SCALER_PATH = os.path.join(HYBRID_MODELS_DIR, 'scaler.pkl')
CONFIG_PATH = os.path.join(HYBRID_MODELS_DIR, 'model_config.pkl')

# Globalne zmienne dla modeli
feature_extractor = None
xgb_model = None
scaler = None
model_config = None

# Załadowanie modeli
try:
    print("Ładowanie modelu hybrydowego...")
    
    # feature extractor (MobileNetV2)
    if os.path.exists(FEATURE_EXTRACTOR_PATH):
        feature_extractor = tf.keras.models.load_model(FEATURE_EXTRACTOR_PATH)
        print("✅ Feature extractor załadowany")
    else:
        print(f"❌ Nie znaleziono feature extractor: {FEATURE_EXTRACTOR_PATH}")
    
    # XGBoost model
    if os.path.exists(XGBOOST_MODEL_PATH):
        with open(XGBOOST_MODEL_PATH, 'rb') as f:
            xgb_model = pickle.load(f)
        print("✅ XGBoost model załadowany")
    else:
        print(f"❌ Nie znaleziono XGBoost model: {XGBOOST_MODEL_PATH}")
    
    # Załaduj scaler
    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        print("✅ Scaler załadowany")
    else:
        print(f"❌ Nie znaleziono scaler: {SCALER_PATH}")
    
    # konfigurację modelu
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'rb') as f:
            model_config = pickle.load(f)
        print("✅ Konfiguracja modelu załadowana")
        print(f"   Optymalny próg: {model_config.get('best_threshold', 0.5)}")
    else:
        print(f"❌ Nie znaleziono konfiguracji: {CONFIG_PATH}")
        model_config = {'best_threshold': 0.49}  # Domyślny próg
    
    if all([feature_extractor, xgb_model, scaler]):
        print("🎉 Model hybrid_tl_xgb załadowany pomyślnie!")
    else:
        print("⚠️ Nie wszystkie komponenty modelu zostały załadowane")
        
except Exception as e:
    print(f"❌ Błąd podczas ładowania modelu: {e}")
    feature_extractor = None
    xgb_model = None
    scaler = None
    model_config = {'best_threshold': 0.49}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_traditional_features(image_path):
    """Ekstrakcja tradycyjnych cech obrazu (identyczna jak w modelu hybrydowym)."""
    try:
        img = load_img(image_path, target_size=(224, 224))
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
        print(f"Błąd podczas ekstrakcji tradycyjnych cech z {image_path}: {e}")
        return np.zeros(50)  # Domyślnie 50 cech

def extract_hybrid_features(image_path):
    """Ekstraktuje zarówno głębokie cechy (MobileNetV2) jak i tradycyjne cechy."""
    try:
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)
        
        # Preprocess dla MobileNetV2
        img_mobilenet = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        img_batch = np.expand_dims(img_mobilenet, axis=0)
        
        # Ekstraktuj głębokie cechy
        deep_features = feature_extractor.predict(img_batch, verbose=0)[0]
        
        # Ekstraktuj tradycyjne cechy
        traditional_features = extract_traditional_features(image_path)
        
        # Połącz cechy
        combined_features = np.concatenate([deep_features, traditional_features])
        
        return combined_features
        
    except Exception as e:
        print(f"Błąd podczas ekstrakcji hybrydowych cech z {image_path}: {e}")
        return np.zeros(1280 + 50)  # MobileNetV2 (1280) + tradycyjne cechy (~50)

def predict_image(image_path):
    """Klasyfikacja znamienia skórnego z pliku obrazu przy użyciu modelu hybrydowego."""
    try:
        if not all([feature_extractor, xgb_model, scaler]):
            return {"error": "Model nie został poprawnie załadowany"}
        
        # Pobierz optymalny próg
        threshold = model_config.get('best_threshold', 0.49)
        
        # Ekstrakcja hybrydowych cech
        features = extract_hybrid_features(image_path)
        if features is None:
            return {"error": "Błąd podczas ekstrakcji cech z obrazu"}
        
        # Standaryzacja cech
        features_scaled = scaler.transform([features])
        
        # Predykcja
        probability = xgb_model.predict_proba(features_scaled)[0, 1]
        
        # Zastosuj optymalny próg decyzyjny
        result = "Melanoma" if probability > threshold else "Benign"
        confidence = float(probability) if result == "Melanoma" else float(1 - probability)
        
        return {
            "prediction": result,
            "confidence": confidence * 100,
            "raw_probability": float(probability) * 100,
            "threshold_used": threshold,
            "model_type": "Hybrid TL+XGBoost"
        }
    except Exception as e:
        print(f"Błąd podczas predykcji: {e}")
        return {"error": str(e)}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'Nie przesłano pliku'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Nie wybrano pliku'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        if not all([feature_extractor, xgb_model, scaler]):
            return jsonify({'error': 'Model hybrydowy nie został poprawnie załadowany'}), 500

        result = predict_image(filepath)
        
        if 'error' in result:
            return jsonify({'error': result['error']}), 500
        
        if result['prediction'] == 'Melanoma':
            result['message'] = "Wykryto cechy charakterystyczne dla czerniaka. Pilnie skonsultuj się z dermatologiem."
            result['risk_level'] = "high"
        else:
            result['message'] = "Obraz klasyfikowany jako znamię łagodne. Zalecamy jednak regularną obserwację."
            result['risk_level'] = "low"
        
        result['image_path'] = filename
        
        return jsonify(result)
    
    return jsonify({'error': 'Niedozwolony format pliku'}), 400

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/health')
def health_check():
    """Endpoint do sprawdzania stanu aplikacji."""
    status = {
        'status': 'ok',
        'feature_extractor_loaded': feature_extractor is not None,
        'xgboost_model_loaded': xgb_model is not None,
        'scaler_loaded': scaler is not None,
        'config_loaded': model_config is not None,
        'optimal_threshold': model_config.get('best_threshold', 'unknown') if model_config else 'unknown'
    }
    return jsonify(status)

if __name__ == '__main__':
    app.run(debug=True)