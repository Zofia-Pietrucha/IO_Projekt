import os
import numpy as np
import tensorflow as tf
import pickle  # Dodaj import dla obsługi plików .pkl
import cv2  # Potrzebne do przetwarzania obrazów
from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Maksymalny rozmiar: 16MB

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Ścieżka do modelu XGBoost
MODEL_PATH = 'results/xgboost/models/xgboost_model.pkl'

# Załadowanie modelu XGBoost
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("Model XGBoost załadowany pomyślnie!")
except Exception as e:
    print(f"Błąd podczas ładowania modelu: {e}")
    model = None

# Optymalny próg decyzyjny (ustalony podczas treningu)
OPTIMAL_THRESHOLD = 0.49  # Zmieniony próg z 0.5 na 0.49 zgodnie z analizą

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Funkcja do ekstrakcji cech z obrazu - MUSI być identyczna jak podczas treningu
def extract_features(image_path):
    """Ekstrakcja cech ze zdjęcia znamienia skórnego dla modelu XGBoost."""
    try:
        # Wczytaj obraz
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))  # Zmień rozmiar do standardowego
        img_array = img / 255.0  # Normalizacja
        
        # Rozdziel kanały RGB
        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
        
        # Konwersja do HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) / 255.0
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        
        # Podstawowe statystyki RGB
        r_mean, g_mean, b_mean = np.mean(r), np.mean(g), np.mean(b)
        r_std, g_std, b_std = np.std(r), np.std(g), np.std(b)
        
        # Podstawowe statystyki HSV
        h_mean, s_mean, v_mean = np.mean(h), np.mean(s), np.mean(v)
        h_std, s_std, v_std = np.std(h), np.std(s), np.std(v)
        
        # Histogramy RGB i HSV (15 przedziałów)
        r_hist = np.histogram(r, bins=15, range=(0, 1))[0] / r.size
        g_hist = np.histogram(g, bins=15, range=(0, 1))[0] / g.size
        b_hist = np.histogram(b, bins=15, range=(0, 1))[0] / b.size
        
        h_hist = np.histogram(h, bins=15, range=(0, 1))[0] / h.size
        s_hist = np.histogram(s, bins=15, range=(0, 1))[0] / s.size
        v_hist = np.histogram(v, bins=15, range=(0, 1))[0] / v.size
        
        # Asymetria i kurtoza dla RGB
        r_skew = np.mean(((r - r_mean) / r_std)**3) if r_std > 0 else 0
        g_skew = np.mean(((g - g_mean) / g_std)**3) if g_std > 0 else 0
        b_skew = np.mean(((b - b_mean) / b_std)**3) if b_std > 0 else 0
        
        r_kurt = np.mean(((r - r_mean) / r_std)**4) if r_std > 0 else 0
        g_kurt = np.mean(((g - g_mean) / g_std)**4) if g_std > 0 else 0
        b_kurt = np.mean(((b - b_mean) / b_std)**4) if b_std > 0 else 0
        
        # Asymetria i kurtoza dla HSV
        h_skew = np.mean(((h - h_mean) / h_std)**3) if h_std > 0 else 0
        s_skew = np.mean(((s - s_mean) / s_std)**3) if s_std > 0 else 0
        v_skew = np.mean(((v - v_mean) / v_std)**3) if v_std > 0 else 0
        
        h_kurt = np.mean(((h - h_mean) / h_std)**4) if h_std > 0 else 0
        s_kurt = np.mean(((s - s_mean) / s_std)**4) if s_std > 0 else 0
        v_kurt = np.mean(((v - v_mean) / v_std)**4) if v_std > 0 else 0
        
        # Cechy asymetrii znamienia
        height, width = r.shape
        left_half = img_array[:, :width//2, :]
        right_half = img_array[:, width//2:, :]
        top_half = img_array[:height//2, :, :]
        bottom_half = img_array[height//2:, :, :]
        
        asymmetry_h = np.mean(np.abs(np.mean(left_half, axis=(0,1)) - np.mean(right_half, axis=(0,1))))
        asymmetry_v = np.mean(np.abs(np.mean(top_half, axis=(0,1)) - np.mean(bottom_half, axis=(0,1))))
        
        # Cechy brzegów i tekstury
        gray = np.mean(img_array, axis=2)
        gray_uint8 = (gray * 255).astype(np.uint8)
        
        # Gradienty Sobela
        sobel_h = cv2.Sobel(gray_uint8, cv2.CV_64F, 1, 0, ksize=3)
        sobel_v = cv2.Sobel(gray_uint8, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobel_h**2 + sobel_v**2)
        
        gradient_mean = np.mean(magnitude)
        gradient_std = np.std(magnitude)
        gradient_max = np.max(magnitude)
        
        # Tekstura - Laplacian
        laplacian = cv2.Laplacian(gray_uint8, cv2.CV_64F)
        laplacian_mean = np.mean(np.abs(laplacian))
        laplacian_std = np.std(laplacian)
        
        # Entropie
        r_entropy = entropy(r)
        g_entropy = entropy(g)
        b_entropy = entropy(b)
        gray_entropy = entropy(gray)
        
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
        
        # Połącz wszystkie cechy w jeden wektor
        features = np.concatenate([
            # Statystyki RGB
            [r_mean, g_mean, b_mean, r_std, g_std, b_std],
            # Statystyki HSV
            [h_mean, s_mean, v_mean, h_std, s_std, v_std],
            # Asymetria i kurtoza RGB
            [r_skew, g_skew, b_skew, r_kurt, g_kurt, b_kurt],
            # Asymetria i kurtoza HSV
            [h_skew, s_skew, v_skew, h_kurt, s_kurt, v_kurt],
            # Cechy asymetrii
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
    except Exception as e:
        print(f"Błąd podczas ekstrakcji cech: {e}")
        return None

# Funkcja pomocnicza do obliczania entropii
def entropy(img_channel):
    """Oblicza entropię dla danego kanału obrazu."""
    hist, _ = np.histogram(img_channel, bins=25, range=(0, 1), density=True)
    hist = hist + 1e-10  # Dodaj małą wartość, aby uniknąć log(0)
    return -np.sum(hist * np.log2(hist))

def predict_image(image_path):
    """Klasyfikacja znamienia skórnego z pliku obrazu przy użyciu modelu XGBoost."""
    try:
        # Ekstrakcja cech
        features = extract_features(image_path)
        if features is None:
            return {"error": "Błąd podczas ekstrakcji cech z obrazu"}
        
        features = np.array([features])  # Przekształć na format 2D wymagany przez model
        
        # Predykcja
        probability = model.predict_proba(features)[0, 1]
        
        # Zastosuj optymalny próg decyzyjny
        result = "Melanoma" if probability > OPTIMAL_THRESHOLD else "Benign"
        confidence = float(probability) if result == "Melanoma" else float(1 - probability)
        
        return {
            "prediction": result,
            "confidence": confidence * 100,
            "raw_probability": float(probability) * 100
        }
    except Exception as e:
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
        
        if model is None:
            return jsonify({'error': 'Model nie został poprawnie załadowany'}), 500

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

if __name__ == '__main__':
    app.run(debug=True)