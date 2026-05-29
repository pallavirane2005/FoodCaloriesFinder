# app.py
from flask import Flask, request, jsonify, render_template_string
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import base64
from calories import get_calories, INDEX_TO_CLASS, CLASS_NAMES

app = Flask(__name__)

MODEL_PATH = 'training/checkpoints/best_model.keras'

if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model not found at {MODEL_PATH}")
    print("Please run: python train_small_data.py first!")
    exit(1)

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded!")

HTML_PAGE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🍽️ Food Calorie Estimator</title>
    <style>
        * { 
            margin: 0; 
            padding: 0; 
            box-sizing: border-box; 
        }
        
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
            color: #eee; 
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: flex-start;
            padding: 40px 20px;
        }
        
        .container {
            max-width: 500px;
            width: 100%;
            animation: fadeIn 0.8s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* ===== HEADER ===== */
        .header {
            text-align: center;
            margin-bottom: 35px;
        }
        
        .header-icon {
            font-size: 60px;
            margin-bottom: 10px;
            display: block;
            animation: bounce 2s infinite;
        }
        
        @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-10px); }
        }
        
        h1 { 
            color: #00d9a5; 
            font-size: 2.2em;
            font-weight: 700;
            text-shadow: 0 0 30px rgba(0, 217, 165, 0.4);
            letter-spacing: -0.5px;
        }
        
        .subtitle {
            color: #8892b0;
            margin-top: 8px;
            font-size: 15px;
        }
        
        /* ===== UPLOAD AREA ===== */
        .upload-area { 
            border: 2px dashed #00d9a5; 
            padding: 40px 30px; 
            text-align: center; 
            border-radius: 20px; 
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(10px);
            transition: all 0.3s ease;
            margin-bottom: 25px;
            position: relative;
            overflow: hidden;
        }
        
        .upload-area:hover, .upload-area.dragover { 
            background: rgba(0, 217, 165, 0.08);
            border-color: #00ffbf;
            transform: translateY(-3px);
            box-shadow: 0 15px 40px rgba(0, 217, 165, 0.15);
        }
        
        .upload-area::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(0, 217, 165, 0.1) 0%, transparent 70%);
            opacity: 0;
            transition: opacity 0.3s;
        }
        
        .upload-area:hover::before {
            opacity: 1;
        }
        
        .upload-icon {
            font-size: 50px;
            margin-bottom: 15px;
            display: block;
        }
        
        .upload-text {
            font-size: 16px;
            color: #ccd6f6;
            margin-bottom: 5px;
        }
        
        .upload-hint {
            color: #8892b0; 
            font-size: 13px;
            margin-bottom: 20px;
        }
        
        input[type="file"] { 
            display: none;
        }
        
        .file-label {
            display: inline-block;
            background: linear-gradient(135deg, #00d9a5 0%, #00b386 100%); 
            color: #0f0c29; 
            padding: 12px 35px; 
            border-radius: 25px; 
            font-size: 15px; 
            cursor: pointer; 
            font-weight: 600;
            transition: all 0.3s;
            border: none;
        }
        
        .file-label:hover { 
            transform: scale(1.05);
            box-shadow: 0 5px 20px rgba(0, 217, 165, 0.4);
        }
        
        /* ===== PREVIEW SECTION ===== */
        .preview-section {
            display: none;
            margin-bottom: 25px;
            animation: slideUp 0.5s ease;
        }
        
        .preview-card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 20px;
            padding: 20px;
            border: 1px solid rgba(0, 217, 165, 0.2);
            position: relative;
        }
        
        .preview-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .preview-title {
            color: #00d9a5;
            font-size: 14px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .remove-btn {
            background: rgba(255, 107, 107, 0.2);
            color: #ff6b6b;
            border: none;
            padding: 6px 14px;
            border-radius: 20px;
            cursor: pointer;
            font-size: 12px;
            transition: all 0.3s;
        }
        
        .remove-btn:hover {
            background: rgba(255, 107, 107, 0.4);
            transform: scale(1.05);
        }
        
        .preview-image {
            width: 100%;
            max-height: 280px;
            object-fit: cover;
            border-radius: 15px;
            border: 2px solid rgba(0, 217, 165, 0.3);
        }
        
        /* ===== ANALYZE BUTTON ===== */
        .analyze-btn {
            width: 100%;
            background: linear-gradient(135deg, #00d9a5 0%, #00b386 100%); 
            color: #0f0c29; 
            padding: 16px; 
            border: none; 
            border-radius: 15px; 
            font-size: 17px; 
            cursor: pointer; 
            font-weight: 700;
            transition: all 0.3s;
            margin-bottom: 25px;
            display: none;
            letter-spacing: 0.5px;
        }
        
        .analyze-btn:hover { 
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(0, 217, 165, 0.3);
        }
        
        .analyze-btn:active {
            transform: translateY(0);
        }
        
        /* ===== LOADING ===== */
        .loading {
            display: none;
            text-align: center;
            padding: 40px;
            background: rgba(255, 255, 255, 0.03);
            border-radius: 20px;
            border: 1px solid rgba(0, 217, 165, 0.2);
            margin-bottom: 25px;
            animation: slideUp 0.5s ease;
        }
        
        .loading-emoji {
            font-size: 60px;
            margin-bottom: 20px;
            display: inline-block;
            animation: foodBounce 1s infinite;
        }
        
        @keyframes foodBounce {
            0%, 100% { transform: scale(1) rotate(0deg); }
            25% { transform: scale(1.2) rotate(-10deg); }
            75% { transform: scale(1.2) rotate(10deg); }
        }
        
        .loading-text {
            color: #00d9a5;
            font-size: 18px;
            font-weight: 600;
        }
        
        .loading-dots::after {
            content: '';
            animation: dots 1.5s infinite;
        }
        
        @keyframes dots {
            0%, 20% { content: '.'; }
            40% { content: '..'; }
            60%, 100% { content: '...'; }
        }
        
        /* ===== RESULT CARD ===== */
        .result { 
            display: none; 
            animation: slideUp 0.6s ease;
        }
        
        @keyframes slideUp {
            from { opacity: 0; transform: translateY(40px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .result-card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 25px;
            overflow: hidden;
            border: 1px solid rgba(0, 217, 165, 0.3);
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        }
        
        .result-image-container {
            position: relative;
            height: 250px;
            overflow: hidden;
        }
        
        .result-image {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        
        .result-image-overlay {
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            height: 50%;
            background: linear-gradient(to top, rgba(15, 12, 41, 0.9), transparent);
        }
        
        .result-body {
            padding: 30px;
        }
        
        .food-category {
            display: inline-block;
            background: rgba(0, 217, 165, 0.15);
            color: #00d9a5;
            padding: 6px 18px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 15px;
        }
        
        .food-name { 
            font-size: 28px; 
            color: #fff;
            font-weight: 700;
            margin-bottom: 5px;
        }
        
        .calories-container {
            text-align: center;
            margin: 25px 0;
            padding: 25px;
            background: rgba(0, 217, 165, 0.08);
            border-radius: 20px;
            border: 1px solid rgba(0, 217, 165, 0.2);
        }
        
        .calories { 
            font-size: 64px; 
            color: #00d9a5; 
            font-weight: 800;
            line-height: 1;
            text-shadow: 0 0 40px rgba(0, 217, 165, 0.4);
        }
        
        .calories-unit {
            font-size: 24px;
            color: #00d9a5;
            font-weight: 600;
        }
        
        .per-serving {
            color: #8892b0;
            font-size: 14px;
            margin-top: 8px;
        }
        
        .confidence-bar {
            margin-top: 20px;
        }
        
        .confidence-label {
            display: flex;
            justify-content: space-between;
            color: #8892b0;
            font-size: 13px;
            margin-bottom: 8px;
        }
        
        .confidence-track {
            height: 8px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            overflow: hidden;
        }
        
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #00d9a5, #00ffbf);
            border-radius: 10px;
            transition: width 1s ease;
            box-shadow: 0 0 10px rgba(0, 217, 165, 0.5);
        }
        
        .confirmed-badge {
            text-align: center;
            margin-top: 25px;
        }
        
        .confirmed-badge span {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: linear-gradient(135deg, #00d9a5 0%, #00b386 100%);
            color: #0f0c29;
            padding: 10px 30px;
            border-radius: 30px;
            font-weight: 700;
            font-size: 14px;
            box-shadow: 0 5px 20px rgba(0, 217, 165, 0.3);
        }
        
        /* ===== ERROR ===== */
        .error { 
            color: #ff6b6b; 
            text-align: center; 
            padding: 20px;
            background: rgba(255, 107, 107, 0.1);
            border-radius: 15px;
            margin-top: 20px;
            border: 1px solid rgba(255, 107, 107, 0.3);
            display: none;
            animation: shake 0.5s ease;
        }
        
        @keyframes shake {
            0%, 100% { transform: translateX(0); }
            25% { transform: translateX(-10px); }
            75% { transform: translateX(10px); }
        }
        
        /* ===== FOOTER ===== */
        .footer {
            text-align: center;
            margin-top: 30px;
            color: #8892b0;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <span class="header-icon">🍽️</span>
            <h1>Food Calorie Estimator</h1>
            <p class="subtitle">Upload a photo and get instant calorie estimates</p>
        </div>
        
        <!-- Upload Area -->
        <div class="upload-area" id="uploadArea">
            <span class="upload-icon">📤</span>
            <p class="upload-text">Drop your food image here</p>
            <p class="upload-hint">or click to browse • JPG, PNG, JPEG</p>
            <label class="file-label" for="fileInput">Choose File</label>
            <input type="file" id="fileInput" name="image" accept="image/*" required>
        </div>
        
        <!-- Preview Section -->
        <div class="preview-section" id="previewSection">
            <div class="preview-card">
                <div class="preview-header">
                    <span class="preview-title">📸 Preview</span>
                    <button class="remove-btn" onclick="removeFile()">✕ Remove</button>
                </div>
                <img class="preview-image" id="previewImage" src="" alt="Preview">
            </div>
        </div>
        
        <!-- Analyze Button -->
        <button class="analyze-btn" id="analyzeBtn" onclick="analyzeFood()">🔍 Analyze Food</button>
        
        <!-- Loading -->
        <div class="loading" id="loading">
            <span class="loading-emoji">🍕</span>
            <p class="loading-text">Analyzing your food<span class="loading-dots"></span></p>
        </div>
        
        <!-- Result -->
        <div id="result" class="result"></div>
        
        <!-- Error -->
        <div id="error" class="error"></div>
        
        <!-- Footer -->
        <div class="footer">
            <p>Powered by AI • Estimates per 100g serving</p>
        </div>
    </div>

    <script>
        const fileInput = document.getElementById('fileInput');
        const uploadArea = document.getElementById('uploadArea');
        const previewSection = document.getElementById('previewSection');
        const previewImage = document.getElementById('previewImage');
        const analyzeBtn = document.getElementById('analyzeBtn');
        const resultDiv = document.getElementById('result');
        const errorDiv = document.getElementById('error');
        const loadingDiv = document.getElementById('loading');
        
        let selectedFile = null;
        
        // File input change
        fileInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files[0]) {
                handleFile(e.target.files[0]);
            }
        });
        
        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                handleFile(e.dataTransfer.files[0]);
            }
        });
        
        // Click to upload
        uploadArea.addEventListener('click', (e) => {
            if (e.target !== fileInput && !e.target.classList.contains('file-label')) {
                fileInput.click();
            }
        });
        
        function handleFile(file) {
            if (!file.type.startsWith('image/')) {
                showError('Please upload an image file (JPG, PNG, JPEG)');
                return;
            }
            
            selectedFile = file;
            const reader = new FileReader();
            reader.onload = (e) => {
                previewImage.src = e.target.result;
                previewSection.style.display = 'block';
                analyzeBtn.style.display = 'block';
                uploadArea.style.display = 'none';
                resultDiv.style.display = 'none';
                errorDiv.style.display = 'none';
            };
            reader.readAsDataURL(file);
        }
        
        function removeFile() {
            selectedFile = null;
            fileInput.value = '';
            previewSection.style.display = 'none';
            analyzeBtn.style.display = 'none';
            uploadArea.style.display = 'block';
            resultDiv.style.display = 'none';
            errorDiv.style.display = 'none';
        }
        
        function showError(msg) {
            errorDiv.textContent = '❌ ' + msg;
            errorDiv.style.display = 'block';
            setTimeout(() => {
                errorDiv.style.display = 'none';
            }, 5000);
        }
        
        async function analyzeFood() {
            if (!selectedFile) return;
            
            const formData = new FormData();
            formData.append('image', selectedFile);
            
            resultDiv.style.display = 'none';
            errorDiv.style.display = 'none';
            loadingDiv.style.display = 'block';
            analyzeBtn.style.display = 'none';
            
            try {
                const response = await fetch('/predict', { method: 'POST', body: formData });
                const data = await response.json();
                loadingDiv.style.display = 'none';
                analyzeBtn.style.display = 'block';
                
                if (data.error) {
                    showError(data.error);
                    return;
                }
                
                const r = data.prediction;
                const html = `
                    <div class="result-card">
                        <div class="result-image-container">
                            <img class="result-image" src="${r.image_base64}" alt="${r.food_class}">
                            <div class="result-image-overlay"></div>
                        </div>
                        <div class="result-body">
                            <span class="food-category">Food Category</span>
                            <div class="food-name">${r.food_class}</div>
                            
                            <div class="calories-container">
                                <div class="calories">${r.calories.total_calories} <span class="calories-unit">kcal</span></div>
                                <div class="per-serving">per 100g serving</div>
                            </div>
                            
                            <div class="confidence-bar">
                                <div class="confidence-label">
                                    <span>Confidence</span>
                                    <span>${r.confidence}%</span>
                                </div>
                                <div class="confidence-track">
                                    <div class="confidence-fill" style="width: 0%" data-width="${r.confidence}%"></div>
                                </div>
                            </div>
                            
                            <div class="confirmed-badge">
                                <span>✅ Confirmed Prediction</span>
                            </div>
                        </div>
                    </div>
                `;
                
                resultDiv.innerHTML = html;
                resultDiv.style.display = 'block';
                
                // Animate confidence bar
                setTimeout(() => {
                    const fill = document.querySelector('.confidence-fill');
                    if (fill) fill.style.width = fill.getAttribute('data-width');
                }, 100);
                
            } catch (err) {
                loadingDiv.style.display = 'none';
                analyzeBtn.style.display = 'block';
                showError(err.message);
            }
        }
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    return HTML_PAGE

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty file'}), 400
    
    try:
        # Read image bytes
        image_bytes = file.read()
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert('RGB')
        
        # Convert to base64 for display
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        img_base64 = f"data:image/png;base64,{img_base64}"
        
        # Resize for model
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = model.predict(img_array, verbose=0)[0]
        predicted_idx = int(np.argmax(predictions))
        confidence = round(float(predictions[predicted_idx]) * 100, 1)
        
        food_class = INDEX_TO_CLASS[predicted_idx]
        calorie_info = get_calories(food_class, portion_grams=100)
        
        return jsonify({
            'success': True,
            'prediction': {
                'food_class': food_class,
                'confidence': confidence,
                'calories': calorie_info,
                'image_base64': img_base64
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print(f"\n{'='*50}")
    print("🍽️ Food Calorie Estimator")
    print(f"{'='*50}")
    print(f"Classes: {CLASS_NAMES}")
    print(f"Server: http://localhost:5000")
    print(f"{'='*50}\n")
    app.run(debug=True, host='127.0.0.1', port=5000)