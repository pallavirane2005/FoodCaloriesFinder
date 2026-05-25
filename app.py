# app.py
from flask import Flask, request, jsonify, render_template_string
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
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
<html>
<head>
    <title>🍽️ Food Calorie Estimator</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Arial, sans-serif; 
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee; 
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .container {
            max-width: 600px;
            width: 100%;
        }
        h1 { 
            text-align: center; 
            color: #4ecca3; 
            margin-bottom: 30px;
            font-size: 2.5em;
            text-shadow: 0 0 20px rgba(78, 204, 163, 0.3);
        }
        .upload-box { 
            border: 3px dashed #4ecca3; 
            padding: 50px 40px; 
            text-align: center; 
            border-radius: 20px; 
            background: rgba(22, 33, 62, 0.8);
            transition: all 0.3s;
            margin-bottom: 30px;
        }
        .upload-box:hover { 
            background: rgba(22, 33, 62, 1);
            transform: translateY(-5px);
            box-shadow: 0 10px 40px rgba(78, 204, 163, 0.2);
        }
        input[type="file"] { 
            margin: 25px 0; 
            color: #fff;
            font-size: 16px;
        }
        button { 
            background: linear-gradient(135deg, #4ecca3 0%, #3db892 100%); 
            color: #1a1a2e; 
            padding: 15px 50px; 
            border: none; 
            border-radius: 10px; 
            font-size: 18px; 
            cursor: pointer; 
            font-weight: bold;
            transition: all 0.3s;
        }
        button:hover { 
            transform: scale(1.05);
            box-shadow: 0 5px 20px rgba(78, 204, 163, 0.4);
        }
        .result { 
            padding: 30px; 
            background: rgba(22, 33, 62, 0.9); 
            border-radius: 20px; 
            display: none; 
            border: 2px solid #4ecca3;
            animation: slideUp 0.5s ease;
        }
        @keyframes slideUp {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .calories { 
            font-size: 72px; 
            color: #4ecca3; 
            font-weight: bold; 
            text-align: center;
            text-shadow: 0 0 30px rgba(78, 204, 163, 0.5);
        }
        .food-name { 
            font-size: 32px; 
            text-align: center; 
            margin: 15px 0; 
            color: #fff;
            font-weight: 600;
        }
        .confidence { 
            text-align: center; 
            color: #aaa; 
            font-size: 16px;
            margin-bottom: 25px;
        }
        .top3 { 
            margin-top: 25px; 
            padding: 20px; 
            background: rgba(15, 52, 96, 0.6); 
            border-radius: 15px; 
        }
        .top3 h3 { 
            color: #4ecca3; 
            margin-bottom: 15px;
            font-size: 20px;
        }
        .top3-item { 
            padding: 12px; 
            border-bottom: 1px solid rgba(255,255,255,0.1); 
            display: flex; 
            justify-content: space-between;
            align-items: center;
            transition: background 0.2s;
        }
        .top3-item:hover {
            background: rgba(255,255,255,0.05);
            border-radius: 8px;
        }
        .top3-item:last-child { border-bottom: none; }
        .error { 
            color: #ff6b6b; 
            text-align: center; 
            padding: 20px;
            background: rgba(255, 107, 107, 0.1);
            border-radius: 10px;
            margin-top: 20px;
        }
        .loading {
            display: none;
            text-align: center;
            color: #4ecca3;
            font-size: 18px;
            margin: 20px 0;
        }
        .spinner {
            border: 4px solid rgba(78, 204, 163, 0.3);
            border-top: 4px solid #4ecca3;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🍽️ Food Calorie Estimator</h1>
        <div class="upload-box">
            <form id="uploadForm" enctype="multipart/form-data">
                <p style="font-size: 18px; margin-bottom: 10px;">Upload a food photo to estimate calories</p>
                <p style="color: #888; font-size: 14px;">Supports: JPG, PNG, JPEG</p>
                <input type="file" name="image" accept="image/*" required><br>
                <button type="submit">🔍 Analyze Food</button>
            </form>
        </div>
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Analyzing your food...</p>
        </div>
        <div id="result" class="result"></div>
        <div id="error" class="error"></div>
    </div>

    <script>
        document.getElementById('uploadForm').onsubmit = async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            document.getElementById('result').style.display = 'none';
            document.getElementById('error').textContent = '';
            document.getElementById('loading').style.display = 'block';
            
            try {
                const response = await fetch('/predict', { method: 'POST', body: formData });
                const data = await response.json();
                document.getElementById('loading').style.display = 'none';
                
                if (data.error) {
                    document.getElementById('error').textContent = '❌ ' + data.error;
                    return;
                }
                
                const r = data.prediction;
                let html = `
                    <div class="food-name">${r.food_class}</div>
                    <div class="calories">${r.calories.total_calories} <span style="font-size: 32px;">kcal</span></div>
                    <div class="confidence">per 100g serving • Confidence: ${r.confidence}%</div>
                    <div class="top3">
                        <h3>🏆 Top Predictions</h3>
                `;
                r.top_3.forEach((item, i) => {
                    const medal = i === 0 ? '🥇' : i === 1 ? '🥈' : '🥉';
                    html += `<div class="top3-item">
                        <span>${medal} ${item.food}</span>
                        <span style="color: #4ecca3; font-weight: 600;">${item.calories} kcal (${item.confidence}%)</span>
                    </div>`;
                });
                html += '</div>';
                
                document.getElementById('result').innerHTML = html;
                document.getElementById('result').style.display = 'block';
            } catch (err) {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('error').textContent = '❌ Error: ' + err.message;
            }
        };
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
        img = Image.open(io.BytesIO(file.read()))
        img = img.convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = model.predict(img_array, verbose=0)[0]
        predicted_idx = int(np.argmax(predictions))
        confidence = round(float(predictions[predicted_idx]) * 100, 1)
        
        food_class = INDEX_TO_CLASS[predicted_idx]
        calorie_info = get_calories(food_class, portion_grams=100)
        
        top_3_indices = np.argsort(predictions)[-3:][::-1]
        top_3 = []
        for idx in top_3_indices:
            idx = int(idx)
            top_3.append({
                'food': INDEX_TO_CLASS[idx],
                'confidence': round(float(predictions[idx]) * 100, 1),
                'calories': get_calories(INDEX_TO_CLASS[idx], 100)['total_calories']
            })
        
        return jsonify({
            'success': True,
            'prediction': {
                'food_class': food_class,
                'confidence': confidence,
                'calories': calorie_info,
                'top_3': top_3
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