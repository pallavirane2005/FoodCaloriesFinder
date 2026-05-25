# test_model.py
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from calories import INDEX_TO_CLASS, CLASS_NAMES

print("=" * 50)
print("TESTING MODEL ON SAMPLE IMAGES")
print("=" * 50)

model_path = 'training/checkpoints/best_model.keras'

if not os.path.exists(model_path):
    print(f"❌ Model not found: {model_path}")
    print("Run: python train_model.py first!")
    exit(1)

print(f"Loading model from: {model_path}")
model = tf.keras.models.load_model(model_path)
print("✅ Model loaded!\n")

for class_name in CLASS_NAMES:
    folder = os.path.join('training', class_name)
    
    if not os.path.exists(folder):
        print(f"❌ Folder not found: {folder}")
        continue
    
    # Get first image
    images = [f for f in os.listdir(folder) 
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not images:
        print(f"⚠️  No images in {folder}")
        continue
    
    img_path = os.path.join(folder, images[0])
    
    # Predict
    img = Image.open(img_path).resize((224, 224)).convert('RGB')
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    pred = model.predict(img_array, verbose=0)[0]
    top_idx = np.argmax(pred)
    confidence = pred[top_idx] * 100
    
    print(f"\n📸 {class_name} (tested with: {images[0]})")
    print(f"   Predicted: {INDEX_TO_CLASS[top_idx]} ({confidence:.1f}%)")
    
    # Top 3
    top3 = np.argsort(pred)[-3:][::-1]
    for i, idx in enumerate(top3, 1):
        print(f"   #{i} {INDEX_TO_CLASS[idx]}: {pred[idx]*100:.1f}%")