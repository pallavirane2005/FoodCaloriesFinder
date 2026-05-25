# test_model.py
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from calories import INDEX_TO_CLASS, CLASS_NAMES

print("=" * 60)
print("TESTING MODEL ON SAMPLE IMAGES FROM EACH CLASS")
print("=" * 60)

model_path = 'training/checkpoints/best_model.keras'

if not os.path.exists(model_path):
    print(f"❌ Model not found: {model_path}")
    print("Run: python train_small_data.py first!")
    exit(1)

print(f"Loading model from: {model_path}")
model = tf.keras.models.load_model(model_path)
print("✅ Model loaded!\n")

correct = 0
total = 0

for class_name in CLASS_NAMES:
    folder = os.path.join('training', class_name)
    
    if not os.path.exists(folder):
        print(f"❌ Folder not found: {folder}")
        continue
    
    images = [f for f in os.listdir(folder) 
              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not images:
        print(f"⚠️  No images in {folder}")
        continue
    
    # Test first 3 images from each class
    test_images = images[:3]
    
    print(f"\n📸 Testing '{class_name}' ({len(test_images)} images):")
    
    for img_file in test_images:
        img_path = os.path.join(folder, img_file)
        
        img = Image.open(img_path).resize((224, 224)).convert('RGB')
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        pred = model.predict(img_array, verbose=0)[0]
        top_idx = int(np.argmax(pred))
        confidence = pred[top_idx] * 100
        
        predicted_class = INDEX_TO_CLASS[top_idx]
        is_correct = "✅" if predicted_class == class_name else "❌"
        
        if predicted_class == class_name:
            correct += 1
        total += 1
        
        print(f"  {is_correct} {img_file:25s} → Predicted: {predicted_class:15s} ({confidence:5.1f}%)")
        
        # Show top 3
        top3 = np.argsort(pred)[-3:][::-1]
        top3_str = " | ".join([f"{INDEX_TO_CLASS[int(i)]} {pred[int(i)]*100:.1f}%" for i in top3])
        print(f"     Top 3: {top3_str}")

print(f"\n{'=' * 60}")
print(f"ACCURACY: {correct}/{total} = {correct/total*100:.1f}%")
print(f"{'=' * 60}")