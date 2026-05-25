# check_data.py
import os

print("=" * 50)
print("CHECKING IMAGE COUNTS PER CLASS")
print("=" * 50)

for folder in ['training', 'validation', 'evaluation']:
    print(f"\n📁 {folder.upper()}:")
    path = os.path.join(os.getcwd(), folder)
    
    if not os.path.exists(path):
        print(f"  ❌ Folder not found: {path}")
        continue
    
    total_images = 0
    for cls in sorted(os.listdir(path)):
        cls_path = os.path.join(path, cls)
        if os.path.isdir(cls_path):
            count = len([f for f in os.listdir(cls_path) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))])
            print(f"  {cls:20s} → {count:4d} images")
            total_images += count
    
    print(f"  {'TOTAL':20s} → {total_images:4d} images")