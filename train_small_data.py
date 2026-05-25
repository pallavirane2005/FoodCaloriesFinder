# train_small_data.py
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
from calories import CLASS_NAMES, CLASS_TO_INDEX

# Paths
DATA_DIR = 'training'  # We'll use training folder (combine all data here)
CHECKPOINT_DIR = 'training/checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 50
NUM_CLASSES = len(CLASS_NAMES)
N_FOLDS = 5  # 5-fold cross validation

print(f"Classes: {NUM_CLASSES}")
print(f"Using K-Fold Cross Validation ({N_FOLDS} folds)")

# Load all image paths and labels
all_images = []
all_labels = []

for class_name in CLASS_NAMES:
    class_dir = os.path.join(DATA_DIR, class_name)
    if not os.path.exists(class_dir):
        continue
    images = [f for f in os.listdir(class_dir) 
              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    for img in images:
        all_images.append(os.path.join(class_dir, img))
        all_labels.append(CLASS_TO_INDEX[class_name])

all_images = np.array(all_images)
all_labels = np.array(all_labels)

print(f"\nTotal images: {len(all_images)}")
for i, cls in enumerate(CLASS_NAMES):
    count = np.sum(all_labels == i)
    print(f"  {cls}: {count}")

# K-Fold Cross Validation
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

best_val_acc = 0
best_model = None

for fold, (train_idx, val_idx) in enumerate(kf.split(all_images), 1):
    print(f"\n{'='*50}")
    print(f"FOLD {fold}/{N_FOLDS}")
    print(f"{'='*50}")
    
    # Split data
    train_paths = all_images[train_idx]
    train_labels = all_labels[train_idx]
    val_paths = all_images[val_idx]
    val_labels = all_labels[val_idx]
    
    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}")
    
    # Create tf.data pipelines
    def load_image(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, IMG_SIZE)
        img = img / 255.0
        return img, label
    
    def augment(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, 0.3)
        image = tf.image.random_contrast(image, 0.7, 1.3)
        image = tf.image.random_saturation(image, 0.7, 1.3)
        image = tf.clip_by_value(image, 0, 1)
        return image, label
    
    # Training dataset
    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    train_ds = train_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    # Validation dataset
    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    val_ds = val_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    # Build model
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3),
        alpha=0.35  # Even smaller model for tiny dataset
    )
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.8),  # Very high dropout
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    # Class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weight_dict = dict(enumerate(class_weights))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    checkpoint = callbacks.ModelCheckpoint(
        f'{CHECKPOINT_DIR}/fold_{fold}_best.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Train
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=[checkpoint, early_stop],
        class_weight=class_weight_dict,
        verbose=1
    )
    
    # Evaluate
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"\nFold {fold} Validation Accuracy: {val_acc:.2%}")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model = model
        print(f"⭐ New best model! Saving...")
        model.save(f'{CHECKPOINT_DIR}/best_model.keras')

print(f"\n{'='*50}")
print(f"Best Validation Accuracy: {best_val_acc:.2%}")
print(f"Best model saved to: {CHECKPOINT_DIR}/best_model.keras")