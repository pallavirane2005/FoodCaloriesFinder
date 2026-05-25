# train_model_v2.py
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
from calories import CLASS_NAMES

TRAIN_DIR = 'training'
VAL_DIR = 'validation'
CHECKPOINT_DIR = 'training/checkpoints'

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16  # Lower if memory issues
EPOCHS = 50
NUM_CLASSES = len(CLASS_NAMES)

print(f"Classes: {CLASS_NAMES}")

# More augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=True,
    zoom_range=0.3,
    brightness_range=[0.7, 1.3],
    shear_range=0.2,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASS_NAMES
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASS_NAMES
)

print(f"\nTraining: {train_generator.samples} images")
print(f"Validation: {val_generator.samples} images")

# Class weights for imbalance
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weight_dict = dict(enumerate(class_weights))
print(f"\nClass weights: {class_weight_dict}")

# Use pretrained MobileNetV2
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False  # Freeze first

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Callbacks
checkpoint = callbacks.ModelCheckpoint(
    f'{CHECKPOINT_DIR}/best_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=7,
    restore_best_weights=True
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7
)

print("\n🚀 Training...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[checkpoint, early_stop, reduce_lr],
    class_weight=class_weight_dict
)

# Fine-tune: unfreeze some layers
print("\n🔓 Fine-tuning...")
base_model.trainable = True
for layer in base_model.layers[:-30]:  # Freeze all except last 30
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),  # Very low LR
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    callbacks=[checkpoint, early_stop, reduce_lr],
    class_weight=class_weight_dict
)

model.save(f'{CHECKPOINT_DIR}/final_model.keras')
print(f"\n✅ Done! Best model: {CHECKPOINT_DIR}/best_model.keras")