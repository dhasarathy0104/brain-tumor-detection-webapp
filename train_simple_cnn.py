"""
Simple and Effective CNN for Brain Tumor Classification
No transfer learning - trains from scratch
Should work reliably with your dataset
"""

import os
import sys
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, MODEL_DIR, CLASSIFICATION_MODEL_PATH

# Configuration
EPOCHS = 100
BATCH_SIZE = 32
IMG_SIZE = (224, 224)  # Bumping internal resolution to 224 to resolve Glioma vs Meningioma textures
LEARNING_RATE = 0.0005

def build_simple_cnn(num_classes=4):
    """
    Build a simple but effective CNN from scratch
    """
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                     input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
                     kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),
        
        # Block 4
        layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_data_generators(train_dir, val_dir):
    """Create data generators with proper settings"""
    
    # Training with strong augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Validation without augmentation
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        color_mode='rgb'  # Ensure RGB
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        color_mode='rgb'
    )
    
    return train_generator, val_generator

def plot_history(history, save_path):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Validation')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train')
    axes[1].plot(history.history['val_loss'], label='Validation')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✓ Plot saved: {save_path}")

def main():
    print("="*70)
    print("🧠 SIMPLE CNN TRAINING - BRAIN TUMOR CLASSIFICATION")
    print("="*70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Setup paths
    train_dir = os.path.join(DATA_DIR, 'classification', 'train')
    val_dir = os.path.join(DATA_DIR, 'classification', 'val')
    
    if not os.path.exists(train_dir):
        print(f"❌ Training data not found: {train_dir}")
        return
    
    # Create data generators
    print("\n📊 Loading data...")
    train_gen, val_gen = create_data_generators(train_dir, val_dir)
    
    print(f"✓ Training samples: {train_gen.samples}")
    print(f"✓ Validation samples: {val_gen.samples}")
    print(f"✓ Classes: {list(train_gen.class_indices.keys())}")
    print(f"✓ Batches per epoch: {len(train_gen)}")
    
    # Verify data loading
    print("\n🔍 Verifying data...")
    x_batch, y_batch = next(train_gen)
    print(f"✓ Batch shape: {x_batch.shape}")
    print(f"✓ Label shape: {y_batch.shape}")
    print(f"✓ Image range: [{x_batch.min():.2f}, {x_batch.max():.2f}]")
    print(f"✓ Sample label: {y_batch[0]}")
    
    # Build model
    print("\n🏗️ Building model...")
    model = build_simple_cnn(num_classes=4)
    
    print("\n📋 Model Architecture:")
    model.summary()
    print(f"\nTotal parameters: {model.count_params():,}")
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            CLASSIFICATION_MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train
    print("\n" + "="*70)
    print("🏋️ TRAINING STARTED")
    print("="*70)
    print(f"Epochs: {EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Image Size: {IMG_SIZE}")
    print("="*70 + "\n")
    
    start_time = datetime.now()
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    # Results
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE")
    print("="*70)
    print(f"Duration: {duration}")
    
    # Evaluate
    val_loss, val_acc = model.evaluate(val_gen, verbose=0)
    
    print(f"\n📊 FINAL RESULTS:")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc*100:.2f}%")
    
    if val_acc >= 0.85:
        print("\n🌟 EXCELLENT! Accuracy ≥ 85%")
    elif val_acc >= 0.75:
        print("\n✅ GOOD! Accuracy ≥ 75%")
    elif val_acc >= 0.65:
        print("\n👍 ACCEPTABLE! Accuracy ≥ 65%")
    else:
        print("\n⚠️  LOW ACCURACY - Check dataset")
    
    # Plot
    plot_path = os.path.join(MODEL_DIR, 'simple_cnn_history.png')
    plot_history(history, plot_path)
    
    print("\n" + "="*70)
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

if __name__ == "__main__":
    main()
