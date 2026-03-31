"""
Neural network models for tumor detection, classification, and segmentation
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16, ResNet50, EfficientNetB0
from tensorflow.keras.optimizers import Adam
from config import IMG_SIZE, LEARNING_RATE, TUMOR_TYPES

class TumorDetectionModel:
    """Binary classification: Tumor vs No Tumor"""
    
    def __init__(self, img_size=IMG_SIZE):
        self.img_size = img_size
        self.model = None
    
    def build_model(self, use_transfer_learning=True):
        """Build tumor detection model"""
        if use_transfer_learning:
            # Use VGG16 as base model
            base_model = VGG16(
                weights='imagenet',
                include_top=False,
                input_shape=(self.img_size[0], self.img_size[1], 3)
            )
            
            # Freeze base model layers
            base_model.trainable = False
            
            # Build model
            model = models.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(1, activation='sigmoid')
            ])
        else:
            # Custom CNN
            model = models.Sequential([
                layers.Conv2D(32, (3, 3), activation='relu', 
                            input_shape=(self.img_size[0], self.img_size[1], 3)),
                layers.MaxPooling2D((2, 2)),
                
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                
                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                
                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                
                layers.Flatten(),
                layers.Dense(512, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(1, activation='sigmoid')
            ])
        
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.model = model
        return model
    
    def train(self, train_generator, val_generator, epochs=50):
        """Train the model"""
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks
        )
        
        return history

class TumorClassificationModel:
    """Multi-class classification: Types of tumors"""
    
    def __init__(self, img_size=IMG_SIZE, num_classes=len(TUMOR_TYPES)):
        self.img_size = img_size
        self.num_classes = num_classes
        self.model = None
    
    def build_model(self, use_transfer_learning=True):
        """Build tumor classification model"""
        if use_transfer_learning:
            # Use EfficientNetB0 as base model
            base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=(self.img_size[0], self.img_size[1], 3)
            )
            
            # Unfreeze last layers for fine-tuning
            base_model.trainable = True
            for layer in base_model.layers[:-20]:  # Freeze all except last 20 layers
                layer.trainable = False
            
            
            model = models.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.BatchNormalization(),
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(self.num_classes, activation='softmax')
            ])
        else:
            # Custom CNN
            model = models.Sequential([
                layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(self.img_size[0], self.img_size[1], 3)),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                
                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                
                layers.Conv2D(256, (3, 3), activation='relu'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                
                layers.Flatten(),
                layers.Dense(512, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(self.num_classes, activation='softmax')
            ])
        
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, train_generator, val_generator, epochs=50):
        """Train the model"""
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks
        )
        
        return history

class TumorSegmentationModel:
    """U-Net for tumor segmentation"""
    
    def __init__(self, img_size=IMG_SIZE):
        self.img_size = img_size
        self.model = None
    
    def build_unet(self):
        """Build U-Net architecture"""
        inputs = layers.Input((self.img_size[0], self.img_size[1], 3))
        
        # Encoder
        c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
        p1 = layers.MaxPooling2D((2, 2))(c1)
        
        c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
        c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
        p2 = layers.MaxPooling2D((2, 2))(c2)
        
        c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
        c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
        p3 = layers.MaxPooling2D((2, 2))(c3)
        
        # Bottleneck
        c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
        c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
        
        # Decoder
        u5 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c4)
        u5 = layers.concatenate([u5, c3])
        c5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u5)
        c5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c5)
        
        u6 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = layers.concatenate([u6, c2])
        c6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u6)
        c6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c6)
        
        u7 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = layers.concatenate([u7, c1])
        c7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u7)
        c7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c7)
        
        outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c7)
        
        model = models.Model(inputs=[inputs], outputs=[outputs])
        
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
