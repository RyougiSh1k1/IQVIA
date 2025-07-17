"""
OUD Prediction - BiGAN (Bidirectional Generative Adversarial Network)
=====================================================================
This script implements a BiGAN-based approach for handling severe class imbalance
in OUD prediction. BiGAN learns bidirectional mappings between data and latent space,
enabling high-quality synthetic sample generation for the minority class.

Architecture:
- Generator: Maps latent space to data space (generates synthetic OUD samples)
- Encoder: Maps data space to latent space (learns representations)
- Discriminator: Distinguishes real/fake pairs of (data, latent code)
- Classifier: Final OUD prediction model trained on augmented data

The approach:
1. Train BiGAN to learn data distribution and generate synthetic OUD samples
2. Use trained generator to augment minority class
3. Train final classifier on balanced dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, roc_curve,
    precision_score, recall_score, f1_score,
    accuracy_score, average_precision_score,
    precision_recall_curve, matthews_corrcoef,
    fbeta_score  # Add this line
)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
import tensorflow.keras.backend as K
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

class BiGANOUDModel:
    """
    BiGAN-based model for imbalanced OUD prediction
    """
    
    def __init__(self, data_path='/sharefolder/wanglab/ML_training/final_OUD_ML_dataset.csv'):
        self.data_path = data_path
        
        # Columns to drop
        self.columns_to_drop = [
            'pat_id', 'most_recent_date', 'start_date_180_days', 
            'start_date_2_years', 'pat_zip3'
        ]
        
        self.target_col = 'oud_label'
        
        # Initialize attributes
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_val = None
        self.y_val = None
        self.scaler = None
        self.feature_names = None
        self.n_features = None
        self.latent_dim = 100  # Dimension of latent space
        self.imbalance_ratio = None
        
        # BiGAN components
        self.generator = None
        self.encoder = None
        self.discriminator = None
        self.bigan = None
        self.classifier = None
        
    def load_and_prepare_data(self):
        """Load and prepare data"""
        print("Loading dataset...")
        df = pd.read_csv(self.data_path)
        print(f"Initial dataset shape: {df.shape}")
        
        # Drop unnecessary columns
        columns_to_drop = [col for col in self.columns_to_drop if col in df.columns]
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
        
        # Handle categorical variables
        print("\nEncoding categorical variables...")
        if 'der_sex' in df.columns:
            le_sex = LabelEncoder()
            df['der_sex'] = le_sex.fit_transform(df['der_sex'])
        
        if 'pay_type' in df.columns:
            le_pay = LabelEncoder()
            df['pay_type'] = le_pay.fit_transform(df['pay_type'])
        
        # Extract features and target
        self.y = df[self.target_col].values
        self.X = df.drop(columns=[self.target_col])
        
        # Handle non-numeric columns
        numeric_columns = self.X.select_dtypes(include=[np.number]).columns
        self.X = self.X[numeric_columns]
        
        self.feature_names = list(self.X.columns)
        self.n_features = len(self.feature_names)
        self.X = self.X.values
        
        # Calculate class distribution
        n_samples = len(self.y)
        class_counts = np.bincount(self.y)
        self.imbalance_ratio = class_counts[0] / class_counts[1]
        
        print(f"\nClass distribution:")
        print(f"Non-OUD (0): {class_counts[0]} ({class_counts[0]/n_samples*100:.2f}%)")
        print(f"OUD (1): {class_counts[1]} ({class_counts[1]/n_samples*100:.2f}%)")
        print(f"Imbalance ratio: 1:{self.imbalance_ratio:.1f}")
        
        # Split data
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            self.X, self.y, test_size=0.15, random_state=42, stratify=self.y
        )
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
        )
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_val = self.scaler.transform(self.X_val)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"\nTrain set: {self.X_train.shape}")
        print(f"Validation set: {self.X_val.shape}")
        print(f"Test set: {self.X_test.shape}")
        
        return True
    
    def build_generator(self):
        """
        Generator network: z -> x
        Maps from latent space to data space
        """
        latent_input = Input(shape=(self.latent_dim,))
        
        x = layers.Dense(256)(latent_input)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Dense(512)(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Dense(512)(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Dense(256)(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.BatchNormalization()(x)
        
        # Output layer with tanh activation
        output = layers.Dense(self.n_features, activation='tanh')(x)
        
        generator = Model(latent_input, output, name='generator')
        return generator
    
    def build_encoder(self):
        """
        Encoder network: x -> z
        Maps from data space to latent space
        """
        data_input = Input(shape=(self.n_features,))
        
        x = layers.Dense(256)(data_input)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(512)(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(512)(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(256)(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dropout(0.3)(x)
        
        # Output latent representation
        latent_output = layers.Dense(self.latent_dim)(x)
        
        encoder = Model(data_input, latent_output, name='encoder')
        return encoder
    
    def build_discriminator(self):
        """
        Discriminator network: (x, z) -> probability
        Determines if (data, latent) pairs are real or generated
        """
        data_input = Input(shape=(self.n_features,))
        latent_input = Input(shape=(self.latent_dim,))
        
        # Process data
        x_data = layers.Dense(256)(data_input)
        x_data = layers.LeakyReLU(0.2)(x_data)
        x_data = layers.Dropout(0.3)(x_data)
        
        # Process latent
        x_latent = layers.Dense(256)(latent_input)
        x_latent = layers.LeakyReLU(0.2)(x_latent)
        x_latent = layers.Dropout(0.3)(x_latent)
        
        # Concatenate
        x = layers.Concatenate()([x_data, x_latent])
        
        x = layers.Dense(512)(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(256)(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(128)(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dropout(0.3)(x)
        
        # Output probability
        output = layers.Dense(1, activation='sigmoid')(x)
        
        discriminator = Model([data_input, latent_input], output, name='discriminator')
        return discriminator
    
    def build_bigan(self):
        """Build and compile the complete BiGAN model"""
        print("\nBuilding BiGAN components...")
        
        # Build components
        self.generator = self.build_generator()
        self.encoder = self.build_encoder()
        self.discriminator = self.build_discriminator()
        
        # Print architectures
        print("\nGenerator architecture:")
        self.generator.summary()
        
        print("\nEncoder architecture:")
        self.encoder.summary()
        
        print("\nDiscriminator architecture:")
        self.discriminator.summary()
        
        # Compile discriminator
        self.discriminator.compile(
            optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Build BiGAN
        # Freeze discriminator for generator/encoder training
        self.discriminator.trainable = False
        
        # Input
        data_input = Input(shape=(self.n_features,))
        latent_input = Input(shape=(self.latent_dim,))
        
        # Generate outputs
        encoded = self.encoder(data_input)
        generated = self.generator(latent_input)
        
        # Discriminator outputs
        real_pair_score = self.discriminator([data_input, encoded])
        fake_pair_score = self.discriminator([generated, latent_input])
        
        # BiGAN model
        self.bigan = Model([data_input, latent_input], 
                          [real_pair_score, fake_pair_score], 
                          name='bigan')
        
        self.bigan.compile(
            optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
            loss=['binary_crossentropy', 'binary_crossentropy']
        )
        
        print("\nBiGAN model built successfully!")
    
    def train_bigan(self, epochs=100, batch_size=256):
        """Train the BiGAN model"""
        print("\n" + "="*60)
        print("TRAINING BiGAN")
        print("="*60)
        
        # Get minority class samples
        minority_indices = np.where(self.y_train == 1)[0]
        minority_samples = self.X_train[minority_indices]
        
        # Training parameters
        n_batches = len(minority_samples) // batch_size
        
        # Labels for training
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))
        
        # Training history
        d_losses = []
        g_losses = []
        
        for epoch in range(epochs):
            # Shuffle minority samples
            np.random.shuffle(minority_samples)
            
            epoch_d_loss = 0
            epoch_g_loss = 0
            
            for batch in range(n_batches):
                # Get batch of real minority samples
                batch_start = batch * batch_size
                batch_end = (batch + 1) * batch_size
                real_batch = minority_samples[batch_start:batch_end]
                
                # Sample latent vectors
                z_batch = np.random.normal(0, 1, (batch_size, self.latent_dim))
                
                # Generate fake samples and encode real samples
                fake_batch = self.generator.predict(z_batch, verbose=0)
                encoded_batch = self.encoder.predict(real_batch, verbose=0)
                
                # Train discriminator
                self.discriminator.trainable = True
                
                # Real pairs (real data, encoded latent)
                d_loss_real = self.discriminator.train_on_batch(
                    [real_batch, encoded_batch], real_labels
                )
                
                # Fake pairs (generated data, sampled latent)
                d_loss_fake = self.discriminator.train_on_batch(
                    [fake_batch, z_batch], fake_labels
                )
                
                d_loss = 0.5 * (d_loss_real[0] + d_loss_fake[0])
                
                # Train generator and encoder
                self.discriminator.trainable = False
                
                # We want discriminator to think these are real
                misleading_labels = np.ones((batch_size, 1))
                
                g_loss = self.bigan.train_on_batch(
                    [real_batch, z_batch],
                    [misleading_labels, misleading_labels]
                )
                
                epoch_d_loss += d_loss
                epoch_g_loss += g_loss[0]
            
            # Average losses
            epoch_d_loss /= n_batches
            epoch_g_loss /= n_batches
            
            d_losses.append(epoch_d_loss)
            g_losses.append(epoch_g_loss)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - D loss: {epoch_d_loss:.4f}, G loss: {epoch_g_loss:.4f}")
        
        # Plot training history
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(d_losses, label='Discriminator Loss')
        plt.plot(g_losses, label='Generator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('BiGAN Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        # Smooth the losses for better visualization
        window = 10
        if len(d_losses) > window:
            d_smooth = np.convolve(d_losses, np.ones(window)/window, mode='valid')
            g_smooth = np.convolve(g_losses, np.ones(window)/window, mode='valid')
            plt.plot(d_smooth, label='Discriminator Loss (smoothed)')
            plt.plot(g_smooth, label='Generator Loss (smoothed)')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('BiGAN Training History (Smoothed)')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('bigan_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\nBiGAN training completed!")
    
    def generate_synthetic_samples(self, n_samples):
        """Generate synthetic minority class samples using trained generator"""
        print(f"\nGenerating {n_samples} synthetic OUD samples...")
        
        # Sample from latent space
        z = np.random.normal(0, 1, (n_samples, self.latent_dim))
        
        # Generate samples
        synthetic_samples = self.generator.predict(z, verbose=0)
        
        return synthetic_samples
    
    def visualize_generated_samples(self, n_samples=1000):
        """Visualize generated samples compared to real samples"""
        print("\nVisualizing generated samples...")
        
        # Generate synthetic samples
        synthetic = self.generate_synthetic_samples(n_samples)
        
        # Get real minority samples
        minority_indices = np.where(self.y_train == 1)[0]
        real_minority = self.X_train[minority_indices]
        
        # Randomly select features to visualize (top 6 important features)
        feature_indices = [0, 1, 2, 3, 4, 5]  # First 6 features
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, feat_idx in enumerate(feature_indices):
            if feat_idx < self.n_features:
                # Plot distributions
                axes[idx].hist(real_minority[:, feat_idx], bins=30, alpha=0.5, 
                             label='Real OUD', density=True, color='blue')
                axes[idx].hist(synthetic[:, feat_idx], bins=30, alpha=0.5, 
                             label='Synthetic OUD', density=True, color='red')
                axes[idx].set_title(f'Feature {feat_idx + 1}')
                axes[idx].set_xlabel('Value')
                axes[idx].set_ylabel('Density')
                axes[idx].legend()
                axes[idx].grid(True, alpha=0.3)
        
        plt.suptitle('Real vs Synthetic OUD Sample Distributions', fontsize=16)
        plt.tight_layout()
        plt.savefig('bigan_generated_samples_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def build_classifier(self):
        """Build the final classifier for OUD prediction"""
        model = keras.Sequential([
            layers.Input(shape=(self.n_features,)),
            
            # First block
            layers.Dense(256, kernel_regularizer=l1_l2(l1=0.001, l2=0.001)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.4),
            
            # Second block
            layers.Dense(128, kernel_regularizer=l1_l2(l1=0.001, l2=0.001)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.4),
            
            # Third block
            layers.Dense(64, kernel_regularizer=l1_l2(l1=0.001, l2=0.001)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.3),
            
            # Fourth block
            layers.Dense(32, kernel_regularizer=l1_l2(l1=0.001, l2=0.001)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.3),
            
            # Output
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        
        return model
    
    def train_classifier_with_augmentation(self):
        """Train classifier on BiGAN-augmented dataset"""
        print("\n" + "="*60)
        print("TRAINING CLASSIFIER WITH BiGAN AUGMENTATION")
        print("="*60)
        
        # Calculate how many synthetic samples to generate
        minority_count = np.sum(self.y_train == 1)
        majority_count = np.sum(self.y_train == 0)
        
        # Generate enough to balance the dataset (50% minority)
        n_synthetic = int(majority_count * 0.5 - minority_count)
        
        print(f"\nOriginal training set:")
        print(f"Majority class: {majority_count}")
        print(f"Minority class: {minority_count}")
        print(f"Generating {n_synthetic} synthetic samples...")
        
        # Generate synthetic samples
        synthetic_samples = self.generate_synthetic_samples(n_synthetic)
        synthetic_labels = np.ones(n_synthetic)
        
        # Combine with original training data
        X_augmented = np.vstack([self.X_train, synthetic_samples])
        y_augmented = np.hstack([self.y_train, synthetic_labels])
        
        # Shuffle the augmented dataset
        shuffle_indices = np.random.permutation(len(X_augmented))
        X_augmented = X_augmented[shuffle_indices]
        y_augmented = y_augmented[shuffle_indices]
        
        print(f"\nAugmented training set:")
        print(f"Total samples: {len(X_augmented)}")
        print(f"Class distribution: {Counter(y_augmented)}")
        
        # Build and train classifier
        self.classifier = self.build_classifier()
        
        print("\nClassifier architecture:")
        self.classifier.summary()
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_auc',
                patience=25,
                restore_best_weights=True,
                mode='max',
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                'best_bigan_classifier.h5',
                monitor='val_auc',
                mode='max',
                save_best_only=True,
                verbose=0
            )
        ]
        
        # Train classifier
        history = self.classifier.fit(
            X_augmented, y_augmented,
            validation_data=(self.X_val, self.y_val),
            epochs=100,
            batch_size=256,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate_bigan_model(self):
        """Comprehensive evaluation of BiGAN-based classifier"""
        print("\n" + "="*60)
        print("EVALUATING BiGAN-BASED CLASSIFIER")
        print("="*60)
        
        # Get predictions
        y_pred_proba = self.classifier.predict(self.X_test, verbose=0).flatten()
        
        # Find optimal threshold
        val_pred_proba = self.classifier.predict(self.X_val, verbose=0).flatten()
        
        thresholds = np.arange(0.1, 0.9, 0.05)
        f1_scores = []
        
        for threshold in thresholds:
            val_pred = (val_pred_proba > threshold).astype(int)
            f1 = f1_score(self.y_val, val_pred)
            f1_scores.append(f1)
        
        best_threshold = thresholds[np.argmax(f1_scores)]
        print(f"Optimal threshold: {best_threshold:.2f}")
        
        # Make predictions
        y_pred = (y_pred_proba > best_threshold).astype(int)
        
        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
        
        metrics = {
            'Model': 'BiGAN-Augmented NN',
            'Accuracy': accuracy_score(self.y_test, y_pred),
            'Precision': precision_score(self.y_test, y_pred, zero_division=0),
            'Recall': recall_score(self.y_test, y_pred, zero_division=0),
            'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'F1-Score': f1_score(self.y_test, y_pred),
            'F2-Score': fbeta_score(self.y_test, y_pred, beta=2),
            'AUC-ROC': roc_auc_score(self.y_test, y_pred_proba),
            'AUC-PR': average_precision_score(self.y_test, y_pred_proba),
            'MCC': matthews_corrcoef(self.y_test, y_pred),
            'Threshold': best_threshold,
            'TP': tp,
            'TN': tn,
            'FP': fp,
            'FN': fn
        }
        
        # Print results
        print(f"\nBiGAN Model Performance:")
        print(f"Accuracy: {metrics['Accuracy']:.4f}")
        print(f"Precision: {metrics['Precision']:.4f}")
        print(f"Recall (Sensitivity): {metrics['Recall']:.4f}")
        print(f"Specificity: {metrics['Specificity']:.4f}")
        print(f"F1-Score: {metrics['F1-Score']:.4f}")
        print(f"F2-Score: {metrics['F2-Score']:.4f}")
        print(f"AUC-ROC: {metrics['AUC-ROC']:.4f}")
        print(f"AUC-PR: {metrics['AUC-PR']:.4f}")
        print(f"MCC: {metrics['MCC']:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
        
        return metrics
    
    def visualize_results(self, history, metrics):
        """Create comprehensive visualizations"""
        
        # 1. Training history
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # Loss
        axes[0].plot(history.history['loss'], label='Train')
        axes[0].plot(history.history['val_loss'], label='Validation')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # AUC
        axes[1].plot(history.history['auc'], label='Train')
        axes[1].plot(history.history['val_auc'], label='Validation')
        axes[1].set_title('Model AUC')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('AUC')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Precision
        axes[2].plot(history.history['precision'], label='Train')
        axes[2].plot(history.history['val_precision'], label='Validation')
        axes[2].set_title('Model Precision')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Precision')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Recall
        axes[3].plot(history.history['recall'], label='Train')
        axes[3].plot(history.history['val_recall'], label='Validation')
        axes[3].set_title('Model Recall')
        axes[3].set_xlabel('Epoch')
        axes[3].set_ylabel('Recall')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        plt.suptitle('BiGAN Classifier Training History', fontsize=16)
        plt.tight_layout()
        plt.savefig('bigan_classifier_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. ROC and PR curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        y_pred_proba = self.classifier.predict(self.X_test, verbose=0).flatten()
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        ax1.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'BiGAN (AUC = {metrics["AUC-ROC"]:.3f})')
        ax1.plot([0, 1], [0, 1], 'k--', lw=1)
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curve - BiGAN Model')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # PR Curve
        precision, recall, _ = precision_recall_curve(self.y_test, y_pred_proba)
        ax2.plot(recall, precision, color='darkorange', lw=2,
                label=f'BiGAN (AP = {metrics["AUC-PR"]:.3f})')
        baseline = self.y_test.sum() / len(self.y_test)
        ax2.axhline(y=baseline, color='k', linestyle='--', lw=1)
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve - BiGAN Model')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('bigan_roc_pr_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = np.array([[metrics['TN'], metrics['FP']], 
                      [metrics['FN'], metrics['TP']]])
        
        # Calculate percentages
        cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create annotation
        annot_text = np.array([[f"{cm[0,0]}\n({cm_pct[0,0]:.1f}%)", 
                               f"{cm[0,1]}\n({cm_pct[0,1]:.1f}%)"],
                              [f"{cm[1,0]}\n({cm_pct[1,0]:.1f}%)", 
                               f"{cm[1,1]}\n({cm_pct[1,1]:.1f}%)"]])
        
        sns.heatmap(cm, annot=annot_text, fmt='', cmap='Blues', 
                   xticklabels=['Predicted 0', 'Predicted 1'],
                   yticklabels=['Actual 0', 'Actual 1'])
        plt.title('BiGAN Model - Confusion Matrix')
        plt.savefig('bigan_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_complete_bigan_pipeline(self):
        """Run the complete BiGAN pipeline"""
        print("="*80)
        print("BiGAN-BASED OUD PREDICTION PIPELINE")
        print("="*80)
        
        # Step 1: Build BiGAN
        self.build_bigan()
        
        # Step 2: Train BiGAN
        self.train_bigan(epochs=100)
        
        # Step 3: Visualize generated samples
        self.visualize_generated_samples()
        
        # Step 4: Train classifier with augmentation
        history = self.train_classifier_with_augmentation()
        
        # Step 5: Evaluate model
        metrics = self.evaluate_bigan_model()
        
        # Step 6: Visualize results
        self.visualize_results(history, metrics)
        
        # Save results
        results_df = pd.DataFrame([metrics])
        results_df.to_csv('bigan_model_results.csv', index=False)
        
        print("\n" + "="*80)
        print("BiGAN PIPELINE COMPLETED")
        print("="*80)
        print("\nKey Advantages of BiGAN Approach:")
        print("1. Generates high-quality synthetic minority samples")
        print("2. Learns bidirectional mapping between data and latent space")
        print("3. Preserves complex feature relationships")
        print("4. Reduces overfitting through data augmentation")
        print("5. Improves minority class representation")
        
        return metrics

def main():
    """Main function"""
    
    # Check GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print(f"GPU available: {physical_devices[0].name}")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    else:
        print("No GPU found, using CPU")
    
    # Initialize
    bigan_model = BiGANOUDModel(
        data_path='/sharefolder/wanglab/ML_training/final_OUD_ML_dataset.csv'
    )
    
    # Load data
    if not bigan_model.load_and_prepare_data():
        print("Failed to load data!")
        return
    
    # Run complete pipeline
    metrics = bigan_model.run_complete_bigan_pipeline()
    
    print("\nOutput files:")
    print("  - bigan_model_results.csv")
    print("  - bigan_training_history.png")
    print("  - bigan_generated_samples_comparison.png")
    print("  - bigan_classifier_training_history.png")
    print("  - bigan_roc_pr_curves.png")
    print("  - bigan_confusion_matrix.png")
    print("  - best_bigan_classifier.h5")

if __name__ == "__main__":
    main()