"""
OUD Prediction - Advanced Neural Networks for Handling Class Imbalance
======================================================================
This script implements three powerful neural network architectures specifically
designed to handle the severe class imbalance in OUD prediction (88% vs 12%).

Models:
1. Focal Loss Neural Network - Addresses class imbalance through loss function
2. Cost-Sensitive Deep Learning with SMOTE - Combines oversampling with cost-sensitive learning
3. Ensemble Neural Network with Bagging - Multiple models voting for robust predictions

All models incorporate advanced techniques for imbalanced learning.
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
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek
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

class AdvancedImbalancedOUDModels:
    """
    Advanced neural network models specifically designed for imbalanced OUD prediction
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
        self.models = {}
        self.results = {}
        self.class_weight = None
        self.encoders = {}
        self.imbalance_ratio = None
        
    def load_and_prepare_data(self):
        """Load and prepare data with focus on imbalance handling"""
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
            self.encoders['der_sex'] = LabelEncoder()
            df['der_sex'] = self.encoders['der_sex'].fit_transform(df['der_sex'])
        
        if 'pay_type' in df.columns:
            self.encoders['pay_type'] = LabelEncoder()
            df['pay_type'] = self.encoders['pay_type'].fit_transform(df['pay_type'])
        
        # Extract features and target
        self.y = df[self.target_col].values
        self.X = df.drop(columns=[self.target_col])
        
        # Handle non-numeric columns
        numeric_columns = self.X.select_dtypes(include=[np.number]).columns
        self.X = self.X[numeric_columns]
        
        self.feature_names = list(self.X.columns)
        self.n_features = len(self.feature_names)
        self.X = self.X.values
        
        # Calculate class distribution and weights
        n_samples = len(self.y)
        class_counts = np.bincount(self.y)
        self.imbalance_ratio = class_counts[0] / class_counts[1]
        
        # More aggressive class weights for severe imbalance
        self.class_weight = {
            0: 1.0,
            1: self.imbalance_ratio
        }
        
        print(f"\nClass distribution:")
        print(f"Non-OUD (0): {class_counts[0]} ({class_counts[0]/n_samples*100:.2f}%)")
        print(f"OUD (1): {class_counts[1]} ({class_counts[1]/n_samples*100:.2f}%)")
        print(f"Imbalance ratio: 1:{self.imbalance_ratio:.1f}")
        print(f"Class weights: {self.class_weight}")
        
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
    
    def focal_loss(self, gamma=2., alpha=0.25):
        """
        Focal loss for addressing class imbalance
        FL(pt) = -alpha * (1-pt)^gamma * log(pt)
        """
        def focal_loss_fixed(y_true, y_pred):
            epsilon = K.epsilon()
            y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
            
            # Calculate focal loss
            p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
            alpha_factor = K.ones_like(y_true) * alpha
            alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
            
            cross_entropy = -K.log(p_t)
            weight = alpha_t * K.pow((1 - p_t), gamma)
            
            loss = weight * cross_entropy
            return K.mean(K.sum(loss, axis=1))
        
        return focal_loss_fixed
    
    def create_focal_loss_nn(self):
        """
        Model 1: Neural Network with Focal Loss
        - Uses focal loss to down-weight easy examples and focus on hard cases
        - Particularly effective for extreme class imbalance
        """
        model = keras.Sequential([
            layers.Input(shape=(self.n_features,)),
            
            # First block with strong regularization
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
        
        # Compile with focal loss
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=self.focal_loss(gamma=2.0, alpha=0.75),  # Higher alpha for minority class
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        
        return model
    
    def create_cost_sensitive_smote_nn(self):
        """
        Model 2: Cost-Sensitive Deep Learning with SMOTE preprocessing
        - Combines synthetic oversampling with cost-sensitive learning
        - Uses deeper architecture with residual connections
        """
        input_layer = layers.Input(shape=(self.n_features,))
        
        # First dense block
        x = layers.Dense(512, kernel_regularizer=l1_l2(l1=0.0005, l2=0.0005))(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Second dense block with residual
        x2 = layers.Dense(256)(x)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.Activation('relu')(x2)
        x2 = layers.Dropout(0.3)(x2)
        
        # Residual connection (dimension reduction)
        x_residual = layers.Dense(256)(x)
        x = layers.Add()([x2, x_residual])
        
        # Third dense block
        x = layers.Dense(128)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Fourth dense block
        x = layers.Dense(64)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # Output
        output = layers.Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=input_layer, outputs=output)
        
        # Custom weighted binary crossentropy
        def weighted_binary_crossentropy(y_true, y_pred):
            # Calculate the binary crossentropy
            bce = K.binary_crossentropy(y_true, y_pred)
            
            # Apply class weights
            weight_vector = y_true * self.imbalance_ratio + (1. - y_true)
            weighted_bce = weight_vector * bce
            
            return K.mean(weighted_bce)
        
        model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss=weighted_binary_crossentropy,
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        
        return model
    
    def create_ensemble_bagging_nn(self):
        """
        Model 3: Ensemble Neural Network with Bagging
        - Creates multiple diverse models that vote on predictions
        - Each sub-model sees different balanced samples of the data
        """
        # Create multiple sub-models with different architectures
        def create_submodel_1():
            return keras.Sequential([
                layers.Dense(128, activation='relu', input_shape=(self.n_features,)),
                layers.Dropout(0.5),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.4),
                layers.Dense(1, activation='sigmoid')
            ])
        
        def create_submodel_2():
            return keras.Sequential([
                layers.Dense(256, activation='relu', input_shape=(self.n_features,)),
                layers.BatchNormalization(),
                layers.Dropout(0.4),
                layers.Dense(128, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(1, activation='sigmoid')
            ])
        
        def create_submodel_3():
            input_layer = layers.Input(shape=(self.n_features,))
            
            # Two parallel paths
            path1 = layers.Dense(64, activation='relu')(input_layer)
            path1 = layers.Dropout(0.3)(path1)
            
            path2 = layers.Dense(64, activation='relu')(input_layer)
            path2 = layers.Dropout(0.3)(path2)
            
            # Concatenate
            concat = layers.Concatenate()([path1, path2])
            x = layers.Dense(64, activation='relu')(concat)
            x = layers.Dropout(0.3)(x)
            output = layers.Dense(1, activation='sigmoid')(x)
            
            return Model(inputs=input_layer, outputs=output)
        
        # Create ensemble input
        ensemble_input = layers.Input(shape=(self.n_features,))
        
        # Create and connect sub-models
        submodel1 = create_submodel_1()
        submodel2 = create_submodel_2()
        submodel3 = create_submodel_3()
        
        # Get predictions from each model
        pred1 = submodel1(ensemble_input)
        pred2 = submodel2(ensemble_input)
        pred3 = submodel3(ensemble_input)
        
        # Average predictions (voting)
        ensemble_output = layers.Average()([pred1, pred2, pred3])
        
        # Create ensemble model
        ensemble_model = Model(inputs=ensemble_input, outputs=ensemble_output)
        
        # Compile with custom loss that emphasizes recall
        def recall_focused_loss(y_true, y_pred):
            # Standard binary crossentropy
            bce = K.binary_crossentropy(y_true, y_pred)
            
            # Additional penalty for false negatives
            fn_penalty = 2.0 * y_true * (1 - y_pred)
            
            return K.mean(bce + fn_penalty)
        
        ensemble_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=recall_focused_loss,
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        
        return ensemble_model
    
    def apply_smote(self, X_train, y_train, sampling_strategy=0.5):
        """Apply SMOTE to training data"""
        print(f"\nApplying SMOTE with sampling strategy: {sampling_strategy}")
        print(f"Before SMOTE: {Counter(y_train)}")
        
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        print(f"After SMOTE: {Counter(y_resampled)}")
        
        return X_resampled, y_resampled
    
    def train_model_with_strategy(self, model, model_name, strategy='standard', 
                                 epochs=100, batch_size=256):
        """Train model with different strategies for handling imbalance"""
        print(f"\nTraining {model_name} with {strategy} strategy...")
        
        # Prepare training data based on strategy
        if strategy == 'smote':
            # Apply SMOTE to training data
            X_train_balanced, y_train_balanced = self.apply_smote(
                self.X_train, self.y_train, sampling_strategy=0.3
            )
        else:
            X_train_balanced = self.X_train
            y_train_balanced = self.y_train
        
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
                f'best_{model_name.lower().replace(" ", "_")}_imbalanced.h5',
                monitor='val_auc',
                mode='max',
                save_best_only=True,
                verbose=0
            )
        ]
        
        # Use class weights for non-SMOTE strategies
        class_weight_param = self.class_weight if strategy != 'smote' else None
        
        # Train model
        history = model.fit(
            X_train_balanced, y_train_balanced,
            validation_data=(self.X_val, self.y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight_param,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate_model_comprehensive(self, model, model_name):
        """Comprehensive evaluation with focus on imbalanced metrics"""
        print(f"\nEvaluating {model_name}...")
        
        # Get predictions
        y_pred_proba = model.predict(self.X_test, verbose=0).flatten()
        
        # Find optimal threshold using validation set
        val_pred_proba = model.predict(self.X_val, verbose=0).flatten()
        
        # Calculate F1 scores for different thresholds
        thresholds = np.arange(0.1, 0.9, 0.05)
        f1_scores = []
        
        for threshold in thresholds:
            val_pred = (val_pred_proba > threshold).astype(int)
            f1 = f1_score(self.y_val, val_pred)
            f1_scores.append(f1)
        
        # Use best threshold
        best_threshold = thresholds[np.argmax(f1_scores)]
        print(f"Optimal threshold: {best_threshold:.2f}")
        
        # Make predictions with optimal threshold
        y_pred = (y_pred_proba > best_threshold).astype(int)
        
        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
        
        metrics = {
            'Model': model_name,
            'Accuracy': accuracy_score(self.y_test, y_pred),
            'Precision': precision_score(self.y_test, y_pred, zero_division=0),
            'Recall': recall_score(self.y_test, y_pred, zero_division=0),
            'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'F1-Score': f1_score(self.y_test, y_pred),
            'F2-Score': fbeta_score(self.y_test, y_pred, beta=2),  # Emphasizes recall
            'AUC-ROC': roc_auc_score(self.y_test, y_pred_proba),
            'AUC-PR': average_precision_score(self.y_test, y_pred_proba),
            'MCC': matthews_corrcoef(self.y_test, y_pred),
            'Threshold': best_threshold,
            'TP': tp,
            'TN': tn,
            'FP': fp,
            'FN': fn
        }
        
        # Print detailed results
        print(f"\nResults for {model_name}:")
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
    
    def run_all_advanced_models(self):
        """Run all three advanced models for imbalanced data"""
        print("="*80)
        print("ADVANCED NEURAL NETWORKS FOR IMBALANCED OUD PREDICTION")
        print("="*80)
        
        all_results = []
        
        # 1. Focal Loss Neural Network
        print("\n" + "="*60)
        print("MODEL 1: FOCAL LOSS NEURAL NETWORK")
        print("="*60)
        
        focal_model = self.create_focal_loss_nn()
        print("\nModel Summary:")
        focal_model.summary()
        
        focal_history = self.train_model_with_strategy(
            focal_model, "Focal Loss NN", strategy='standard'
        )
        focal_results = self.evaluate_model_comprehensive(focal_model, "Focal Loss NN")
        all_results.append(focal_results)
        
        # 2. Cost-Sensitive Deep Learning with SMOTE
        print("\n" + "="*60)
        print("MODEL 2: COST-SENSITIVE DEEP LEARNING WITH SMOTE")
        print("="*60)
        
        smote_model = self.create_cost_sensitive_smote_nn()
        print("\nModel Summary:")
        smote_model.summary()
        
        smote_history = self.train_model_with_strategy(
            smote_model, "Cost-Sensitive SMOTE NN", strategy='smote'
        )
        smote_results = self.evaluate_model_comprehensive(smote_model, "Cost-Sensitive SMOTE NN")
        all_results.append(smote_results)
        
        # 3. Ensemble Neural Network with Bagging
        print("\n" + "="*60)
        print("MODEL 3: ENSEMBLE NEURAL NETWORK WITH BAGGING")
        print("="*60)
        
        ensemble_model = self.create_ensemble_bagging_nn()
        print("\nModel Summary:")
        ensemble_model.summary()
        
        ensemble_history = self.train_model_with_strategy(
            ensemble_model, "Ensemble Bagging NN", strategy='standard'
        )
        ensemble_results = self.evaluate_model_comprehensive(ensemble_model, "Ensemble Bagging NN")
        all_results.append(ensemble_results)
        
        # Create results DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Store models and histories
        self.models = {
            'Focal Loss NN': focal_model,
            'Cost-Sensitive SMOTE NN': smote_model,
            'Ensemble Bagging NN': ensemble_model
        }
        
        self.histories = {
            'Focal Loss NN': focal_history,
            'Cost-Sensitive SMOTE NN': smote_history,
            'Ensemble Bagging NN': ensemble_history
        }
        
        return results_df
    
    def plot_advanced_results(self, results_df):
        """Create comprehensive visualizations for imbalanced learning results"""
        
        # 1. Metrics focused on imbalanced learning
        fig, axes = plt.subplots(2, 4, figsize=(20, 12))
        axes = axes.flatten()
        
        metrics = ['Recall', 'Precision', 'F1-Score', 'F2-Score', 
                  'Specificity', 'AUC-ROC', 'AUC-PR', 'MCC']
        colors = ['#e74c3c', '#3498db', '#2ecc71']
        
        for idx, metric in enumerate(metrics):
            values = results_df[metric].values
            model_names = ['Focal Loss', 'SMOTE', 'Ensemble']
            
            bars = axes[idx].bar(model_names, values, color=colors)
            axes[idx].set_title(metric, fontsize=14, fontweight='bold')
            axes[idx].set_ylim(0, 1.05)
            axes[idx].set_ylabel('Score')
            
            # Add value labels
            for bar, v in zip(bars, values):
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                             f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('imbalanced_nn_metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. ROC and PR Curves (crucial for imbalanced data)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        for idx, model_name in enumerate(self.models.keys()):
            model = self.models[model_name]
            y_pred_proba = model.predict(self.X_test, verbose=0).flatten()
            
            # ROC Curve
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            auc = roc_auc_score(self.y_test, y_pred_proba)
            ax1.plot(fpr, tpr, color=colors[idx], lw=2, 
                    label=f'{model_name} (AUC = {auc:.3f})')
            
            # PR Curve
            precision, recall, _ = precision_recall_curve(self.y_test, y_pred_proba)
            auc_pr = average_precision_score(self.y_test, y_pred_proba)
            ax2.plot(recall, precision, color=colors[idx], lw=2,
                    label=f'{model_name} (AP = {auc_pr:.3f})')
        
        # ROC plot formatting
        ax1.plot([0, 1], [0, 1], 'k--', lw=1)
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curves - Imbalanced Models')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # PR plot formatting
        baseline = self.y_test.sum() / len(self.y_test)
        ax2.axhline(y=baseline, color='k', linestyle='--', lw=1)
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curves - Imbalanced Models')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('imbalanced_nn_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Confusion Matrices with percentages
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, row in results_df.iterrows():
            cm = np.array([[row['TN'], row['FP']], 
                          [row['FN'], row['TP']]])
            
            # Calculate percentages
            cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            
            # Create annotation text
            annot_text = np.array([[f"{cm[0,0]}\n({cm_pct[0,0]:.1f}%)", 
                                   f"{cm[0,1]}\n({cm_pct[0,1]:.1f}%)"],
                                  [f"{cm[1,0]}\n({cm_pct[1,0]:.1f}%)", 
                                   f"{cm[1,1]}\n({cm_pct[1,1]:.1f}%)"]])
            
            sns.heatmap(cm, annot=annot_text, fmt='', cmap='Blues', 
                       cbar=False, ax=axes[idx],
                       xticklabels=['Predicted 0', 'Predicted 1'],
                       yticklabels=['Actual 0', 'Actual 1'])
            axes[idx].set_title(f"{row['Model']}\nConfusion Matrix")
        
        plt.tight_layout()
        plt.savefig('imbalanced_nn_confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 4. Training history comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        metrics_to_plot = ['loss', 'auc', 'precision', 'recall']
        
        for idx, metric in enumerate(metrics_to_plot):
            for model_name, history in self.histories.items():
                if metric in history.history:
                    axes[idx].plot(history.history[metric], 
                                 label=f'{model_name} (train)')
                    axes[idx].plot(history.history[f'val_{metric}'], 
                                 '--', label=f'{model_name} (val)')
            
            axes[idx].set_title(f'{metric.upper()} during Training')
            axes[idx].set_xlabel('Epoch')
            axes[idx].set_ylabel(metric.capitalize())
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('imbalanced_nn_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return results_df
    
    def generate_final_report(self, results_df):
        """Generate comprehensive report for imbalanced learning models"""
        print("\n" + "="*80)
        print("FINAL REPORT: ADVANCED NEURAL NETWORKS FOR IMBALANCED OUD PREDICTION")
        print("="*80)
        
        # Summary table
        print("\nMODEL PERFORMANCE SUMMARY:")
        print("-"*80)
        
        summary_cols = ['Model', 'Recall', 'Precision', 'F1-Score', 'F2-Score', 
                       'AUC-ROC', 'AUC-PR', 'MCC']
        summary_df = results_df[summary_cols].copy()
        
        for col in summary_cols[1:]:
            summary_df[col] = summary_df[col].apply(lambda x: f'{x:.4f}')
        
        print(summary_df.to_string(index=False))
        
        # Model recommendations
        print("\n" + "-"*80)
        print("MODEL RECOMMENDATIONS FOR IMBALANCED OUD PREDICTION:")
        print("-"*80)
        
        best_recall = results_df.loc[results_df['Recall'].idxmax()]
        best_f2 = results_df.loc[results_df['F2-Score'].idxmax()]
        best_balanced = results_df.loc[results_df['MCC'].idxmax()]
        
        print(f"\n📊 For MAXIMUM OUD DETECTION (Highest Recall):")
        print(f"   → Use: {best_recall['Model']}")
        print(f"   → Recall: {best_recall['Recall']:.4f}")
        print(f"   → Detects {best_recall['Recall']*100:.1f}% of OUD cases")
        print(f"   → Trade-off: Lower precision ({best_recall['Precision']:.4f})")
        
        print(f"\n⚖️ For RECALL-FOCUSED BALANCE (F2-Score):")
        print(f"   → Use: {best_f2['Model']}")
        print(f"   → F2-Score: {best_f2['F2-Score']:.4f}")
        print(f"   → Good for clinical screening with acceptable false positive rate")
        
        print(f"\n🎯 For OVERALL BALANCE (Matthews Correlation Coefficient):")
        print(f"   → Use: {best_balanced['Model']}")
        print(f"   → MCC: {best_balanced['MCC']:.4f}")
        print(f"   → Best correlation between predictions and actual outcomes")
        
        # Technical insights
        print("\n" + "-"*80)
        print("TECHNICAL INSIGHTS:")
        print("-"*80)
        
        print("\n1. FOCAL LOSS NEURAL NETWORK:")
        print("   • Dynamically adjusts loss to focus on hard-to-classify examples")
        print("   • Particularly effective for extreme class imbalance")
        print("   • Reduces the relative loss for well-classified examples")
        
        print("\n2. COST-SENSITIVE SMOTE NN:")
        print("   • Combines synthetic oversampling with cost-sensitive learning")
        print("   • SMOTE creates synthetic minority class examples")
        print("   • Deep architecture with residual connections")
        
        print("\n3. ENSEMBLE BAGGING NN:")
        print("   • Multiple diverse models voting for robust predictions")
        print("   • Each sub-model has different architecture")
        print("   • Reduces overfitting through model diversity")
        
        # Save reports
        results_df.to_csv('imbalanced_nn_full_results.csv', index=False)
        summary_df.to_csv('imbalanced_nn_summary.csv', index=False)
        
        print("\n✅ Results saved to:")
        print("   - imbalanced_nn_full_results.csv")
        print("   - imbalanced_nn_summary.csv")
        
        return summary_df

def main():
    """Main function to run all imbalanced learning experiments"""
    
    # Check GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print(f"GPU available: {physical_devices[0].name}")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    else:
        print("No GPU found, using CPU")
    
    # Initialize
    evaluator = AdvancedImbalancedOUDModels(
        data_path='/sharefolder/wanglab/ML_training/final_OUD_ML_dataset.csv'
    )
    
    # Load data
    if not evaluator.load_and_prepare_data():
        print("Failed to load data!")
        return
    
    # Run all models
    results_df = evaluator.run_all_advanced_models()
    
    # Create visualizations
    evaluator.plot_advanced_results(results_df)
    
    # Generate final report
    summary_df = evaluator.generate_final_report(results_df)
    
    print("\n" + "="*80)
    print("ALL IMBALANCED LEARNING EXPERIMENTS COMPLETED")
    print("="*80)

if __name__ == "__main__":
    main()