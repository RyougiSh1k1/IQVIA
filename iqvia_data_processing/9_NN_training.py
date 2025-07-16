"""
OUD (Opioid Use Disorder) Prediction - Neural Network Models with Full Features
===============================================================================
This script implements three neural network architectures for OUD prediction:
1. Simple Feedforward NN - Baseline deep learning model
2. Wide & Deep NN - Combines memorization and generalization
3. Attention-based NN - Focuses on important feature interactions

Now includes demographic features (age, gender, payment type) and
socioeconomic determinants of health (SDOH) from ZIP-level census data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score
)

# Try to import TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model, regularizers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.utils import plot_model
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    print("Error: TensorFlow not installed. Please install with: pip install tensorflow")
    
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
if HAS_TENSORFLOW:
    tf.random.set_seed(42)

class OUDNeuralNetworks:
    """
    Neural Network models for OUD prediction with focus on 
    handling class imbalance and extracting complex patterns.
    Now includes demographic and SDOH features.
    """
    
    def __init__(self, data_path='/sharefolder/wanglab/ML_training/final_OUD_ML_dataset.csv'):
        self.data_path = data_path
        
        # Define feature groups
        self.mme_features = [
            'MME_last_365_days', 'MME_last_2_years', 'MME_prior_1_year',
            'MME_120_2_years'
        ]
        
        self.prescriber_features = [
            'prscbr_last_2_years', 'prscrbr_last_180_days'
        ]
        
        self.demographic_features = [
            'age', 'der_sex', 'pay_type'
        ]
        
        self.sdoh_features = [
            'age_median', 'income_household_median', 'home_ownership',
            'education_highschool', 'education_college_or_above', 
            'unemployment_rate', 'poverty', 'disabled',
            'race_white', 'race_black', 'race_asian', 'race_native',
            'race_pacific', 'hispanic'
        ]
        
        # Columns to drop
        self.columns_to_drop = [
            'pat_id', 'most_recent_date', 'start_date_180_days', 
            'start_date_2_years'
        ]
        
        self.target_col = 'oud_label'
        
        # Initialize attributes
        self.n_features = None
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
        self.models = {}
        self.histories = {}
        self.results = {}
        self.class_weight = None
        self.encoders = {}
        
    def load_and_prepare_data(self):
        """Load data and prepare for neural network training"""
        print("Loading dataset...")
        df = pd.read_csv(self.data_path)
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Drop specified columns
        print(f"\nDropping columns: {self.columns_to_drop}")
        df = df.drop(columns=[col for col in self.columns_to_drop if col in df.columns])
        
        # Handle categorical variables
        print("\nEncoding categorical variables...")
        
        # Encode gender (assuming F=0, M=1)
        if 'der_sex' in df.columns:
            self.encoders['der_sex'] = LabelEncoder()
            df['der_sex'] = self.encoders['der_sex'].fit_transform(df['der_sex'])
            print(f"Gender encoding: {dict(zip(self.encoders['der_sex'].classes_, self.encoders['der_sex'].transform(self.encoders['der_sex'].classes_)))}")
        
        # Encode payment type
        if 'pay_type' in df.columns:
            self.encoders['pay_type'] = LabelEncoder()
            df['pay_type'] = self.encoders['pay_type'].fit_transform(df['pay_type'])
            print(f"Payment type encoding: {dict(zip(self.encoders['pay_type'].classes_, self.encoders['pay_type'].transform(self.encoders['pay_type'].classes_)))}")
        
        # Extract features and target
        self.y = df[self.target_col].values
        self.X = df.drop(columns=[self.target_col]).values
        self.feature_names = [col for col in df.columns if col != self.target_col]
        self.n_features = len(self.feature_names)
        
        # Calculate class weights for imbalanced data
        n_samples = len(self.y)
        n_classes = 2
        class_counts = np.bincount(self.y)
        self.class_weight = {
            0: n_samples / (n_classes * class_counts[0]),
            1: n_samples / (n_classes * class_counts[1])
        }
        
        print(f"\nClass distribution:")
        print(f"Non-OUD (0): {class_counts[0]} ({class_counts[0]/n_samples*100:.2f}%)")
        print(f"OUD (1): {class_counts[1]} ({class_counts[1]/n_samples*100:.2f}%)")
        print(f"Class weights: {self.class_weight}")
        
        # Split data: 70% train, 15% validation, 15% test
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
        print(f"Number of features: {self.n_features}")
        
        # Print feature groups info
        self._print_feature_group_info()
        
        return True
    
    def _print_feature_group_info(self):
        """Print information about feature groups"""
        print("\n" + "="*60)
        print("FEATURE GROUPS IN NEURAL NETWORK")
        print("="*60)
        
        feature_groups = {
            'MME Features': self.mme_features,
            'Prescriber Features': self.prescriber_features,
            'Demographic Features': self.demographic_features,
            'SDOH Features': self.sdoh_features
        }
        
        total_features = 0
        for group_name, features in feature_groups.items():
            available = [f for f in features if f in self.feature_names]
            total_features += len(available)
            print(f"{group_name}: {len(available)} features")
        
        print(f"\nTotal features used: {total_features}")
        print(f"Total features in dataset: {self.n_features}")
    
    def create_simple_feedforward_nn(self):
        """
        Model 1: Simple Feedforward Neural Network
        - Multiple hidden layers with dropout
        - Batch normalization for stable training
        - L2 regularization to prevent overfitting
        - Adapted for larger feature set
        """
        model = keras.Sequential([
            layers.Input(shape=(self.n_features,)),
            
            # First hidden layer - larger due to more features
            layers.Dense(128, kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dropout(0.3),
            
            # Second hidden layer
            layers.Dense(64, kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dropout(0.3),
            
            # Third hidden layer
            layers.Dense(32, kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dropout(0.2),
            
            # Fourth hidden layer
            layers.Dense(16, kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dropout(0.2),
            
            # Output layer
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        
        return model
    
    def create_wide_deep_nn(self):
        """
        Model 2: Wide & Deep Neural Network
        - Wide component: memorizes feature interactions
        - Deep component: generalizes to unseen patterns
        - Particularly effective for healthcare data with both linear and non-linear patterns
        - Enhanced for demographic and SDOH features
        """
        # Input layer
        input_layer = layers.Input(shape=(self.n_features,))
        
        # Wide component - direct connection from input to output
        # Good for memorizing simple patterns in demographic/SDOH features
        wide = layers.Dense(1, activation='linear')(input_layer)
        
        # Deep component - multiple hidden layers
        deep = layers.Dense(128, activation='relu', 
                           kernel_regularizer=regularizers.l2(0.001))(input_layer)
        deep = layers.BatchNormalization()(deep)
        deep = layers.Dropout(0.3)(deep)
        
        deep = layers.Dense(64, activation='relu',
                           kernel_regularizer=regularizers.l2(0.001))(deep)
        deep = layers.BatchNormalization()(deep)
        deep = layers.Dropout(0.3)(deep)
        
        deep = layers.Dense(32, activation='relu',
                           kernel_regularizer=regularizers.l2(0.001))(deep)
        deep = layers.BatchNormalization()(deep)
        deep = layers.Dropout(0.2)(deep)
        
        deep = layers.Dense(16, activation='relu',
                           kernel_regularizer=regularizers.l2(0.001))(deep)
        deep = layers.BatchNormalization()(deep)
        deep = layers.Dropout(0.2)(deep)
        
        deep_output = layers.Dense(1, activation='linear')(deep)
        
        # Combine wide and deep
        combined = layers.Add()([wide, deep_output])
        output = layers.Activation('sigmoid')(combined)
        
        model = Model(inputs=input_layer, outputs=output)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        
        return model
    
    def create_attention_based_nn(self):
        """
        Model 3: Attention-based Neural Network
        - Self-attention mechanism to identify important feature interactions
        - Particularly useful for understanding which features contribute most to predictions
        - Enhanced to handle larger feature space
        """
        # Input layer
        input_layer = layers.Input(shape=(self.n_features,))
        
        # Feature embedding layer - larger embedding for more features
        embedded = layers.Dense(64, activation='relu')(input_layer)
        embedded = layers.BatchNormalization()(embedded)
        
        # Reshape for attention mechanism
        reshaped = layers.Reshape((1, 64))(embedded)
        
        # Self-attention layer
        attention = layers.MultiHeadAttention(
            num_heads=8,
            key_dim=8,
            dropout=0.2
        )(reshaped, reshaped)
        
        # Add residual connection
        attention = layers.Add()([reshaped, attention])
        attention = layers.LayerNormalization()(attention)
        
        # Flatten back
        flattened = layers.Flatten()(attention)
        
        # Feature interaction layers
        hidden = layers.Dense(128, activation='relu',
                             kernel_regularizer=regularizers.l2(0.001))(flattened)
        hidden = layers.BatchNormalization()(hidden)
        hidden = layers.Dropout(0.3)(hidden)
        
        hidden = layers.Dense(64, activation='relu',
                             kernel_regularizer=regularizers.l2(0.001))(hidden)
        hidden = layers.BatchNormalization()(hidden)
        hidden = layers.Dropout(0.3)(hidden)
        
        hidden = layers.Dense(32, activation='relu',
                             kernel_regularizer=regularizers.l2(0.001))(hidden)
        hidden = layers.BatchNormalization()(hidden)
        hidden = layers.Dropout(0.2)(hidden)
        
        # Output layer
        output = layers.Dense(1, activation='sigmoid')(hidden)
        
        model = Model(inputs=input_layer, outputs=output)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        
        return model
    
    def train_model(self, model, model_name):
        """Train a neural network model with appropriate callbacks"""
        print(f"\nTraining {model_name}...")
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
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
                f'best_{model_name.lower().replace(" ", "_")}_full_features.h5',
                monitor='val_auc',
                mode='max',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        history = model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=100,
            batch_size=256,
            class_weight=self.class_weight,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate_model(self, model, model_name):
        """Evaluate a trained model on test set"""
        print(f"\nEvaluating {model_name}...")
        
        # Get predictions
        y_pred_proba = model.predict(self.X_test).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
        
        metrics = {
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'sensitivity': recall_score(self.y_test, y_pred),
            'specificity': tn / (tn + fp),
            'precision': precision_score(self.y_test, y_pred),
            'f1_score': f1_score(self.y_test, y_pred),
            'auc_roc': roc_auc_score(self.y_test, y_pred_proba),
            'auc_pr': average_precision_score(self.y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(self.y_test, y_pred),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        # Print results
        print(f"\nPerformance Metrics for {model_name}:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Sensitivity (Recall): {metrics['sensitivity']:.4f}")
        print(f"Specificity: {metrics['specificity']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
        print(f"AUC-PR: {metrics['auc_pr']:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
        
        return metrics
    
    def plot_training_history(self):
        """Plot training history for all models"""
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        
        metrics_to_plot = ['loss', 'auc', 'precision', 'recall']
        
        for i, (model_name, history) in enumerate(self.histories.items()):
            for j, metric in enumerate(metrics_to_plot):
                ax = axes[i, j]
                
                # Plot training and validation metrics
                ax.plot(history.history[metric], label=f'Train {metric}')
                ax.plot(history.history[f'val_{metric}'], label=f'Val {metric}')
                
                ax.set_title(f'{model_name} - {metric.upper()}')
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric.capitalize())
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('oud_nn_training_history_full_features.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_comparison(self):
        """Create comprehensive visualizations for NN model comparison"""
        
        # 1. Metrics Comparison
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        metrics_to_plot = [
            ('sensitivity', 'Sensitivity (Recall)'),
            ('specificity', 'Specificity'),
            ('precision', 'Precision'),
            ('f1_score', 'F1-Score'),
            ('auc_roc', 'AUC-ROC'),
            ('auc_pr', 'AUC-PR')
        ]
        
        model_names = list(self.results.keys())
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for idx, (metric, title) in enumerate(metrics_to_plot):
            values = [self.results[model][metric] for model in model_names]
            bars = axes[idx].bar(model_names, values, color=colors)
            axes[idx].set_title(title, fontsize=14, fontweight='bold')
            axes[idx].set_ylim(0, 1.05)
            axes[idx].set_ylabel('Score')
            
            # Add value labels
            for bar, v in zip(bars, values):
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                             f'{v:.3f}', ha='center', va='bottom')
            
            # Rotate x-labels
            axes[idx].tick_params(axis='x', rotation=30)
        
        plt.tight_layout()
        plt.savefig('oud_nn_metrics_comparison_full_features.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. ROC and PR Curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # ROC Curves
        for name, color in zip(model_names, colors):
            fpr, tpr, _ = roc_curve(self.y_test, self.results[name]['y_pred_proba'])
            auc = self.results[name]['auc_roc']
            ax1.plot(fpr, tpr, color=color, lw=2, 
                    label=f'{name} (AUC = {auc:.3f})')
        
        ax1.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
        ax1.set_xlabel('False Positive Rate', fontsize=12)
        ax1.set_ylabel('True Positive Rate', fontsize=12)
        ax1.set_title('ROC Curves - Neural Network Models', fontsize=14, fontweight='bold')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        
        # PR Curves
        for name, color in zip(model_names, colors):
            precision, recall, _ = precision_recall_curve(
                self.y_test, self.results[name]['y_pred_proba']
            )
            auc_pr = self.results[name]['auc_pr']
            ax2.plot(recall, precision, color=color, lw=2,
                    label=f'{name} (AP = {auc_pr:.3f})')
        
        baseline = self.y_test.sum() / len(self.y_test)
        ax2.axhline(y=baseline, color='k', linestyle='--', lw=1,
                   label=f'Baseline (AP = {baseline:.3f})')
        
        ax2.set_xlabel('Recall', fontsize=12)
        ax2.set_ylabel('Precision', fontsize=12)
        ax2.set_title('Precision-Recall Curves - Neural Network Models', fontsize=14, fontweight='bold')
        ax2.legend(loc='lower left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('oud_nn_roc_pr_curves_full_features.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_summary_report(self):
        """Generate comprehensive summary report for NN models"""
        print("\n" + "="*80)
        print("NEURAL NETWORK MODELS - SUMMARY REPORT (WITH FULL FEATURES)")
        print("="*80)
        
        # Create summary DataFrame
        summary_data = []
        for model_name, metrics in self.results.items():
            summary_data.append({
                'Model': model_name,
                'Architecture': self._get_architecture_summary(model_name),
                'AUC-ROC': f"{metrics['auc_roc']:.4f}",
                'AUC-PR': f"{metrics['auc_pr']:.4f}",
                'Sensitivity': f"{metrics['sensitivity']:.4f}",
                'Specificity': f"{metrics['specificity']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        print("\nModel Performance Summary:")
        print(summary_df.to_string(index=False))
        
        # Model-specific insights
        print("\n" + "-"*80)
        print("MODEL-SPECIFIC INSIGHTS:")
        print("-"*80)
        
        print("\n1. Simple Feedforward NN:")
        print("   - Best for: Baseline deep learning performance")
        print("   - Advantages: Fast training, straightforward interpretation")
        print("   - Enhanced with 4 layers to capture complex SDOH patterns")
        
        print("\n2. Wide & Deep NN:")
        print("   - Best for: Capturing both memorization and generalization")
        print("   - Advantages: Handles linear demographic features + non-linear clinical patterns")
        print("   - Wide component captures direct SDOH effects")
        
        print("\n3. Attention-based NN:")
        print("   - Best for: Understanding feature importance dynamically")
        print("   - Advantages: Can identify which combination of clinical/demographic/SDOH features matter")
        print("   - Attention weights can reveal interaction patterns")
        
        # Feature set insights
        print("\n" + "-"*80)
        print("FEATURE SET INSIGHTS:")
        print("-"*80)
        print("\nThe neural networks now process:")
        print(f"• {len(self.mme_features)} MME features (prescription patterns)")
        print(f"• {len(self.prescriber_features)} Prescriber features (healthcare utilization)")
        print(f"• {len(self.demographic_features)} Demographic features (age, gender, insurance)")
        print(f"• {len(self.sdoh_features)} SDOH features (socioeconomic factors)")
        print(f"• Total: {self.n_features} features")
        
        # Clinical deployment recommendations
        print("\n" + "-"*80)
        print("CLINICAL DEPLOYMENT RECOMMENDATIONS:")
        print("-"*80)
        
        best_sensitivity = max(self.results.items(), key=lambda x: x[1]['sensitivity'])
        best_balanced = max(self.results.items(), key=lambda x: x[1]['f1_score'])
        
        print(f"\n For Clinical Screening:")
        print(f"   → Deploy: {best_sensitivity[0]}")
        print(f"   → Sensitivity: {best_sensitivity[1]['sensitivity']:.4f}")
        print(f"   → Rationale: Maximizes detection of OUD cases")
        
        print(f"\n For Balanced Performance:")
        print(f"   → Deploy: {best_balanced[0]}")
        print(f"   → F1-Score: {best_balanced[1]['f1_score']:.4f}")
        print(f"   → Rationale: Best trade-off between precision and recall")
        
        # Computational considerations
        print("\n" + "-"*80)
        print("COMPUTATIONAL REQUIREMENTS:")
        print("-"*80)
        
        for model_name, model in self.models.items():
            total_params = model.count_params()
            print(f"\n{model_name}:")
            print(f"   Total parameters: {total_params:,}")
            print(f"   Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
            print(f"   Input features: {self.n_features}")
        
        # Save summary
        summary_df.to_csv('oud_nn_model_summary_full_features.csv', index=False)
        print("\n✓ Summary saved to 'oud_nn_model_summary_full_features.csv'")
        
        return summary_df
    
    def _get_architecture_summary(self, model_name):
        """Get brief architecture summary for each model"""
        summaries = {
            'Simple Feedforward': f'4 hidden layers (128-64-32-16), {self.n_features} inputs',
            'Wide & Deep': f'Wide path + 4 deep layers, {self.n_features} inputs',
            'Attention-based': f'Multi-head attention (8 heads) + 3 layers, {self.n_features} inputs'
        }
        return summaries.get(model_name, 'Custom architecture')
    
    def run_complete_evaluation(self):
        """Run the complete neural network evaluation pipeline"""
        if not HAS_TENSORFLOW:
            print("ERROR: TensorFlow is required but not installed.")
            print("Please install with: pip install tensorflow")
            return None
            
        print("Starting Neural Network Model Evaluation for OUD Prediction")
        print("   Using Full Feature Set (Clinical + Demographic + SDOH)")
        print("="*80)
        
        # Load and prepare data
        if not self.load_and_prepare_data():
            return None
        
        # Create models
        print("\nCreating neural network models...")
        self.models = {
            'Simple Feedforward': self.create_simple_feedforward_nn(),
            'Wide & Deep': self.create_wide_deep_nn(),
            'Attention-based': self.create_attention_based_nn()
        }
        
        # Train and evaluate each model
        for model_name, model in self.models.items():
            print(f"\n{'='*60}")
            print(f"Processing {model_name}")
            print('='*60)
            
            # Print model summary
            print("\nModel Architecture:")
            model.summary()
            
            # Train model
            history = self.train_model(model, model_name)
            self.histories[model_name] = history
            
            # Evaluate model
            metrics = self.evaluate_model(model, model_name)
            self.results[model_name] = metrics
        
        # Visualizations
        print("\nGenerating visualizations...")
        self.plot_training_history()
        self.plot_model_comparison()
        
        # Generate summary
        summary = self.generate_summary_report()
        
        print("\n Neural Network evaluation completed successfully!")
        
        return {
            'results': self.results,
            'summary': summary,
            'histories': self.histories
        }

def main():
    """Main function to run neural network evaluation"""
    
    # Check if TensorFlow is available
    if not HAS_TENSORFLOW:
        print("ERROR: TensorFlow is not installed.")
        print("Please install it with: pip install tensorflow")
        print("For GPU support: pip install tensorflow-gpu")
        return
    
    # Enable GPU if available
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print(f"GPU available: {physical_devices[0].name}")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    else:
        print("No GPU found, using CPU")
    
    # Initialize evaluator with your dataset path
    evaluator = OUDNeuralNetworks(
        data_path='/sharefolder/wanglab/ML_training/final_OUD_ML_dataset.csv'
    )
    
    # Run evaluation
    results = evaluator.run_complete_evaluation()
    
    if results:
        print("\n" + "="*80)
        print("EVALUATION COMPLETE")
        print("="*80)
        print(" All neural network models trained and evaluated")
        print(" Training history plots saved")
        print(" Model comparison visualizations saved")
        print(" Summary report generated")
        print("\n Output files:")
        print("  - oud_nn_training_history_full_features.png")
        print("  - oud_nn_metrics_comparison_full_features.png")
        print("  - oud_nn_roc_pr_curves_full_features.png")
        print("  - oud_nn_model_summary_full_features.csv")
        print("  - best_*.h5 (saved model weights)")

if __name__ == "__main__":
    main()