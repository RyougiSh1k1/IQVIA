"""
OUD (Opioid Use Disorder) Prediction - Simple Neural Network Models
===================================================================
This script implements the specific neural network architectures from the CSV:
1. Single Layer NN - One fully connected layer with 3 configurations:
   - LR=0.01, Epochs=100
   - LR=0.001, Epochs=100
   - LR=0.001, Epochs=200
2. 3 Layer NN - Three fully connected layers (128->64->1)

All models use Adam optimizer and Binary Cross Entropy Loss.
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
    accuracy_score
)
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class SimpleOUDNeuralNetworks:
    """
    Simple Neural Network models for OUD prediction:
    - Single Layer NN (with 3 configurations)
    - 3 Layer NN
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
        
    def load_and_prepare_data(self):
        """Load and prepare data for neural network training"""
        print("Loading dataset...")
        df = pd.read_csv(self.data_path)
        print(f"Initial dataset shape: {df.shape}")
        
        # Drop unnecessary columns
        columns_to_drop = [col for col in self.columns_to_drop if col in df.columns]
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
            print(f"Dropped columns: {columns_to_drop}")
        
        # Handle categorical variables
        print("\nEncoding categorical variables...")
        
        # Encode gender if present
        if 'der_sex' in df.columns:
            self.encoders['der_sex'] = LabelEncoder()
            df['der_sex'] = self.encoders['der_sex'].fit_transform(df['der_sex'])
        
        # Encode payment type if present
        if 'pay_type' in df.columns:
            self.encoders['pay_type'] = LabelEncoder()
            df['pay_type'] = self.encoders['pay_type'].fit_transform(df['pay_type'])
        
        # Extract features and target
        self.y = df[self.target_col].values
        self.X = df.drop(columns=[self.target_col])
        
        # Handle any remaining non-numeric columns
        numeric_columns = self.X.select_dtypes(include=[np.number]).columns
        self.X = self.X[numeric_columns]
        
        self.feature_names = list(self.X.columns)
        self.n_features = len(self.feature_names)
        
        print(f"\nFinal feature count: {self.n_features}")
        print(f"Features: {self.feature_names[:10]}..." if len(self.feature_names) > 10 else f"Features: {self.feature_names}")
        
        # Convert to numpy arrays
        self.X = self.X.values
        
        # Calculate class weights
        n_samples = len(self.y)
        class_counts = np.bincount(self.y)
        self.class_weight = {
            0: n_samples / (2 * class_counts[0]),
            1: n_samples / (2 * class_counts[1])
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
        
        return True
    
    def create_single_layer_nn(self, learning_rate=0.01):
        """
        Single Layer NN: input -> FC -> sigmoid
        Just one fully connected layer directly to output
        """
        model = keras.Sequential([
            layers.Input(shape=(self.n_features,)),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        return model
    
    def create_three_layer_nn(self, learning_rate=0.01):
        """
        3 Layer NN: input -> Linear(128) -> Linear(64) -> Linear(1) -> Sigmoid
        Three fully connected layers as specified
        """
        model = keras.Sequential([
            layers.Input(shape=(self.n_features,)),
            layers.Dense(128, activation='linear'),
            layers.Dense(64, activation='linear'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        return model
    
    def train_model(self, model, model_name, epochs=100, batch_size=256, use_callbacks=True):
        """Train a neural network model"""
        print(f"\nTraining {model_name}...")
        
        callbacks = []
        if use_callbacks:
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=20,
                    restore_best_weights=True,
                    verbose=0
                ),
                ModelCheckpoint(
                    f'best_{model_name.lower().replace(" ", "_").replace("=", "").replace(".", "_")}.h5',
                    monitor='val_auc',
                    mode='max',
                    save_best_only=True,
                    verbose=0
                )
            ]
        
        # Train model
        history = model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=self.class_weight,
            callbacks=callbacks,
            verbose=0
        )
        
        # Print training progress
        print(f"Training completed - Final epoch {len(history.history['loss'])}")
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        final_train_auc = history.history['auc'][-1]
        final_val_auc = history.history['val_auc'][-1]
        
        print(f"Final - Train Loss: {final_train_loss:.4f}, Val Loss: {final_val_loss:.4f}")
        print(f"Final - Train AUC: {final_train_auc:.4f}, Val AUC: {final_val_auc:.4f}")
        
        return history
    
    def evaluate_model_detailed(self, model, model_name, model_structure, lr, epochs):
        """Evaluate model and return detailed metrics matching CSV format"""
        print(f"\nEvaluating {model_name}...")
        
        # Get predictions
        y_pred_proba = model.predict(self.X_test, verbose=0).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
        
        # Calculate metrics
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        recall = recall_score(self.y_test, y_pred, zero_division=0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        auc_roc = roc_auc_score(self.y_test, y_pred_proba)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        # Total observations
        total_obs = len(self.y_test)
        
        # Store results in format matching CSV
        result = {
            'Model Name': model_name,
            'Model Structure': model_structure,
            'Optimizer': 'Adam',
            'Criterion': 'BCELoss',
            'Epochs': epochs,
            'Learning Rate': lr,
            'Cross-Validation': None,  # To be calculated
            'Precision': precision,
            'Recall': recall,
            'AUC-ROC': auc_roc,
            'TP': tp,
            'TN': tn,
            'FP': fp,
            'FN': fn,
            'NPV': npv,
            'Total Obs': total_obs,
            'Specificity': specificity,
            'Accuracy': accuracy
        }
        
        # Print detailed results
        print(f"\nDetailed Results for {model_name}:")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"NPV: {npv:.4f}")
        print(f"AUC-ROC: {auc_roc:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
        print(f"Total observations: {total_obs}")
        
        return result
    
    def run_cross_validation(self, model_fn, model_name, n_splits=5, **model_params):
        """Run stratified k-fold cross-validation"""
        print(f"\nRunning {n_splits}-fold cross-validation for {model_name}...")
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = []
        
        # Combine train and validation data for CV
        X_cv = np.vstack([self.X_train, self.X_val])
        y_cv = np.hstack([self.y_train, self.y_val])
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_cv, y_cv)):
            print(f"  Fold {fold + 1}/{n_splits}...", end='\r')
            
            # Split data
            X_train_fold = X_cv[train_idx]
            y_train_fold = y_cv[train_idx]
            X_val_fold = X_cv[val_idx]
            y_val_fold = y_cv[val_idx]
            
            # Create and train model
            model = model_fn(**model_params)
            
            # Train with fewer epochs for CV
            model.fit(
                X_train_fold, y_train_fold,
                validation_data=(X_val_fold, y_val_fold),
                epochs=50,
                batch_size=256,
                class_weight=self.class_weight,
                verbose=0
            )
            
            # Evaluate on validation fold
            y_pred_proba = model.predict(X_val_fold, verbose=0).flatten()
            auc_score = roc_auc_score(y_val_fold, y_pred_proba)
            cv_scores.append(auc_score)
            
            # Clear model to free memory
            tf.keras.backend.clear_session()
        
        mean_cv_score = np.mean(cv_scores)
        std_cv_score = np.std(cv_scores)
        
        print(f"\nCross-validation AUC: {mean_cv_score:.4f} (±{std_cv_score:.4f})")
        
        return mean_cv_score
    
    def run_all_experiments(self):
        """Run all experiments matching the CSV specifications"""
        print("="*80)
        print("RUNNING SIMPLE NEURAL NETWORK EXPERIMENTS")
        print("="*80)
        
        all_results = []
        
        # 1. Single Layer NN - Configuration 1: LR=0.01, Epochs=100
        print("\n" + "="*60)
        print("EXPERIMENT 1: Single Layer NN (LR=0.01, Epochs=100)")
        print("="*60)
        
        model1 = self.create_single_layer_nn(learning_rate=0.01)
        print("\nModel Summary:")
        model1.summary()
        
        history1 = self.train_model(model1, "Single Layer NN (LR=0.01)", epochs=100)
        result1 = self.evaluate_model_detailed(
            model1, 
            "Single Layer NN (Config 1)",
            "Single Layer NN (input -> FC -> sigmoid)",
            0.01,
            100
        )
        
        cv_score1 = self.run_cross_validation(
            self.create_single_layer_nn, "Single Layer NN (LR=0.01)", 
            n_splits=5, learning_rate=0.01
        )
        result1['Cross-Validation'] = cv_score1
        all_results.append(result1)
        
        # 2. Single Layer NN - Configuration 2: LR=0.001, Epochs=100
        print("\n" + "="*60)
        print("EXPERIMENT 2: Single Layer NN (LR=0.001, Epochs=100)")
        print("="*60)
        
        model2 = self.create_single_layer_nn(learning_rate=0.001)
        history2 = self.train_model(model2, "Single Layer NN (LR=0.001)", epochs=100)
        result2 = self.evaluate_model_detailed(
            model2,
            "Single Layer NN (Config 2)",
            "Single Layer NN (input -> FC -> sigmoid)",
            0.001,
            100
        )
        
        cv_score2 = self.run_cross_validation(
            self.create_single_layer_nn, "Single Layer NN (LR=0.001)",
            n_splits=5, learning_rate=0.001
        )
        result2['Cross-Validation'] = cv_score2
        all_results.append(result2)
        
        # 3. Single Layer NN - Configuration 3: LR=0.001, Epochs=200
        print("\n" + "="*60)
        print("EXPERIMENT 3: Single Layer NN (LR=0.001, Epochs=200)")
        print("="*60)
        
        model3 = self.create_single_layer_nn(learning_rate=0.001)
        history3 = self.train_model(model3, "Single Layer NN (LR=0.001, E=200)", epochs=200)
        result3 = self.evaluate_model_detailed(
            model3,
            "Single Layer NN (Config 3)",
            "Single Layer NN (input -> FC -> sigmoid)",
            0.001,
            200
        )
        
        # Use same CV as config 2 (same model, just more epochs)
        result3['Cross-Validation'] = cv_score2
        all_results.append(result3)
        
        # 4. 3 Layer NN Model
        print("\n" + "="*60)
        print("EXPERIMENT 4: 3 Layer NN Model")
        print("="*60)
        
        model4 = self.create_three_layer_nn(learning_rate=0.01)
        print("\nModel Summary:")
        model4.summary()
        
        history4 = self.train_model(model4, "3 Layer NN", epochs=100)
        result4 = self.evaluate_model_detailed(
            model4,
            "3 Layer NN Model",
            "3 Layer NN Model (input -> Linear(128) -> Linear(64) -> Linear(1) -> Sigmoid)",
            0.01,
            100
        )
        
        cv_score4 = self.run_cross_validation(
            self.create_three_layer_nn, "3 Layer NN",
            n_splits=5, learning_rate=0.01
        )
        result4['Cross-Validation'] = cv_score4
        all_results.append(result4)
        
        # Create results DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Clear session
        tf.keras.backend.clear_session()
        
        return results_df
    
    def plot_results_comparison(self, results_df):
        """Create visualizations comparing all models"""
        
        # 1. Metrics comparison
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        metrics = ['Precision', 'Recall', 'Specificity', 'NPV', 'AUC-ROC', 'Accuracy']
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
        
        for idx, metric in enumerate(metrics):
            values = results_df[metric].values
            model_labels = [f"Single NN\n(LR=0.01)", 
                           f"Single NN\n(LR=0.001)",
                           f"Single NN\n(LR=0.001, E=200)",
                           f"3 Layer NN"]
            
            bars = axes[idx].bar(range(len(values)), values, color=colors)
            axes[idx].set_title(metric, fontsize=14, fontweight='bold')
            axes[idx].set_ylim(0, 1.05)
            axes[idx].set_ylabel('Score')
            axes[idx].set_xticks(range(len(values)))
            axes[idx].set_xticklabels(model_labels, rotation=0, ha='center')
            
            # Add value labels
            for bar, v in zip(bars, values):
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                             f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('simple_nn_metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Model performance summary table
        print("\n" + "="*80)
        print("MODEL PERFORMANCE SUMMARY TABLE")
        print("="*80)
        
        # Create a cleaner summary for display
        summary_df = results_df.copy()
        summary_df['Model'] = ['Single Layer NN (LR=0.01, E=100)',
                               'Single Layer NN (LR=0.001, E=100)', 
                               'Single Layer NN (LR=0.001, E=200)',
                               '3 Layer NN (LR=0.01, E=100)']
        
        display_cols = ['Model', 'Precision', 'Recall', 'Specificity', 
                       'AUC-ROC', 'Cross-Validation']
        
        for col in display_cols[1:]:
            if col in summary_df.columns:
                summary_df[col] = summary_df[col].apply(lambda x: f'{x:.4f}' if pd.notna(x) else 'N/A')
        
        print(summary_df[display_cols].to_string(index=False))
        
        # 3. Confusion Matrix Visualization
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        for idx, row in results_df.iterrows():
            cm = np.array([[row['TN'], row['FP']], 
                          [row['FN'], row['TP']]])
            
            model_short_names = ["Single NN (LR=0.01)",
                                "Single NN (LR=0.001)", 
                                "Single NN (LR=0.001, E=200)",
                                "3 Layer NN"]
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       cbar=False, ax=axes[idx],
                       xticklabels=['Predicted 0', 'Predicted 1'],
                       yticklabels=['Actual 0', 'Actual 1'])
            axes[idx].set_title(f"{model_short_names[idx]}\nConfusion Matrix")
        
        plt.tight_layout()
        plt.savefig('simple_nn_confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return summary_df

def main():
    """Main function to run all experiments"""
    
    # Check GPU availability
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print(f"GPU available: {physical_devices[0].name}")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    else:
        print("No GPU found, using CPU")
    
    # Initialize the evaluator
    evaluator = SimpleOUDNeuralNetworks(
        data_path='/sharefolder/wanglab/ML_training/final_OUD_ML_dataset.csv'
    )
    
    # Load and prepare data
    if not evaluator.load_and_prepare_data():
        print("Failed to load data!")
        return
    
    # Run all experiments
    results_df = evaluator.run_all_experiments()
    
    # Save results in CSV format matching the original
    results_df.to_csv('simple_nn_results_full.csv', index=False)
    print("\nFull results saved to 'simple_nn_results_full.csv'")
    
    # Create visualizations
    summary_df = evaluator.plot_results_comparison(results_df)
    
    # Save summary
    summary_df.to_csv('simple_nn_summary.csv', index=False)
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*80)
    print("\nOutput files:")
    print("  - simple_nn_results_full.csv (full results matching CSV format)")
    print("  - simple_nn_summary.csv (summary table)")
    print("  - simple_nn_metrics_comparison.png")
    print("  - simple_nn_confusion_matrices.png")
    print("  - best_*.h5 (saved model weights)")

if __name__ == "__main__":
    main()