"""
OUD (Opioid Use Disorder) Prediction Models
============================================
This script implements and evaluates three ML models for OUD prediction:
1. Logistic Regression - Interpretable baseline with feature coefficients
2. Random Forest - Captures non-linear patterns and feature interactions
3. XGBoost - State-of-the-art gradient boosting for imbalanced data

The models are specifically chosen for healthcare prediction with class imbalance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, average_precision_score,
    f1_score, make_scorer
)
from sklearn.utils import class_weight
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

class OUDModelEvaluator:
    """
    Comprehensive evaluator for OUD prediction models with focus on 
    handling class imbalance and healthcare-specific metrics
    """
    
    def __init__(self, data_path='/sharefolder/wanglab/MME/final_dataset_with_oud_labels.csv'):
        self.data_path = data_path
        self.feature_cols = [
            'MME_last_365_days', 'MME_last_2_years', 'MME_prior_1_year',
            'MME_120_2_years', 'prscbr_last_2_years', 'prscrbr_last_180_days'
        ]
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        
    def load_and_prepare_data(self):
        """Load data and prepare for modeling"""
        print("Loading dataset...")
        df = pd.read_csv(self.data_path)
        print(f"Dataset shape: {df.shape}")
        
        # Extract features and target
        self.X = df[self.feature_cols]
        self.y = df['oud_label']
        
        # Check class distribution
        class_dist = self.y.value_counts()
        print(f"\nClass distribution:")
        print(f"Non-OUD (0): {class_dist[0]} ({class_dist[0]/len(self.y)*100:.2f}%)")
        print(f"OUD (1): {class_dist[1]} ({class_dist[1]/len(self.y)*100:.2f}%)")
        print(f"Class imbalance ratio: 1:{class_dist[0]/class_dist[1]:.1f}")
        
        # Handle missing values if any
        if self.X.isnull().sum().sum() > 0:
            print(f"\nHandling {self.X.isnull().sum().sum()} missing values...")
            self.X = self.X.fillna(self.X.median())
        
        # Split data stratified to maintain class distribution
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        print(f"\nTrain set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        
        return True
    
    def build_models(self):
        """Build three ML models suitable for OUD prediction"""
        
        # Calculate class weights for imbalanced data
        class_weights = class_weight.compute_class_weight(
            'balanced', classes=np.unique(self.y_train), y=self.y_train
        )
        class_weight_dict = dict(zip(np.unique(self.y_train), class_weights))
        print(f"\nClass weights: {class_weight_dict}")
        
        # 1. Logistic Regression - Interpretable baseline
        print("\n1. Building Logistic Regression...")
        self.models['Logistic Regression'] = {
            'model': LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                random_state=42,
                solver='liblinear',
                C=1.0
            ),
            'scaler': StandardScaler()
        }
        
        # 2. Random Forest - Handles non-linearity and interactions
        print("2. Building Random Forest...")
        self.models['Random Forest'] = {
            'model': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'scaler': None  # Tree-based models don't need scaling
        }
        
        # 3. XGBoost - State-of-the-art for imbalanced data
        print("3. Building XGBoost...")
        scale_pos_weight = class_weight_dict[0] / class_weight_dict[1]
        self.models['XGBoost'] = {
            'model': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            ),
            'scaler': None
        }
        
        return True
    
    def train_and_evaluate_models(self):
        """Train models and evaluate with healthcare-focused metrics"""
        
        # Define stratified k-fold for cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model_info in self.models.items():
            print(f"\n{'='*60}")
            print(f"Training {name}")
            print('='*60)
            
            model = model_info['model']
            scaler = model_info['scaler']
            
            # Prepare data
            if scaler is not None:
                X_train_scaled = scaler.fit_transform(self.X_train)
                X_test_scaled = scaler.transform(self.X_test)
            else:
                X_train_scaled = self.X_train
                X_test_scaled = self.X_test
            
            # Train model
            model.fit(X_train_scaled, self.y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Cross-validation scores
            cv_scores = cross_val_score(
                model, X_train_scaled, self.y_train, 
                cv=skf, scoring='roc_auc', n_jobs=-1
            )
            
            # Calculate comprehensive metrics
            tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
            
            metrics = {
                'accuracy': (tp + tn) / (tp + tn + fp + fn),
                'sensitivity': tp / (tp + fn),  # Recall for OUD class
                'specificity': tn / (tn + fp),
                'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'f1_score': f1_score(self.y_test, y_pred),
                'auc_roc': roc_auc_score(self.y_test, y_pred_proba),
                'auc_pr': average_precision_score(self.y_test, y_pred_proba),
                'cv_auc_mean': cv_scores.mean(),
                'cv_auc_std': cv_scores.std(),
                'confusion_matrix': confusion_matrix(self.y_test, y_pred),
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            self.results[name] = metrics
            
            # Print results
            print(f"\nPerformance Metrics:")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Sensitivity (Recall): {metrics['sensitivity']:.4f}")
            print(f"Specificity: {metrics['specificity']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"F1-Score: {metrics['f1_score']:.4f}")
            print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
            print(f"AUC-PR: {metrics['auc_pr']:.4f}")
            print(f"CV AUC: {metrics['cv_auc_mean']:.4f} (±{metrics['cv_auc_std']:.4f})")
            
            print(f"\nConfusion Matrix:")
            print(metrics['confusion_matrix'])
            print(f"True Negatives: {tn}")
            print(f"False Positives: {fp}")
            print(f"False Negatives: {fn}")
            print(f"True Positives: {tp}")
            
            # Feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_imp = pd.DataFrame({
                    'feature': self.feature_cols,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                print(f"\nFeature Importances:")
                for _, row in feature_imp.iterrows():
                    print(f"  {row['feature']}: {row['importance']:.4f}")
            
            # Coefficients for logistic regression
            elif hasattr(model, 'coef_'):
                coefficients = model.coef_[0]
                feature_coef = pd.DataFrame({
                    'feature': self.feature_cols,
                    'coefficient': coefficients
                }).sort_values('coefficient', key=abs, ascending=False)
                print(f"\nFeature Coefficients:")
                for _, row in feature_coef.iterrows():
                    print(f"  {row['feature']}: {row['coefficient']:.4f}")
        
        return True
    
    def plot_model_comparison(self):
        """Create comprehensive visualizations for model comparison"""
        
        # 1. Metrics Comparison Bar Plot
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
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        
        for idx, (metric, title) in enumerate(metrics_to_plot):
            values = [self.results[model][metric] for model in model_names]
            axes[idx].bar(model_names, values, color=colors)
            axes[idx].set_title(title, fontsize=14, fontweight='bold')
            axes[idx].set_ylim(0, 1.05)
            axes[idx].set_ylabel('Score')
            
            # Add value labels
            for i, v in enumerate(values):
                axes[idx].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
            
            # Rotate x-labels
            axes[idx].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('oud_model_metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. ROC Curves
        plt.figure(figsize=(10, 8))
        for name, color in zip(model_names, colors):
            fpr, tpr, _ = roc_curve(self.y_test, self.results[name]['y_pred_proba'])
            auc = self.results[name]['auc_roc']
            plt.plot(fpr, tpr, color=color, lw=2, 
                    label=f'{name} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - OUD Prediction Models', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.savefig('oud_roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Precision-Recall Curves (Important for imbalanced data)
        plt.figure(figsize=(10, 8))
        for name, color in zip(model_names, colors):
            precision, recall, _ = precision_recall_curve(
                self.y_test, self.results[name]['y_pred_proba']
            )
            auc_pr = self.results[name]['auc_pr']
            plt.plot(recall, precision, color=color, lw=2,
                    label=f'{name} (AP = {auc_pr:.3f})')
        
        # Add baseline (random classifier)
        baseline = self.y_test.sum() / len(self.y_test)
        plt.axhline(y=baseline, color='k', linestyle='--', lw=1,
                   label=f'Baseline (AP = {baseline:.3f})')
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves - OUD Prediction Models', fontsize=14, fontweight='bold')
        plt.legend(loc='lower left')
        plt.grid(True, alpha=0.3)
        plt.savefig('oud_pr_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 4. Confusion Matrices
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, (name, color) in enumerate(zip(model_names, colors)):
            cm = self.results[name]['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       cbar=False, ax=axes[idx],
                       xticklabels=['Non-OUD', 'OUD'],
                       yticklabels=['Non-OUD', 'OUD'])
            axes[idx].set_title(f'{name}\nConfusion Matrix', fontsize=14)
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig('oud_confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return True
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\n" + "="*80)
        print("SUMMARY REPORT: OUD PREDICTION MODEL EVALUATION")
        print("="*80)
        
        # Create summary DataFrame
        summary_data = []
        for model_name, metrics in self.results.items():
            summary_data.append({
                'Model': model_name,
                'AUC-ROC': f"{metrics['auc_roc']:.4f}",
                'AUC-PR': f"{metrics['auc_pr']:.4f}",
                'Sensitivity': f"{metrics['sensitivity']:.4f}",
                'Specificity': f"{metrics['specificity']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}",
                'CV AUC': f"{metrics['cv_auc_mean']:.4f} (±{metrics['cv_auc_std']:.3f})"
            })
        
        summary_df = pd.DataFrame(summary_data)
        print("\nModel Performance Summary:")
        print(summary_df.to_string(index=False))
        
        # Healthcare-specific recommendations
        print("\n" + "-"*80)
        print("HEALTHCARE-SPECIFIC INSIGHTS:")
        print("-"*80)
        
        # Find best model for different criteria
        best_sensitivity = max(self.results.items(), key=lambda x: x[1]['sensitivity'])
        best_precision = max(self.results.items(), key=lambda x: x[1]['precision'])
        best_f1 = max(self.results.items(), key=lambda x: x[1]['f1_score'])
        
        print(f"\n Clinical Recommendations:")
        print(f"\n1. For SCREENING (maximize sensitivity to catch all potential OUD cases):")
        print(f"   → Use {best_sensitivity[0]}")
        print(f"   → Sensitivity: {best_sensitivity[1]['sensitivity']:.4f}")
        print(f"   → This will identify {best_sensitivity[1]['sensitivity']*100:.1f}% of OUD cases")
        
        print(f"\n2. For TARGETED INTERVENTION (balance precision and recall):")
        print(f"   → Use {best_f1[0]}")
        print(f"   → F1-Score: {best_f1[1]['f1_score']:.4f}")
        print(f"   → Precision: {best_f1[1]['precision']:.4f}")
        
        print(f"\n3. For RESOURCE-LIMITED SETTINGS (maximize precision):")
        print(f"   → Use {best_precision[0]}")
        print(f"   → Precision: {best_precision[1]['precision']:.4f}")
        print(f"   → {best_precision[1]['precision']*100:.1f}% of positive predictions are correct")
        
        # Calculate potential impact
        print("\n" + "-"*80)
        print("POTENTIAL CLINICAL IMPACT:")
        print("-"*80)
        
        for name, metrics in self.results.items():
            tn, fp, fn, tp = metrics['confusion_matrix'].ravel()
            total = tn + fp + fn + tp
            
            print(f"\n{name}:")
            print(f"  • Correctly identified OUD patients: {tp} ({tp/total*100:.2f}%)")
            print(f"  • Missed OUD patients: {fn} ({fn/total*100:.2f}%)")
            print(f"  • False alarms: {fp} ({fp/total*100:.2f}%)")
            print(f"  • Number needed to screen: {(tp+fp)/tp:.1f}")
        
        # Save summary
        summary_df.to_csv('oud_model_summary.csv', index=False)
        print("\n✓ Summary saved to 'oud_model_summary.csv'")
        
        return summary_df
    
    def run_complete_evaluation(self):
        """Run the complete model evaluation pipeline"""
        print("Starting OUD Prediction Model Evaluation")
        print("="*80)
        
        # Load and prepare data
        if not self.load_and_prepare_data():
            return None
        
        # Build models
        if not self.build_models():
            return None
        
        # Train and evaluate
        if not self.train_and_evaluate_models():
            return None
        
        # Visualize results
        self.plot_model_comparison()
        
        # Generate summary
        summary = self.generate_summary_report()
        
        print("\n Evaluation completed successfully!")
        
        return {
            'results': self.results,
            'summary': summary
        }

def main():
    """Main function to run OUD prediction model evaluation"""
    
    # Initialize evaluator
    evaluator = OUDModelEvaluator()
    
    # Run evaluation
    results = evaluator.run_complete_evaluation()
    
    if results:
        print("\n" + "="*80)
        print("EVALUATION COMPLETE")
        print("="*80)
        print("All models trained and evaluated")
        print("Visualizations saved")
        print("Summary report generated")
        print("\n Output files:")
        print("  - oud_model_metrics_comparison.png")
        print("  - oud_roc_curves.png")
        print("  - oud_pr_curves.png")
        print("  - oud_confusion_matrices.png")
        print("  - oud_model_summary.csv")

if __name__ == "__main__":
    main()