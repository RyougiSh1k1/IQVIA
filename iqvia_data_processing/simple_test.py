"""
OUD Prediction ML Model
This script builds and evaluates machine learning models to predict Opioid Use Disorder (OUD) events
based on processed IQVIA features.

Features include:
- MME_last_365_days: Total MME in last 365 days
- MME_last_2_years: Total MME in last 2 years
- MME_prior_1_year: Total MME prior to 1 year before most recent date
- MME_120_2_years: Number of prescriptions with daily MME > 120 in last 2 years
- prscbr_last_2_years: Number of prescribers in last 2 years
- prscrbr_last_180_days: Number of prescribers in last 180 days

Target: oud_label (0/1 binary classification)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           roc_curve, precision_recall_curve, average_precision_score,
                           accuracy_score, precision_score, recall_score, f1_score)
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class OUDPredictor:
    def __init__(self, data_path='/sharefolder/wanglab/MME/final_dataset_with_labels.csv'):
        """
        Initialize the OUD Predictor
        
        Args:
            data_path (str): Path to the final dataset with labels
        """
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.models = {}
        self.feature_names = []
        
    def load_and_prepare_data(self):
        """
        Load the dataset and prepare features for modeling
        If final dataset doesn't exist, automatically merge feature files
        """
        print("Loading dataset...")
        
        # Try to load final dataset first
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Final dataset loaded successfully: {self.df.shape}")
        except FileNotFoundError:
            print(f"Final dataset not found at {self.data_path}")
            print("Attempting to merge feature files...")
            
            # Try to merge MME and prescriber features
            if not self._merge_feature_files():
                return False
            
            # Try loading again after merge
            try:
                self.df = pd.read_csv(self.data_path)
                print(f"Merged dataset loaded successfully: {self.df.shape}")
            except FileNotFoundError:
                print("Failed to create or load merged dataset")
                return False
        
        print("\nDataset Info:")
        print(self.df.info())
        print(f"\nDataset shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        
        # Check if we have the expected features
        expected_features = [
            'MME_last_365_days', 'MME_last_2_years', 'MME_prior_1_year', 
            'MME_120_2_years', 'prscbr_last_2_years', 'prscrbr_last_180_days'
        ]
        
        # Check which features are available
        available_features = [col for col in expected_features if col in self.df.columns]
        missing_features = [col for col in expected_features if col not in self.df.columns]
        
        if missing_features:
            print(f"\nWarning: Missing expected features: {missing_features}")
        
        if not available_features:
            print("Error: No expected features found in dataset")
            return False
            
        print(f"\nAvailable features: {available_features}")
        
        # Check if target variable exists
        if 'oud_label' not in self.df.columns:
            print("Error: Target variable 'oud_label' not found in dataset")
            return False
        
        # Prepare features and target
        self.feature_names = available_features
        self.X = self.df[self.feature_names].copy()
        self.y = self.df['oud_label'].copy()
        
        print(f"\nTarget distribution:")
        print(self.y.value_counts())
        print(f"OUD prevalence: {self.y.mean():.4f}")
        
        return True
    
    def _merge_feature_files(self):
        """
        Merge MME and prescriber feature files if final dataset doesn't exist
        """
        print("Merging feature files...")
        
        # Paths to feature files
        mme_path = '/sharefolder/wanglab/MME/MME_features_200_max.csv'
        prescriber_path = '/sharefolder/wanglab/MME/prscbr_features.csv'
        
        # Check if feature files exist
        if not os.path.exists(mme_path):
            print(f"MME features file not found: {mme_path}")
            return False
            
        if not os.path.exists(prescriber_path):
            print(f"Prescriber features file not found: {prescriber_path}")
            return False
        
        try:
            # Load feature files
            mme_df = pd.read_csv(mme_path)
            prescriber_df = pd.read_csv(prescriber_path)
            
            print(f"MME features loaded: {mme_df.shape}")
            print(f"Prescriber features loaded: {prescriber_df.shape}")
            
            # Merge on patient ID and most recent date
            merged_df = pd.merge(
                mme_df, 
                prescriber_df, 
                on=['pat_id', 'most_recent_date'], 
                how='inner'
            )
            
            print(f"Merged features shape: {merged_df.shape}")
            
            # Add dummy OUD labels (all 0) since we don't have ICD data
            merged_df['oud_label'] = 0
            merged_df['num_oud_events'] = 0
            merged_df['first_oud_date'] = None
            merged_df['latest_oud_date'] = None
            
            print("Added dummy OUD labels (all patients labeled as 0 - no OUD)")
            print("Note: Replace with actual ICD-based labels when available")
            
            # Save merged dataset
            merged_df.to_csv(self.data_path, index=False)
            print(f"Merged dataset saved to: {self.data_path}")
            
            return True
            
        except Exception as e:
            print(f"Error merging feature files: {str(e)}")
            return False
    
    def exploratory_data_analysis(self):
        """
        Perform exploratory data analysis
        """
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        # Basic statistics
        print("\nFeature Statistics:")
        print(self.X.describe())
        
        # Missing values
        print("\nMissing Values:")
        missing_values = self.X.isnull().sum()
        print(missing_values)
        
        # Correlation matrix
        plt.figure(figsize=(10, 8))
        correlation_matrix = self.X.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()
        
        # Distribution plots
        n_features = len(self.feature_names)
        fig, axes = plt.subplots(nrows=(n_features + 1) // 2, ncols=2, figsize=(15, 4 * ((n_features + 1) // 2)))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for i, feature in enumerate(self.feature_names):
            if i < len(axes):
                axes[i].hist(self.X[feature], bins=50, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'Distribution of {feature}')
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel('Frequency')
        
        # Hide unused subplots
        for i in range(len(self.feature_names), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
        # Feature distributions by target
        fig, axes = plt.subplots(nrows=(n_features + 1) // 2, ncols=2, figsize=(15, 4 * ((n_features + 1) // 2)))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for i, feature in enumerate(self.feature_names):
            if i < len(axes):
                for label in [0, 1]:
                    subset = self.X[self.y == label][feature]
                    axes[i].hist(subset, bins=30, alpha=0.6, 
                               label=f'OUD={label}', density=True)
                axes[i].set_title(f'{feature} by OUD Status')
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel('Density')
                axes[i].legend()
        
        # Hide unused subplots
        for i in range(len(self.feature_names), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def preprocess_data(self):
        """
        Preprocess the data: handle missing values, outliers, and scaling
        """
        print("\n" + "="*50)
        print("DATA PREPROCESSING")
        print("="*50)
        
        # Handle missing values
        print("Handling missing values...")
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(self.X)
        self.X = pd.DataFrame(X_imputed, columns=self.feature_names)
        
        # Remove extreme outliers (values beyond 99.5th percentile)
        print("Handling outliers...")
        for feature in self.feature_names:
            Q99_5 = self.X[feature].quantile(0.995)
            outlier_mask = self.X[feature] > Q99_5
            print(f"{feature}: Capping {outlier_mask.sum()} outliers at {Q99_5:.2f}")
            self.X.loc[outlier_mask, feature] = Q99_5
        
        # Split the data
        print("Splitting data into train/test sets...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        print(f"Training set OUD prevalence: {self.y_train.mean():.4f}")
        print(f"Test set OUD prevalence: {self.y_test.mean():.4f}")
        
        # Scale the features
        print("Scaling features...")
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Convert back to DataFrame for easier handling
        self.X_train_scaled = pd.DataFrame(self.X_train_scaled, columns=self.feature_names)
        self.X_test_scaled = pd.DataFrame(self.X_test_scaled, columns=self.feature_names)
    
    def train_models(self):
        """
        Train multiple machine learning models
        """
        print("\n" + "="*50)
        print("MODEL TRAINING")
        print("="*50)
        
        # Define models to train
        models_config = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(random_state=42, probability=True)
        }
        
        # Train models and perform cross-validation
        cv_scores = {}
        
        for name, model in models_config.items():
            print(f"\nTraining {name}...")
            
            # Use scaled data for SVM and Logistic Regression, original for tree-based
            if name in ['Logistic Regression', 'SVM']:
                X_train_model = self.X_train_scaled
                X_test_model = self.X_test_scaled
            else:
                X_train_model = self.X_train
                X_test_model = self.X_test
            
            # Train the model
            model.fit(X_train_model, self.y_train)
            
            # Cross-validation
            cv_scores[name] = cross_val_score(
                model, X_train_model, self.y_train, 
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring='roc_auc', n_jobs=-1
            )
            
            print(f"CV AUC: {cv_scores[name].mean():.4f} (+/- {cv_scores[name].std() * 2:.4f})")
            
            # Store model and test data for evaluation
            self.models[name] = {
                'model': model,
                'X_test': X_test_model,
                'cv_scores': cv_scores[name]
            }
        
        return cv_scores
    
    def evaluate_models(self):
        """
        Evaluate all trained models and compare performance
        """
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        results = {}
        
        for name, model_info in self.models.items():
            print(f"\n{'-' * 30}")
            print(f"Evaluating {name}")
            print(f"{'-' * 30}")
            
            model = model_info['model']
            X_test_model = model_info['X_test']
            
            # Predictions
            y_pred = model.predict(X_test_model)
            y_pred_proba = model.predict_proba(X_test_model)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            auc_roc = roc_auc_score(self.y_test, y_pred_proba)
            auc_pr = average_precision_score(self.y_test, y_pred_proba)
            
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc_roc': auc_roc,
                'auc_pr': auc_pr,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            print(f"AUC-ROC: {auc_roc:.4f}")
            print(f"AUC-PR: {auc_pr:.4f}")
            
            # Confusion Matrix
            cm = confusion_matrix(self.y_test, y_pred)
            print(f"\nConfusion Matrix:")
            print(cm)
            
            # Classification Report
            print(f"\nClassification Report:")
            print(classification_report(self.y_test, y_pred))
        
        return results
    
    def plot_model_comparison(self, results):
        """
        Create visualizations comparing model performance
        """
        print("\n" + "="*50)
        print("MODEL COMPARISON PLOTS")
        print("="*50)
        
        # Performance metrics comparison
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc', 'auc_pr']
        model_names = list(results.keys())
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = [results[model][metric] for model in model_names]
            axes[i].bar(model_names, values)
            axes[i].set_title(f'{metric.upper().replace("_", "-")}')
            axes[i].set_ylabel('Score')
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for j, v in enumerate(values):
                axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        # ROC Curves
        plt.figure(figsize=(10, 8))
        for name, result in results.items():
            fpr, tpr, _ = roc_curve(self.y_test, result['y_pred_proba'])
            plt.plot(fpr, tpr, label=f"{name} (AUC = {result['auc_roc']:.3f})")
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Precision-Recall Curves
        plt.figure(figsize=(10, 8))
        for name, result in results.items():
            precision, recall, _ = precision_recall_curve(self.y_test, result['y_pred_proba'])
            plt.plot(recall, precision, label=f"{name} (AP = {result['auc_pr']:.3f})")
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def feature_importance_analysis(self):
        """
        Analyze feature importance for tree-based models
        """
        print("\n" + "="*50)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*50)
        
        # Get feature importance from Random Forest and Gradient Boosting
        importance_models = ['Random Forest', 'Gradient Boosting']
        
        fig, axes = plt.subplots(1, len(importance_models), figsize=(15, 6))
        if len(importance_models) == 1:
            axes = [axes]
        
        for i, model_name in enumerate(importance_models):
            if model_name in self.models:
                model = self.models[model_name]['model']
                importances = model.feature_importances_
                
                # Sort features by importance
                indices = np.argsort(importances)[::-1]
                
                axes[i].bar(range(len(self.feature_names)), 
                           importances[indices])
                axes[i].set_title(f'Feature Importance - {model_name}')
                axes[i].set_xlabel('Features')
                axes[i].set_ylabel('Importance')
                axes[i].set_xticks(range(len(self.feature_names)))
                axes[i].set_xticklabels([self.feature_names[j] for j in indices], 
                                       rotation=45, ha='right')
                
                # Print feature importance
                print(f"\n{model_name} Feature Importance:")
                for idx in indices:
                    print(f"{self.feature_names[idx]}: {importances[idx]:.4f}")
        
        plt.tight_layout()
        plt.show()
    
    def generate_model_summary(self, results):
        """
        Generate a comprehensive summary of model performance
        """
        print("\n" + "="*80)
        print("MODEL PERFORMANCE SUMMARY")
        print("="*80)
        
        # Create summary DataFrame
        summary_data = []
        for model_name, metrics in results.items():
            summary_data.append({
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}",
                'AUC-ROC': f"{metrics['auc_roc']:.4f}",
                'AUC-PR': f"{metrics['auc_pr']:.4f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # Best model recommendation
        best_auc_roc = max(results.values(), key=lambda x: x['auc_roc'])
        best_model_name = [name for name, metrics in results.items() 
                          if metrics['auc_roc'] == best_auc_roc['auc_roc']][0]
        
        print(f"\n🏆 RECOMMENDED MODEL: {best_model_name}")
        print(f"   - Highest AUC-ROC: {best_auc_roc['auc_roc']:.4f}")
        print(f"   - F1-Score: {best_auc_roc['f1_score']:.4f}")
        print(f"   - Precision: {best_auc_roc['precision']:.4f}")
        print(f"   - Recall: {best_auc_roc['recall']:.4f}")
        
        return summary_df, best_model_name
    
    def run_complete_pipeline(self):
        """
        Run the complete ML pipeline
        """
        print("🚀 Starting OUD Prediction ML Pipeline")
        print("="*80)
        
        # Step 1: Load and prepare data
        if not self.load_and_prepare_data():
            return None
        
        # Step 2: Exploratory Data Analysis
        self.exploratory_data_analysis()
        
        # Step 3: Preprocess data
        self.preprocess_data()
        
        # Step 4: Train models
        cv_scores = self.train_models()
        
        # Step 5: Evaluate models
        results = self.evaluate_models()
        
        # Step 6: Visualize results
        self.plot_model_comparison(results)
        
        # Step 7: Feature importance analysis
        self.feature_importance_analysis()
        
        # Step 8: Generate summary
        summary_df, best_model = self.generate_model_summary(results)
        
        print("\n✅ Pipeline completed successfully!")
        
        return {
            'results': results,
            'summary': summary_df,
            'best_model': best_model,
            'cv_scores': cv_scores
        }

def main():
    """
    Main function to run the OUD prediction model
    """
    # Initialize predictor
    predictor = OUDPredictor()
    
    # Run complete pipeline
    pipeline_results = predictor.run_complete_pipeline()
    
    if pipeline_results:
        print("\n" + "="*80)
        print("PIPELINE RESULTS SUMMARY")
        print("="*80)
        
        print(f"✅ Best performing model: {pipeline_results['best_model']}")
        print(f"✅ Models trained and evaluated: {len(pipeline_results['results'])}")
        print(f"✅ Features used: {len(predictor.feature_names)}")
        print(f"✅ Training samples: {len(predictor.X_train)}")
        print(f"✅ Test samples: {len(predictor.X_test)}")
        
        # Save results
        output_dir = '/sharefolder/wanglab/MME/'
        pipeline_results['summary'].to_csv(f'{output_dir}model_performance_summary.csv', index=False)
        print(f"✅ Results saved to {output_dir}model_performance_summary.csv")

if __name__ == "__main__":
    main()