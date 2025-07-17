"""
OUD (Opioid Use Disorder) Prediction Models - Enhanced with Demographics & SDOH
===============================================================================
This script implements and evaluates three ML models for OUD prediction:
1. Logistic Regression - Interpretable baseline with feature coefficients
2. Random Forest - Captures non-linear patterns and feature interactions
3. XGBoost - State-of-the-art gradient boosting for imbalanced data

Now includes demographic features (age, gender, payment type) and
socioeconomic determinants of health (SDOH) from ZIP-level census data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
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
import time
import optuna
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import EditedNearestNeighbours

class OUDModelEvaluator:
    """
    Comprehensive evaluator for OUD prediction models with focus on 
    handling class imbalance and healthcare-specific metrics.
    Now includes demographic and SDOH features.
    """
    
    def __init__(self, data_path='/sharefolder/wanglab/ML_training/final_OUD_ML_dataset.csv',
    use_smote=False, use_rus=False, use_enn=False, use_knn = False, n_models = 1,
            model_selection=[
                'Logistic Regression', 'Random Forest', 'XGBoost'],
            corr_feat=False):
        self.data_path = data_path
        self.use_smote = use_smote
        self.use_rus = use_rus
        self.use_enn = use_enn
        self.use_knn = use_knn
        self.n_models = n_models
        self.corr_feat = corr_feat
        
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
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.models = {}
        self.results = {}
        self.encoders = {}
        self.model_selection = model_selection if model_selection else ['Logistic Regression', 'Random Forest', 'XGBoost']

    def add_knn_features(self, X, y, k=5):
        """Add KNN-derived features: neighbor class ratio and average distance."""
        from sklearn.neighbors import NearestNeighbors

        # Ensure numerical-only input
        numeric_X = X.select_dtypes(include=[np.number])

        nbrs = NearestNeighbors(n_neighbors=k+1).fit(numeric_X)
        distances, indices = nbrs.kneighbors(numeric_X)

        # Remove self-reference (first neighbor)
        distances = distances[:, 1:]
        indices = indices[:, 1:]

        # Neighbor label stats
        neighbor_labels = np.array([y.iloc[neighbor_ids].values for neighbor_ids in indices])
        knn_oud_ratio = neighbor_labels.sum(axis=1) / k
        knn_avg_distance = distances.mean(axis=1)

        # Add features
        X = X.copy()
        X['knn_oud_ratio'] = knn_oud_ratio
        X['knn_avg_distance'] = knn_avg_distance

        return X
    
    def load_and_prepare_data(self):
        """Load data and prepare for modeling with enhanced features"""
        print("Loading dataset...", flush = True)
        df = pd.read_csv(self.data_path)
        print(f"Dataset shape: {df.shape}", flush = True)
        print(f"Columns: {list(df.columns)}", flush = True)
        
        # Drop specified columns
        print(f"\nDropping columns: {self.columns_to_drop}", flush = True)
        df = df.drop(columns=[col for col in self.columns_to_drop if col in df.columns])
        
        # Handle categorical variables
        print("\nEncoding categorical variables...", flush = True)
        
        # Encode gender (assuming F=0, M=1)
        if 'der_sex' in df.columns:
            self.encoders['der_sex'] = LabelEncoder()
            df['der_sex'] = self.encoders['der_sex'].fit_transform(df['der_sex'])
            print(f"Gender encoding: {dict(zip(self.encoders['der_sex'].classes_, self.encoders['der_sex'].transform(self.encoders['der_sex'].classes_)))}", flush = True)
        
        # Encode payment type
        if 'pay_type' in df.columns:
            self.encoders['pay_type'] = LabelEncoder()
            df['pay_type'] = self.encoders['pay_type'].fit_transform(df['pay_type'])
            print(f"Payment type encoding: {dict(zip(self.encoders['pay_type'].classes_, self.encoders['pay_type'].transform(self.encoders['pay_type'].classes_)))}", flush = True)
        
        # Optionally remove highly correlated features for Logistic Regression
        if self.corr_feat and 'Logistic Regression' in self.model_selection:
            print("\nRemoving highly correlated features (corr > 0.9)...", flush=True)
            numeric_df = df.select_dtypes(include=[np.number]).drop(columns=[self.target_col])
            corr_matrix = numeric_df.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]

            if to_drop:
                print(f"Dropping {len(to_drop)} correlated features: {to_drop}", flush=True)
                df.drop(columns=to_drop, inplace=True)
            else:
                print("No highly correlated features found.", flush=True)

        # Separate features and target
        self.y = df[self.target_col]
        self.X = df.drop(columns=[self.target_col])
        self.feature_names = list(self.X.columns)
        
        # Check class distribution
        class_dist = self.y.value_counts()
        print(f"\nClass distribution:", flush = True)
        print(f"Non-OUD (0): {class_dist[0]} ({class_dist[0]/len(self.y)*100:.2f}%)", flush = True)
        print(f"OUD (1): {class_dist[1]} ({class_dist[1]/len(self.y)*100:.2f}%)", flush = True)
        print(f"Class imbalance ratio: 1:{class_dist[0]/class_dist[1]:.1f}", flush = True)
        
        # Handle missing values if any
        if self.X.isnull().sum().sum() > 0:
            print(f"\nHandling {self.X.isnull().sum().sum()} missing values...", flush = True)
            self.X = self.X.fillna(self.X.median())
        
        # Feature statistics by group
        self._print_feature_group_stats()
        
        # Conditionally add KNN features
        if self.use_knn and 'Logistic Regression' in self.model_selection:
            print("\nAdding KNN-based neighborhood features (for Logistic Regression only)...", flush=True)
            df = df.dropna()  # Ensure clean data
            self.y = df[self.target_col]
            self.X = df.drop(columns=[self.target_col])
            self.X = self.add_knn_features(self.X, self.y)
            print("Added features: 'knn_oud_ratio', 'knn_avg_distance'", flush=True)


        # Split data stratified to maintain class distribution
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        print(f"\nTrain set: {self.X_train.shape}", flush = True)
        print(f"Test set: {self.X_test.shape}", flush = True)
        
        return True
    
    def _print_feature_group_stats(self):
        """Print statistics for each feature group"""
        print("\n" + "="*60, flush = True)
        print("FEATURE GROUP STATISTICS", flush = True)
        print("="*60, flush = True)
        
        feature_groups = {
            'MME Features': self.mme_features,
            'Prescriber Features': self.prescriber_features,
            'Demographic Features': self.demographic_features,
            'SDOH Features': self.sdoh_features
        }
        
        for group_name, features in feature_groups.items():
            available_features = [f for f in features if f in self.feature_names]
            if available_features:
                print(f"\n{group_name} ({len(available_features)} features):", flush = True)
                group_data = self.X[available_features]
                print(group_data.describe().round(2), flush = True)
        
    def build_models(self, tune=True):
        """Build ensemble models with n_models using same hyperparameters"""
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(self.y_train), y=self.y_train)
        class_weight_dict = dict(zip(np.unique(self.y_train), class_weights))

        def clone_model(model_type, params, seed_offset=0):
                return [
                    LogisticRegression(
                        C=params['C'],
                        class_weight='balanced',
                        max_iter=1000,
                        solver='liblinear',
                        random_state=42 + i + seed_offset
                    ) if model_type == 'Logistic Regression' else
                    RandomForestClassifier(
                        n_estimators=params['n_estimators'],
                        max_depth=params['max_depth'],
                        min_samples_split=params['min_samples_split'],
                        min_samples_leaf=params['min_samples_leaf'],
                        class_weight='balanced',
                        random_state=42 + i + seed_offset,
                        n_jobs=-1
                    ) if model_type == 'Random Forest' else
                    xgb.XGBClassifier(
                        n_estimators=params['n_estimators'],
                        max_depth=params['max_depth'],
                        learning_rate=params['learning_rate'],
                        subsample=params['subsample'],
                        colsample_bytree=params['colsample_bytree'],
                        scale_pos_weight=class_weight_dict[0] / class_weight_dict[1] if not self.use_smote and not self.use_rus else 1,
                        random_state=42 + i + seed_offset,
                        eval_metric='logloss'
                    )
                    for i in range(self.n_models)
                ]

        if tune:
            lr_params = self._tune_model('Logistic Regression', self.X_train, self.y_train) \
                if 'Logistic Regression' in self.model_selection else {'C': 1.0}

            rf_params = self._tune_model('Random Forest', self.X_train, self.y_train) \
                if 'Random Forest' in self.model_selection else {
                    'n_estimators': 200, 'max_depth': 10,
                    'min_samples_split': 20, 'min_samples_leaf': 10
                }

            xgb_params = self._tune_model('XGBoost', self.X_train, self.y_train) \
                if 'XGBoost' in self.model_selection else {
                    'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.01,
                    'subsample': 0.8, 'colsample_bytree': 0.8
                }
        else:
            lr_params = {'C': 1.0}
            rf_params = {'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 20, 'min_samples_leaf': 10}
            xgb_params = {
                'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.01,
                'subsample': 0.8, 'colsample_bytree': 0.8
            }

        if 'Logistic Regression' in self.model_selection:
            self.models['Logistic Regression'] = {
                'model': clone_model('Logistic Regression', lr_params),
                'scaler': StandardScaler()
            }

        if 'Random Forest' in self.model_selection:
            self.models['Random Forest'] = {
                'model': clone_model('Random Forest', rf_params),
                'scaler': None
            }

        if 'XGBoost' in self.model_selection:
            self.models['XGBoost'] = {
                'model': clone_model('XGBoost', xgb_params),
                'scaler': None
            }

        return True


    def _tune_model(self, model_name, X, y):
        """Tune hyperparameters using Optuna"""
        print(f"\nTuning {model_name} with Optuna...", flush = True)

        def objective(trial):
            if model_name == 'Logistic Regression':
                C = trial.suggest_loguniform('C', 1e-4, 10)
                model = LogisticRegression(
                    C=C,
                    max_iter=1000,
                    class_weight='balanced',
                    solver='liblinear',
                    random_state=42
                )
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                return cross_val_score(model, X_scaled, y, cv=5, scoring='roc_auc').mean()

            elif model_name == 'Random Forest':
                model = RandomForestClassifier(
                    n_estimators=trial.suggest_int('n_estimators', 100, 500),
                    max_depth=trial.suggest_int('max_depth', 5, 20),
                    min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
                    min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 20),
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1
                )
                return cross_val_score(model, X, y, cv=5, scoring='roc_auc').mean()

            elif model_name == 'XGBoost':
                model = xgb.XGBClassifier(
                    n_estimators=trial.suggest_int('n_estimators', 100, 300),
                    max_depth=trial.suggest_int('max_depth', 3, 10),
                    learning_rate=trial.suggest_float('learning_rate', 0.001, 0.1),
                    subsample=trial.suggest_float('subsample', 0.5, 1.0),
                    colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    scale_pos_weight=self.y_train.value_counts()[0] / self.y_train.value_counts()[1],
                    random_state=42,
                    eval_metric='logloss'
                )
                return cross_val_score(model, X, y, cv=5, scoring='roc_auc').mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=25, timeout=600)

        print(f"Best hyperparameters for {model_name}: {study.best_params}", flush = True)
        return study.best_params

    def train_and_evaluate_models(self):
        """Train models and evaluate with healthcare-focused metrics"""
        
        # Define stratified k-fold for cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model_info in self.models.items():
            print(f"\n{'='*60}", flush = True)
            print(f"Training {name} with {self.n_models} model(s) (majority voting)")
            print('='*60, flush = True)
            
            start_time = time.time()

            model_list = model_info['model']
            scaler = model_info['scaler']
            
            # Prepare data
            if scaler is not None:
                X_train_scaled = scaler.fit_transform(self.X_train)
                X_test_scaled = scaler.transform(self.X_test)
            else:
                X_train_scaled = self.X_train
                X_test_scaled = self.X_test
            
            # Resampling logic
            print("Resampling strategy:", flush=True)
            steps = []
            if self.use_smote:
                print("  • SMOTE enabled", flush=True)
                steps.append(('smote', SMOTE(random_state=42)))
            if self.use_enn:
                print("  • ENN enabled", flush=True)
                steps.append(('enn', EditedNearestNeighbours()))
            if self.use_rus:
                print("  • RUS enabled", flush=True)
                steps.append(('rus', RandomUnderSampler(random_state=42)))

            if steps:
                from imblearn.pipeline import Pipeline as ImbPipeline
                sampling_pipeline = ImbPipeline(steps)
                X_resampled, y_resampled = sampling_pipeline.fit_resample(X_train_scaled, self.y_train)
                print(f"  → Resampled from {X_train_scaled.shape[0]} to {X_resampled.shape[0]} samples", flush=True)
            else:
                X_resampled, y_resampled = X_train_scaled, self.y_train

            # Train n_models and collect predictions
            y_preds = []
            y_probas = []

            for i, model in enumerate(model_list):
                print(f" → Training model {i + 1}/{self.n_models}", flush = True)
                model.fit(X_resampled, y_resampled)
                y_preds.append(model.predict(X_test_scaled))
                y_probas.append(model.predict_proba(X_test_scaled)[:, 1])

            y_preds = np.array(y_preds)
            y_probas = np.array(y_probas)

            # Majority vote
            y_pred_final = (y_preds.sum(axis=0) >= (self.n_models / 2)).astype(int)

            # Average probabilities
            y_pred_proba_final = y_probas.mean(axis=0)

            # Evaluation metrics
            tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred_final).ravel()
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0

            metrics = {
                'accuracy': (tp + tn) / (tp + tn + fp + fn),
                'sensitivity': tp / (tp + fn),
                'specificity': tn / (tn + fp),
                'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'npv': npv,
                'f1_score': f1_score(self.y_test, y_pred_final),
                'auc_roc': roc_auc_score(self.y_test, y_pred_proba_final),
                'auc_pr': average_precision_score(self.y_test, y_pred_proba_final),
                'cv_auc_mean': cross_val_score(model_list[0], X_resampled, y_resampled, cv=skf, scoring='roc_auc').mean(),
                'cv_auc_std': cross_val_score(model_list[0], X_resampled, y_resampled, cv=skf, scoring='roc_auc').std(),
                'confusion_matrix': confusion_matrix(self.y_test, y_pred_final),
                'y_pred': y_pred_final,
                'y_pred_proba': y_pred_proba_final
            }

            self.results[name] = metrics

            elapsed_time = time.time() - start_time
            minutes, seconds = divmod(elapsed_time, 60)
            print(f"\n⏱️ Training and evaluation for {name} completed in {int(minutes)}m {seconds:.1f}s.")

            # Display metrics
            print(f"\nPerformance Metrics:")
            for k, v in metrics.items():
                if isinstance(v, float):
                    print(f"{k.replace('_', ' ').capitalize()}: {v:.4f}")
            print("\nConfusion Matrix:")
            print(metrics['confusion_matrix'])
            print(f"True Negatives: {tn}")
            print(f"False Positives: {fp}")
            print(f"False Negatives: {fn}")
            print(f"True Positives: {tp}")

            # Feature importances (optional from one model)
            if hasattr(model_list[0], 'feature_importances_'):
                importances = model_list[0].feature_importances_
                feature_imp = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)

                print(f"\nTop 15 Feature Importances:")
                for _, row in feature_imp.head(15).iterrows():
                    print(f"  {row['feature']}: {row['importance']:.4f}")

            elif hasattr(model_list[0], 'coef_'):
                coefficients = model_list[0].coef_[0]
                feature_coef = pd.DataFrame({
                    'feature': self.X_train.columns, ############### ORIGINALLY self.feature_names,
                    'coefficient': coefficients
                }).sort_values('coefficient', key=abs, ascending=False)

                print(f"\nTop 15 Feature Coefficients (by magnitude):")
                for _, row in feature_coef.head(15).iterrows():
                    print(f"  {row['feature']}: {row['coefficient']:.4f}")

        return True
    
    def analyze_feature_importance_by_group(self):
        """Analyze feature importance grouped by feature type"""
        print("\n" + "="*80, flush = True)
        print("FEATURE IMPORTANCE BY GROUP", flush = True)
        print("="*80, flush = True)
        
        # Get feature importance from tree-based models
        for model_name in ['Random Forest', 'XGBoost']:
            if model_name in self.models:
                model = self.models[model_name]['model']
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    
                    # Create importance DataFrame
                    imp_df = pd.DataFrame({
                        'feature': self.feature_names,
                        'importance': importances
                    })
                    
                    # Categorize features
                    def categorize_feature(feature):
                        if feature in self.mme_features:
                            return 'MME'
                        elif feature in self.prescriber_features:
                            return 'Prescriber'
                        elif feature in self.demographic_features:
                            return 'Demographic'
                        elif feature in self.sdoh_features:
                            return 'SDOH'
                        else:
                            return 'Other'
                    
                    imp_df['category'] = imp_df['feature'].apply(categorize_feature)
                    
                    # Group importance by category
                    category_importance = imp_df.groupby('category')['importance'].agg(['sum', 'mean', 'count'])
                    category_importance = category_importance.sort_values('sum', ascending=False)
                    
                    print(f"\n{model_name} - Importance by Feature Category:", flush = True)
                    print(category_importance.round(4), flush = True)
                    
                    # Plot
                    plt.figure(figsize=(10, 6))
                    category_importance['sum'].plot(kind='bar')
                    plt.title(f'{model_name} - Total Feature Importance by Category')
                    plt.xlabel('Feature Category')
                    plt.ylabel('Total Importance')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.savefig(f'feature_importance_by_category_{model_name.lower().replace(" ", "_")}.png', dpi=300)
                    plt.show()
    
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
        print("\n" + "="*80, flush = True)
        print("SUMMARY REPORT: OUD PREDICTION MODEL EVALUATION", flush = True)
        print("="*80, flush = True)
        
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
                'NPV': f"{metrics['npv']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}",
                'CV AUC': f"{metrics['cv_auc_mean']:.4f} (±{metrics['cv_auc_std']:.3f})"
            })
        
        summary_df = pd.DataFrame(summary_data)
        print("\nModel Performance Summary:", flush = True)
        print(summary_df.to_string(index=False), flush = True)
        
        # Healthcare-specific recommendations
        print("\n" + "-"*80, flush = True)
        print("HEALTHCARE-SPECIFIC INSIGHTS:", flush = True)
        print("-"*80, flush = True)
        
        # Find best model for different criteria
        best_sensitivity = max(self.results.items(), key=lambda x: x[1]['sensitivity'])
        best_precision = max(self.results.items(), key=lambda x: x[1]['precision'])
        best_f1 = max(self.results.items(), key=lambda x: x[1]['f1_score'])
        
        print(f"\n Clinical Recommendations:", flush = True)
        print(f"\n1. For SCREENING (maximize sensitivity to catch all potential OUD cases):", flush = True)
        print(f"   → Use {best_sensitivity[0]}", flush = True)
        print(f"   → Sensitivity: {best_sensitivity[1]['sensitivity']:.4f}", flush = True)
        print(f"   → This will identify {best_sensitivity[1]['sensitivity']*100:.1f}% of OUD cases", flush = True)
        
        print(f"\n2. For TARGETED INTERVENTION (balance precision and recall):", flush = True)
        print(f"   → Use {best_f1[0]}", flush = True)
        print(f"   → F1-Score: {best_f1[1]['f1_score']:.4f}", flush = True)
        print(f"   → Precision: {best_f1[1]['precision']:.4f}", flush = True)
        
        print(f"\n3. For RESOURCE-LIMITED SETTINGS (maximize precision):", flush = True)
        print(f"   → Use {best_precision[0]}", flush = True)
        print(f"   → Precision: {best_precision[1]['precision']:.4f}", flush = True)
        print(f"   → {best_precision[1]['precision']*100:.1f}% of positive predictions are correct", flush = True)
        
        # Feature insights
        print("\n" + "-"*80, flush = True)
        print("FEATURE INSIGHTS:", flush = True)
        print("-"*80, flush = True)
        print("\nThe models now incorporate:", flush = True)
        print("• Prescription patterns (MME features)", flush = True)
        print("• Healthcare utilization (prescriber counts)", flush = True)
        print("• Patient demographics (age, gender, insurance)", flush = True)
        print("• Social determinants of health (income, education, employment, etc.)", flush = True)
        print("\nThis comprehensive feature set enables better risk stratification", flush = True)
        print("and can help identify both clinical and social risk factors for OUD.", flush = True)
        
        # Save summary
        summary_df.to_csv('oud_model_summary_full_features.csv', index=False)
        print("\n✓ Summary saved to 'oud_model_summary_full_features.csv'", flush = True)
        
        return summary_df
    
    def run_complete_evaluation(self):
        """Run the complete model evaluation pipeline"""
        print("Starting OUD Prediction Model Evaluation with Full Feature Set", flush = True)
        print("="*80, flush = True)
        
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
        
        # Analyze feature importance by group
        self.analyze_feature_importance_by_group()
        
        # Generate summary
        summary = self.generate_summary_report()
        
        print("\n Evaluation completed successfully!", flush = True)
        
        return {
            'results': self.results,
            'summary': summary
        }

import argparse

def main():
    """Main function to run OUD prediction model evaluation"""

    parser = argparse.ArgumentParser(description="Run OUD prediction model evaluation.")
    parser.add_argument("--data_path", type=str, default="/sharefolder/wanglab/ML_training/final_OUD_ML_dataset.csv", help="Path to the dataset CSV file")
    parser.add_argument("--use_smote", action="store_true", help="Enable SMOTE oversampling")
    parser.add_argument("--use_rus", action="store_true", help="Enable random undersampling")
    parser.add_argument("--use_enn", action="store_true", help="Enable Edited Nearest Neighbors cleaning")
    parser.add_argument("--use_knn", action="store_true", help="Enable K-Nearest Neighbors cleaning")
    parser.add_argument("--n_models", type=int, default=1, help="Number of models to train in the ensemble")
    parser.add_argument("--corr_feat", action="store_true", help="Enable Removal of Correlated Features")
    parser.add_argument("--model_selection", nargs="+", default=["Logistic Regression", "Random Forest", "XGBoost"], help="List of models to include")

    args = parser.parse_args()

    evaluator = OUDModelEvaluator(
        data_path=args.data_path,
        use_smote=args.use_smote,
        use_rus=args.use_rus,
        use_enn=args.use_enn,
        use_knn=args.use_knn,
        n_models=args.n_models,
        corr_feat=args.corr_feat,
        model_selection=args.model_selection
    )

    results = evaluator.run_complete_evaluation()

    if results:
        print("\n" + "="*80, flush=True)
        print("EVALUATION COMPLETE", flush=True)
        print("="*80, flush=True)
        print(" All models trained and evaluated with full feature set", flush=True)
        print(" Visualizations saved", flush=True)
        print(" Summary report generated", flush=True)
        print("\n Output files:", flush=True)
        print("  - oud_model_metrics_comparison.png", flush=True)
        print("  - oud_roc_curves.png", flush=True)
        print("  - oud_pr_curves.png", flush=True)
        print("  - oud_confusion_matrices.png", flush=True)
        print("  - feature_importance_by_category_*.png", flush=True)
        print("  - oud_model_summary_full_features.csv", flush=True)

if __name__ == "__main__":
    main()
