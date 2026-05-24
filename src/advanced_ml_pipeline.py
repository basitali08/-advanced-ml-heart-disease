import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
import optuna
import shap
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

print('='*70)
print('ADVANCED MACHINE LEARNING PIPELINE - HEART DISEASE PREDICTION')
print('='*70)

for d in ['models', 'results']:
    os.makedirs(d, exist_ok=True)

df = pd.read_csv('data/heart_disease_uci.csv')
print(f'\nDataset: {df.shape[0]} samples, {df.shape[1]} features')
print(f'Target distribution: {df.target.value_counts().to_dict()}')

# Feature engineering
df['age_squared'] = df['age'] ** 2
df['chol_bp_ratio'] = df['chol'] / df['trestbps']
df['age_hr_product'] = df['age'] * df['thalach']
df['bp_age_ratio'] = df['trestbps'] / df['age']

# Handle missing values
for col in df.columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

feature_cols = [c for c in df.columns if c != 'target']

X = df[feature_cols]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print(f'\nTrain: {len(X_train)}, Test: {len(X_test)}')
print(f'Features: {len(feature_cols)} (including engineered)')

# ============= OPTUNA HYPERPARAMETER TUNING =============
print('\n' + '='*70)
print('HYPERPARAMETER TUNING WITH OPTUNA')
print('='*70)

def objective_rf(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_float('max_features', 0.3, 1.0),
    }
    model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    scores = cross_val_score(model, X_train_s, y_train, cv=5, scoring='accuracy')
    return scores.mean()

study_rf = optuna.create_study(direction='maximize', study_name='RF Tuning')
study_rf.optimize(objective_rf, n_trials=20, show_progress_bar=False)
print(f'Best RF params: {study_rf.best_params}')
print(f'Best RF CV accuracy: {study_rf.best_value:.4f}')

def objective_gb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
    }
    model = GradientBoostingClassifier(**params, random_state=42)
    scores = cross_val_score(model, X_train_s, y_train, cv=5, scoring='accuracy')
    return scores.mean()

study_gb = optuna.create_study(direction='maximize', study_name='GB Tuning')
study_gb.optimize(objective_gb, n_trials=20, show_progress_bar=False)
print(f'Best GB params: {study_gb.best_params}')
print(f'Best GB CV accuracy: {study_gb.best_value:.4f}')

# ============= TRAIN MODELS =============
print('\n' + '='*70)
print('TRAINING MODELS')
print('='*70)

models = {
    'Logistic Regression': LogisticRegression(max_iter=2000, C=0.1, random_state=42),
    'Random Forest Tuned': RandomForestClassifier(**study_rf.best_params, random_state=42, n_jobs=-1),
    'Gradient Boosting Tuned': GradientBoostingClassifier(**study_gb.best_params, random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42),
}

results = []
best_model = None
best_score = 0
best_name = ''

for name, model in models.items():
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1] if hasattr(model, 'predict_proba') else None

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0
    cv_scores = cross_val_score(model, X_train_s, y_train, cv=10, scoring='accuracy')

    results.append({
        'Model': name,
        'Accuracy': round(accuracy, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1': round(f1, 4),
        'AUC': round(auc, 4),
        'CV_Mean': round(cv_scores.mean(), 4),
        'CV_Std': round(cv_scores.std(), 4)
    })

    status = '*** BEST ***' if accuracy > best_score else ''
    print(f'{name:30s} Acc: {accuracy:.4f} | AUC: {auc:.4f} | CV: {cv_scores.mean():.4f} (+/-{cv_scores.std():.4f}) {status}')

    if accuracy > best_score:
        best_score = accuracy
        best_model = model
        best_name = name

# Save best model
with open('models/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)
results_df.to_csv('results/model_comparison.csv', index=False)
print(f'\nBest model: {best_name} with accuracy {best_score:.4f}')

# ============= SHAP ANALYSIS =============
print('\n' + '='*70)
print('SHAP MODEL INTERPRETATION')
print('='*70)

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test_s)

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test_s, feature_names=feature_cols, show=False)
plt.tight_layout()
plt.savefig('results/shap_summary.png', dpi=200, bbox_inches='tight')
plt.close()
print('SHAP summary plot saved')

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_s, feature_names=feature_cols, plot_type='bar', show=False)
plt.tight_layout()
plt.savefig('results/shap_importance.png', dpi=200, bbox_inches='tight')
plt.close()
print('SHAP feature importance saved')

# ============= FINAL EVALUATION =============
print('\n' + '='*70)
print('FINAL EVALUATION')
print('='*70)

y_pred_best = best_model.predict(X_test_s)
y_prob_best = best_model.predict_proba(X_test_s)[:, 1]

print(f'Classification Report:\n{classification_report(y_test, y_pred_best)}')

cm = confusion_matrix(y_test, y_pred_best)
print(f'Confusion Matrix:\n{cm}')

fpr, tpr, _ = roc_curve(y_test, y_prob_best)
auc_score = roc_auc_score(y_test, y_prob_best)
print(f'ROC AUC: {auc_score:.4f}')

# ============= VISUALIZATIONS =============
print('\n' + '='*70)
print('GENERATING VISUALIZATIONS')
print('='*70)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Model comparison
ax = axes[0, 0]
colors = ['#2ecc71' if m == best_name else '#3498db' for m in results_df['Model']]
bars = ax.barh(results_df['Model'], results_df['Accuracy'], color=colors)
ax.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xlim(0, 1)
for bar, val in zip(bars, results_df['Accuracy']):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center', fontsize=10)

# 2. Confusion Matrix
ax = axes[0, 1]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
            xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease'])
ax.set_title(f'Confusion Matrix - {best_name}', fontsize=14, fontweight='bold')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')

# 3. ROC Curve
ax = axes[0, 2]
ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'{best_name} (AUC = {auc_score:.3f})')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax.fill_between(fpr, tpr, alpha=0.15, color='blue')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)

# 4. Feature Importance (SHAP)
ax = axes[1, 0]
if isinstance(shap_values, list):
    sv = shap_values[1]
elif hasattr(shap_values, 'shape') and shap_values.ndim == 3:
    sv = shap_values[:, :, 1]
else:
    sv = shap_values
sv_mean = np.abs(sv).mean(axis=0).flatten()
top_n = min(10, len(feature_cols))
shap_imp = pd.DataFrame({
    'Feature': feature_cols[:len(sv_mean)],
    'Importance': sv_mean
}).sort_values('Importance', ascending=True).tail(top_n)
ax.barh(shap_imp['Feature'], shap_imp['Importance'], color='coral')
ax.set_title('Top 10 Features (SHAP)', fontsize=14, fontweight='bold')
ax.set_xlabel('Mean |SHAP Value|')

# 5. CV Scores
ax = axes[1, 1]
models_list = [m for m in models.keys()]
cv_means = [r['CV_Mean'] for r in results]
cv_stds = [r['CV_Std'] for r in results]
ax.barh(models_list, cv_means, xerr=cv_stds, color='lightgreen', capsize=5)
ax.set_xlabel('Cross-Validation Accuracy', fontsize=12, fontweight='bold')
ax.set_title('10-Fold CV Scores', fontsize=14, fontweight='bold')
ax.set_xlim(0, 1)

# 6. Metrics comparison
ax = axes[1, 2]
metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
x = np.arange(len(metrics))
width = 0.2
for i, (_, row) in enumerate(results_df.iterrows()):
    vals = [row[m] for m in metrics]
    ax.bar(x + i*width, vals, width, label=row['Model'])
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(metrics)
ax.set_ylabel('Score')
ax.set_title('All Metrics Comparison', fontsize=14, fontweight='bold')
ax.legend(fontsize=8)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('results/all_plots.png', dpi=200, bbox_inches='tight')
print('All plots saved to results/all_plots.png')

print('\n' + '='*70)
print('PIPELINE COMPLETE!')
print('='*70)
print(f'\nBest Model: {best_name}')
print(f'Accuracy: {best_score:.4f}')
print(f'ROC AUC: {auc_score:.4f}')
print(f'\nFiles saved:')
print(f'  - models/best_model.pkl')
print(f'  - models/scaler.pkl')
print(f'  - results/model_comparison.csv')
print(f'  - results/shap_summary.png')
print(f'  - results/shap_importance.png')
print(f'  - results/all_plots.png')