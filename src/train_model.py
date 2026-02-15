import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

def train_and_evaluate():
    input_file = os.path.join("data", "processed", "pathway_features.csv")
    output_dir = os.path.join("data", "results")
    
    print("Loading features...")
    df = pd.read_csv(input_file)
    
    # Drop Name and ID for training
    X = df.drop(['Pathway_ID', 'Pathway_Name', 'AD_Related'], axis=1)
    y = df['AD_Related']
    
    print(f"Dataset shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    # Initialize Model
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    
    # Cross-Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("Performing 5-Fold Cross-Validation...")
    cv_scores = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc')
    print(f"Mean ROC-AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
    
    # predictions for detailed metrics
    y_pred_proba = cross_val_predict(clf, X, y, cv=cv, method='predict_proba')[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Metrics
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    
    print("\n--- Model Performance (CV) ---")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    
    # Feature Importance
    clf.fit(X, y) # Fit on full data for interpretation
    importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    
    print("\n--- Feature Importance ---")
    print(importances)
    
    # Save Report
    with open(os.path.join(output_dir, "ml_performance_report.txt"), "w") as f:
        f.write(f"Mean ROC-AUC: {cv_scores.mean():.3f}\n")
        f.write(f"Accuracy: {acc:.3f}\n")
        f.write(f"Precision: {prec:.3f}\n")
        f.write(f"Recall: {rec:.3f}\n")
        f.write(f"F1 Score: {f1:.3f}\n\n")
        f.write("Feature Importances:\n")
        f.write(importances.to_string())
        
    # Plot Feature Importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances.values, y=importances.index)
    plt.title("Feature Importance in Predicting AD Pathways")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance.png"))
    
    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"Random Forest (AUC = {roc_auc_score(y, y_pred_proba):.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "roc_curve.png"))
    plt.savefig(os.path.join(output_dir, "roc_curve.png"))
    
    # --- PROPOSED UPDATE: Comparison of ML vs Statistical Baseline ---
    # We want to show what the ML found that the Stats missed (or ranked lower).
    
    # Get probabilities for the whole dataset
    all_probs = clf.predict_proba(X)[:, 1]
    df['ML_Probability'] = all_probs
    
    # Rank by ML Probability
    df_compare = df[['Pathway_ID', 'Pathway_Name', 'n_proteins', 'AD_Related', 'ML_Probability']].copy()
    df_compare['Statistical_Rank'] = df_compare['n_proteins'].rank(ascending=False)
    df_compare['ML_Rank'] = df_compare['ML_Probability'].rank(ascending=False)
    
    # Calculate Rank Difference (Positive = ML likes it more than Stats)
    df_compare['Rank_Diff'] = df_compare['Statistical_Rank'] - df_compare['ML_Rank']
    
    df_compare = df_compare.sort_values(by='ML_Probability', ascending=False)
    
    compare_file = os.path.join(output_dir, "model_vs_stats_comparison.csv")
    df_compare.head(50).to_csv(compare_file, index=False)
    
    print("\n--- Top ML Predictions Comparison ---")
    print(df_compare[['Pathway_Name', 'n_proteins', 'ML_Probability', 'AD_Related']].head(10))
    
    print(f" Comparison saved to {compare_file}")
    print(f" Plots saved to {output_dir}")

if __name__ == "__main__":
    train_and_evaluate()
