# -*- coding: utf-8 -*-
"""
Phishing Website Detection Machine Learning Project

Objective: Compare performance of three classification models 
(Logistic Regression, SVM, Random Forest) for detecting phishing websites.

Dataset Features:
- ranking: Page ranking
- isIp: Presence of IP address in URL
- valid: Domain registration status
- activeDuration: Domain age
- urlLen: URL length
- is@: Contains '@' symbol
- isredirect: Has redirect symbols (--)
- haveDash: Contains dashes in domain
- domainLen: Domain name length
- nosOfSubdomain: Number of subdomains
- label: 0 (legitimate) or 1 (phishing)
"""

# ======================
# 1. IMPORT LIBRARIES
# ======================
import numpy as np  # Numerical operations
import pandas as pd  # Data manipulation
import matplotlib.pyplot as plt  # Visualization
import seaborn as sns  # Enhanced visualization
from sklearn.model_selection import train_test_split  # Data splitting
from sklearn.preprocessing import StandardScaler  # Feature scaling
from sklearn.metrics import (accuracy_score, confusion_matrix, 
                           classification_report, roc_auc_score, 
                           roc_curve)  # Evaluation metrics
from sklearn.linear_model import LogisticRegression  # Linear model
from sklearn.svm import SVC  # Support Vector Machine
from sklearn.ensemble import RandomForestClassifier  # Ensemble method

# Set random seed for reproducibility
np.random.seed(42)

# ======================
# 2. DATA PREPARATION
# ======================
# Load dataset from CSV file
df = pd.read_csv("phishing_dataset.csv")

# Display basic dataset info
print("Dataset shape:", df.shape)
print("\nFirst 5 samples:")
print(df.head())

# Check class distribution
print("\nClass distribution (0=Legitimate, 1=Phishing):")
print(df['label'].value_counts(normalize=True))

# ======================
# 3. DATA PREPROCESSING
# ======================
# Separate features (X) and target (y)
# Drop 'domain' (text feature) and 'label' (target)
X = df.drop(['domain', 'label'], axis=1)  
y = df['label']

# Split data into training (80%) and testing (20%) sets
# stratify=y maintains class distribution in splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)

# Standardize features (center to mean and scale to unit variance)
# Critical for SVM and helpful for Logistic Regression
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit and transform training data
X_test = scaler.transform(X_test)  # Transform test data (using training params)

# ======================
# 4. MODEL DEFINITION
# ======================
"""
Three model types being compared:
1. Logistic Regression (Linear model)
   - Simple interpretable model
   - Good baseline performance
2. Support Vector Machine (SVM with RBF kernel)
   - Effective for high-dimensional data
   - Can capture complex relationships
3. Random Forest (Ensemble method)
   - Handles non-linear relationships well
   - Robust to outliers
"""
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,  # Increased iterations for convergence
        random_state=42
    ),
    "SVM": SVC(
        probability=True,  # Enable probability estimates
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        random_state=42  # For reproducible results
    )
}

# ======================
# 5. MODEL TRAINING & EVALUATION
# ======================
# Dictionary to store all evaluation results
results = {}

for name, model in models.items():
    print(f"\n=== Training {name} ===")
    
    # Train model on training data
    model.fit(X_train, y_train)
    
    # Make predictions on test set
    y_pred = model.predict(X_test)
    
    # Get probability estimates (for ROC AUC)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probability of class 1 (phishing)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    # Store results
    results[name] = {
        "model": model,
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "confusion_matrix": conf_matrix,
        "report": class_report,
        "y_prob": y_prob
    }
    
    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)

# ======================
# 6. VISUALIZATION
# ======================
# Plot ROC curves for model comparison
plt.figure(figsize=(8, 6))
for name in models:
    # Calculate ROC curve points
    fpr, tpr, _ = roc_curve(y_test, results[name]['y_prob'])
    
    # Plot curve with AUC score in legend
    plt.plot(fpr, tpr, label=f"{name} (AUC = {results[name]['roc_auc']:.2f})")

# Add diagonal reference line (random classifier)
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')

# Customize plot
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# ======================
# 7. FEATURE IMPORTANCE (FOR RANDOM FOREST)
# ======================
if hasattr(results['Random Forest']['model'], 'feature_importances_'):
    # Get feature importance scores
    importances = results['Random Forest']['model'].feature_importances_
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    sorted_features = X.columns[indices]
    sorted_importances = importances[indices]
    
    # Create bar plot
    plt.figure(figsize=(10, 6))
    plt.title("Random Forest Feature Importances")
    plt.bar(range(len(importances)), sorted_importances, align="center")
    plt.xticks(range(len(importances)), sorted_features, rotation=90)
    plt.ylabel("Importance Score")
    plt.tight_layout()
    plt.show()

"""
Key Analysis Points:
1. Accuracy shows overall correctness
2. ROC AUC measures class separation capability
3. Confusion matrix reveals false positives/negatives
4. Feature importance explains model decisions (for Random Forest)

Typical Expected Results:
- Random Forest usually achieves highest accuracy
- Logistic Regression provides fastest training
- SVM offers good balance but slower training
"""