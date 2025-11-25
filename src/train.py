import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import json

# 1. Load Data
print("Loading data...")
df = pd.read_csv('../data/dataset.csv', sep=';')

# 2. Preprocessing Setup
X = df.drop('y', axis=1)
y = df['y'].apply(lambda x: 1 if x == 'yes' else 0)

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# --- FEATURES ---
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, interaction_only=False, include_bias=False))
])

categorical_transformer = OneHotEncoder(handle_unknown='ignore', drop='first')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 3. Build the Pipeline
#  'liblinear' handles both l1 and l2 penalties well.
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(solver='liblinear', random_state=42, max_iter=1000))
])

# 4. HYPERPARAMETER TUNING

param_dist = {
    # C: Regularization Strength. (0.01 = Strong penalty, 100 = Weak penalty)
    'classifier__C': np.logspace(-2, 2, 20), 
    
    # Penalty: l1 (Lasso) removes features, l2 (Ridge) shrinks them.
    'classifier__penalty': ['l1', 'l2'],
    
    # Class Weight: 'balanced' boosts the minority class. None treats them equally.
    'classifier__class_weight': ['balanced', None]
}

print("Starting Hyperparameter Tuning (RandomizedSearchCV)...")
# We increase n_iter to 20 to try more combinations.
search = RandomizedSearchCV(pipeline, param_dist, n_iter=20, cv=3, scoring='f1', random_state=42, verbose=1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Run the search
search.fit(X_train, y_train)

print(f"\nBest Parameters Found: {search.best_params_}")
print(f"Best Cross-Validation Score (F1): {search.best_score_:.4f}")

# 5. Final Evaluation
best_model = search.best_estimator_
print("\nEvaluating Best Model on Test Set...")
score = best_model.score(X_test, y_test)
print(f"Test Accuracy: {score:.4f}")
print("Classification Report:")
print(classification_report(y_test, best_model.predict(X_test)))

# --- Extract Coefficients ---
print("\nExtracting Feature Importance...")

# 1. Get the Classifier and Preprocessor from the pipeline
classifier = best_model.named_steps['classifier']
preprocessor = best_model.named_steps['preprocessor']

# 2. Get all feature names (including interaction terms)

feature_names = preprocessor.get_feature_names_out()

# 3. Get the coefficients
coeffs = classifier.coef_[0]

# 4. Combine them into a DataFrame
importance_df = pd.DataFrame({
    'feature': feature_names,
    'coefficient': coeffs,
    'abs_importance': np.abs(coeffs)
})

# 5. Sort by importance (absolute value)
top_features = importance_df.sort_values('abs_importance', ascending=False).head(20)

# 6. Convert to dictionary for JSON
top_features_dict = top_features[['feature', 'coefficient']].to_dict(orient='records')

print("\nTop 5 Most Important Features:")
print(top_features[['feature', 'coefficient']].head(5))

# 6. Save Artifacts
joblib.dump(best_model, 'model.joblib')

mlops_report = {
    "model_type": "LogisticRegression",
    "hyperparameters": search.best_params_,
    "metrics": {
        "test_accuracy": score,
        "classification_report": classification_report(y_test, best_model.predict(X_test), output_dict=True)
    },
    "top_20_features": top_features_dict  # <--- We added this!
}

# Convert numpy types to python types for JSON serialization
def convert(o):
    if isinstance(o, np.generic): return o.item()
    raise TypeError

with open("metrics.json", "w") as f:
    json.dump(mlops_report, f, indent=4, default=convert)

print("\nSUCCESS: Model pipeline saved to src/model.joblib")
print("SUCCESS: Metrics with coefficients saved to src/metrics.json")