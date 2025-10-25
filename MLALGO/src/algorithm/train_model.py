from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def train_random_forest(X, y, save_path=None):
    """Train and optionally save a Random Forest model."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"✅ Model trained successfully — Accuracy: {acc:.2f}")
    if save_path:
        joblib.dump(model, save_path)
        print(f"💾 Model saved to {save_path}")
    return model
