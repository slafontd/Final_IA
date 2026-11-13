from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyClassifier

FEATURES = ["pm25", "temperatura", "humedad", "viento", "estacion"]
TARGET = "categoria"

def train_and_evaluate(df):
    """Entrena regresión logística y compara contra baseline de clase más frecuente."""
    work = df.dropna(subset=[TARGET]).copy()
    X, y = work[FEATURES], work[TARGET]
    counts = y.value_counts()
    eligible = counts[counts >= 2].index
    mask = y.isin(eligible)
    X, y = X[mask], y[mask]
    if y.nunique() < 2 or len(y) < 10:
        raise ValueError("Se necesitan al menos 10 registros y dos categorías con 2 o más ejemplos.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    numeric = ["pm25", "temperatura", "humedad", "viento"]
    categorical = ["estacion"]
    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)
    ])
    model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1500, class_weight="balanced"))])
    baseline = DummyClassifier(strategy="most_frequent")
    model.fit(X_train, y_train)
    baseline.fit(X_train, y_train)
    pred, base_pred = model.predict(X_test), baseline.predict(X_test)
    metrics = {
        "accuracy_modelo": round(accuracy_score(y_test, pred), 4),
        "accuracy_baseline": round(accuracy_score(y_test, base_pred), 4),
        "f1_macro": round(f1_score(y_test, pred, average="macro", zero_division=0), 4),
        "reporte": classification_report(y_test, pred, zero_division=0)
    }
    return model, metrics

def predict_one(model, row):
    import pandas as pd
    frame = pd.DataFrame([row])[FEATURES]
    return str(model.predict(frame)[0])
