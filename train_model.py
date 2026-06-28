import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

load_dotenv()
db_url = os.getenv("SUPABASE_DB_URL")
engine = create_engine(db_url)

print("Pulling engineered features down from Supabase...")
df = pd.read_sql("SELECT * FROM engineered_market_features", engine)

X = df.drop(columns=['date', 'ticker', 'volatility_5d', 'target_high_volatility'])
y = df['target_high_volatility']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

print(f"Training set: {X_train.shape[0]} rows | Testing set: {X_test.shape[0]} rows")
print("Hyperparameter tuning ...")

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

tscv = TimeSeriesSplit(n_splits=5)

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=tscv,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"\n Best Settings Found: {grid_search.best_params_}")

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("\n=========================================")
print("   OPTIMIZED MODEL PERFORMANCE REPORT   ")
print("=========================================")
print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.2%}\n")
print(classification_report(y_test, y_pred))
print("=========================================")

importances = best_model.feature_importances_
feature_names = X.columns
print("\n FEATURE IMPORTANCE RANKINGS:")
for name, importance in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
    print(f"- {name}: {importance:.2%}")