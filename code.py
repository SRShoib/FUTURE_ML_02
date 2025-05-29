import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, RegressorMixin

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.random import set_seed

# Point directly to your AAPL.csv:
DATA_DIR   = "./data/stocks"    # or "./data/etfs"

TICKER     = "AAPL"
FEATURES   = ["Open","High","Low","Adj Close","Volume"]
TARGET     = "Adj Close"
SEED       = 42
TEST_SIZE  = 0.2
EPOCHS     = 30
CV_FOLDS   = 3

set_seed(SEED)
np.random.seed(SEED)

# LOAD & SORT
df = pd.read_csv(os.path.join(DATA_DIR, f"{TICKER}.csv"), parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# SCALE FEATURES
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[FEATURES])

# SEQUENCE GENERATOR
def make_sequences(data, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i, FEATURES.index(TARGET)])
    return np.array(X), np.array(y)

# CUSTOM ESTIMATOR
class SequenceRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        units=50,
        dropout=0.2,
        lr=1e-3,
        lookback=60,
        epochs=EPOCHS,
        batch_size=32,
        verbose=0
    ):
        self.units       = units
        self.dropout     = dropout
        self.lr          = lr
        self.lookback    = lookback
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.verbose     = verbose

    def _build_model(self):
        m = Sequential([
            Bidirectional(LSTM(self.units, return_sequences=True),
                          input_shape=(self.lookback, len(FEATURES))),
            Dropout(self.dropout),
            Bidirectional(LSTM(self.units // 2)),
            Dropout(self.dropout),
            Dense(1)
        ])
        m.compile(
            optimizer=Adam(learning_rate=self.lr),
            loss="mse"
        )
        return m

    def fit(self, X, y):
        Xs = X[:, -self.lookback:, :]
        self.model_ = self._build_model()
        self.model_.fit(
            Xs, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose
        )
        return self

    def predict(self, X):
        Xs = X[:, -self.lookback:, :]
        return self.model_.predict(Xs, verbose=0).ravel()

# GRID-SEARCH SETUP
param_grid = {
    "units":      [50, 100],
    "dropout":    [0.2, 0.3],
    "lr":         [1e-3, 1e-4],
    "lookback":   [30, 60],
    "batch_size": [16, 32],
}

# Prebuild sequences for the largest lookback
max_lb = max(param_grid["lookback"])
X_all, y_all = make_sequences(scaled, max_lb)

base = SequenceRegressor(epochs=EPOCHS, verbose=0)
grid = GridSearchCV(
    estimator=base,
    param_grid=param_grid,
    cv=CV_FOLDS,
    scoring="neg_mean_squared_error",
    n_jobs=1
)
grid.fit(X_all, y_all)

best = grid.best_params_
print("Best hyperparameters:", best)

# RE-GENERATE SEQUENCES WITH BEST LOOKBACK
lb = best["lookback"]
X, y = make_sequences(scaled, lb)

# TRAIN/TEST SPLIT
split = int((1 - TEST_SIZE) * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# FINAL MODEL TRAINING
final = SequenceRegressor(
    units      = best["units"],
    dropout    = best["dropout"],
    lr         = best["lr"],
    lookback   = best["lookback"],
    epochs     = EPOCHS,
    batch_size = best["batch_size"],
    verbose    = 1
)
history = final.fit(X_train, y_train)

# PREDICTION & INVERSE SCALE
y_pred = final.predict(X_test)

inv_test = scaler.inverse_transform(
    np.hstack([
        np.zeros((len(y_test), len(FEATURES)-1)),
        y_test.reshape(-1,1)
    ])
)[:, -1]

inv_pred = scaler.inverse_transform(
    np.hstack([
        np.zeros((len(y_pred), len(FEATURES)-1)),
        y_pred.reshape(-1,1)
    ])
)[:, -1]

# METRICS
print(f"Test MSE: {mean_squared_error(inv_test, inv_pred):.4f}")
print(f"Test MAE: {mean_absolute_error(inv_test, inv_pred):.4f}")
print(f"Test R²:  {r2_score(inv_test, inv_pred):.4f}")

# VISUALIZATIONS
dates = df["Date"].iloc[split + lb:]

# a) Actual vs Predicted Time Series
plt.figure(figsize=(12,6))
plt.plot(dates, inv_test,  label="Actual")
plt.plot(dates, inv_pred,  label="Predicted")
plt.title(f"AAPL Adj Close: Actual vs Predicted")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.show()

# b) Training Loss Curve
plt.figure(figsize=(8,4))
plt.plot(history.model_.history.history['loss'], label='Train Loss')
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.tight_layout()
plt.show()

# c) Scatter Actual vs Predicted
plt.figure(figsize=(6,6))
plt.scatter(inv_test, inv_pred, alpha=0.5)
mn, mx = min(inv_test.min(), inv_pred.min()), max(inv_test.max(), inv_pred.max())
plt.plot([mn, mx], [mn, mx], 'r--')
plt.title('Actual vs Predicted Scatter')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.tight_layout()
plt.show()

# d) Error Histogram
errors = inv_pred - inv_test
plt.figure(figsize=(8,4))
plt.hist(errors, bins=30)
plt.title('Prediction Error Distribution')
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()
