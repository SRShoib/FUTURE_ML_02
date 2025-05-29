# Stock Price Forecasting with PyTorch

This project implements a multi-feature LSTM-based time series forecasting model using PyTorch to predict daily stock prices. It performs a manual grid search over key hyperparameters, trains a final model, evaluates on a held-out test set, and provides comprehensive visualizations of results.

---

## 📂 Project Structure

```text
├── data/
│   ├── stocks/                 # Directory containing per-ticker CSV files
│   │   └── AAPL.csv            # Example: Apple Inc. price data
│   └── etfs/                   # (Optional) ETF price data
├── stock_price_forecasting_pytorch.py  # Main training & evaluation script
└── README.md                   # Project overview and usage instructions
```

---

## ⚙️ Dependencies

- Python 3.7+  
- [PyTorch](https://pytorch.org/)  
- scikit-learn  
- pandas  
- numpy  
- matplotlib  

Install all requirements via:

```bash
pip install torch scikit-learn pandas numpy matplotlib
```

---

## 📝 Configuration

At the top of `stock_price_forecasting_pytorch.py`, adjust the following constants as needed:

| Variable     | Description                                       | Default        |
|--------------|---------------------------------------------------|----------------|
| DATA_DIR     | Path to the folder with CSV price files           | `./data/stocks`|
| TICKER       | Ticker symbol to forecast (filename without .csv) | `AAPL`         |
| FEATURES     | List of feature columns used for training         | Open, High, Low, Adj Close, Volume |
| TARGET       | Column to predict (adjusted close)                | `Adj Close`    |
| SEED         | Random seed for reproducibility                   | `42`           |
| TEST_SIZE    | Fraction of data reserved for test evaluation     | `0.2`          |
| EPOCHS       | Number of training epochs                         | `30`           |
| VAL_SPLIT    | Fraction of train split used for validation       | `0.1`          |
| param_grid   | Hyperparameter grid for manual search             | see script     |

---

## 🚀 Usage

1. Place your per-ticker CSV files (e.g., `AAPL.csv`, `GOOG.csv`) under `data/stocks/` (or `data/etfs/`).
2. Modify configuration constants in the script if necessary.
3. Run the script:

   ```bash
   python stock_price_forecasting_pytorch.py
   ```

The script will:
- Read and sort the CSV by date.
- Scale feature columns using Min-Max normalization.
- Generate overlapping sequences for `lookback` days to predict the next-day price.
- Perform a manual grid search over:
  - `lookback` window sizes (e.g., 30 vs 60 days)
  - LSTM `hidden_size` values (e.g., 50 vs 100)
  - `dropout` rates (0.2 vs 0.3)
  - Learning rates (`lr`: 1e-3 vs 1e-4)
  - `batch_size` (16 vs 32)
- Print the best hyperparameters found.
- Retrain a final model on the full training set.
- Evaluate on the held-out test set and print:
  - Test MSE
  - Test MAE
  - Test R²
- Generate and display:
  1. Training loss curve  
  2. Actual vs. Predicted price line plot  
  3. Actual vs. Predicted scatter plot  
  4. Prediction error histogram

---

## 📊 Sample Output

```text
Testing: lookback=30, hidden=50, drop=0.2, lr=0.001, bs=16
...
Best hyperparameters: {'lookback': 60, 'hidden_size': 100, 'dropout': 0.2, 'lr': 0.001, 'batch_size': 32}
Final model trained.
Test MSE: 25.1234
Test MAE: 3.4567
Test R²:  0.8921
```

Visualizations will pop up in separate windows (or inline if using a Jupyter notebook).

---

## 🔧 Customization & Extensions

- **Additional technical indicators**: Add moving averages, RSI, or MACD as extra features.
- **Different architectures**: Swap to GRU, 1D-CNN, or Transformer-based models.
- **Automated hyperparameter tuning**: Integrate Optuna or Ray Tune.
- **Walk-forward validation**: Retrain model periodically to adapt to regime shifts.

---

## 📄 License

This project is released under the MIT License. Feel free to use and adapt it for your own research or projects.
