# 📈 Stock Price Forecasting using LSTM

This project builds a deep learning-based forecasting system to predict stock prices using historical data. It demonstrates time-series modeling with LSTM networks, grid search optimization, and data visualization to compare model predictions against actual values.

---

## 🔍 Project Objectives

- Load and preprocess historical stock price data (Open, High, Low, Close, Volume)
- Build an LSTM-based time-series forecasting model
- Tune hyperparameters using grid search
- Evaluate model performance (MSE, MAE, R²)
- Visualize actual vs predicted prices and prediction error

---

## 🧠 Skills Gained

- Time series data preprocessing and sequence modeling
- LSTM model construction and training with Keras
- Hyperparameter optimization using GridSearchCV
- Model evaluation with common regression metrics
- Data visualization for deep learning outputs

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

## 🛠 Tools & Libraries Used

- Python 3.x
- Pandas, NumPy
- Matplotlib for plotting
- Scikit-learn for preprocessing, metrics, grid search
- TensorFlow / Keras for building LSTM model

---

## 📁 Dataset Structure

All stock price data is stored in `.csv` files within:

```
data/
  └── stocks/
       └── AAPL.csv
       └── TSLA.csv
       └── ...
```

Each `.csv` contains:

| Column      | Description                        |
|-------------|------------------------------------|
| `Date`      | Trading date                       |
| `Open`      | Price at market open               |
| `High`      | Highest price during the day       |
| `Low`       | Lowest price during the day        |
| `Close`     | Closing price                      |
| `Adj Close` | Adjusted closing price             |
| `Volume`    | Number of shares traded            |

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

### Run the model

```bash
python output.py
```

This will:
- Search for best hyperparameters using GridSearchCV
- Train the best model on historical data
- Display visualizations including:
  - Actual vs Predicted prices
  - Loss curve
  - Prediction error histogram

---

## 📊 Results

- Model is evaluated on test data using:
  - Mean Squared Error (MSE)
  - Mean Absolute Error (MAE)
  - R² Score
- Plots provide visual understanding of forecasting quality

---

## 📌 Example Output
## Result
![Result](https://github.com/user-attachments/assets/96be2fa5-8a0d-42a3-9341-e7c571249cf6)

## Loss Curve
![Training Loss Curve](https://github.com/user-attachments/assets/a3d8354b-9207-4cfb-8c92-e3b651334606)

## AAPL Adc Close
![AAPL Adc Close](https://github.com/user-attachments/assets/387591a2-ef9c-4a82-a0a3-a466d21a6770)


## Prediction Error Distrubution
![Prediction Error Distribution](https://github.com/user-attachments/assets/4fe5414b-b6fc-4909-81dd-d09a517704e1)

## Actual vs Predcited Scatter

![Actual vs Predicted Scatter](https://github.com/user-attachments/assets/9f8b7dd4-e364-4c53-904b-2c156f67e920)

---

## 🔧 Customization & Extensions

- **Additional technical indicators**: Add moving averages, RSI, or MACD as extra features.
- **Different architectures**: Swap to GRU, 1D-CNN, or Transformer-based models.
- **Automated hyperparameter tuning**: Integrate Optuna or Ray Tune.
- **Walk-forward validation**: Retrain model periodically to adapt to regime shifts.

---

## 📄 License

This project is released under the MIT License. Feel free to use and adapt it for your own research or projects.
