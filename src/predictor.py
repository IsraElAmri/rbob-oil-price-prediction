"""
RBOB Gasoline Price Predictor
Backend module for data processing and model training.
Forecasting next-day RBOB gasoline price without feature leakage.
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class GasolinePredictor:
    """Predictor class for next-day RBOB gasoline price forecasting"""

    def __init__(self, data_path=None):
        if data_path is None:
            candidates = [
                'data/oil_prices.csv',
                '../data/oil_prices.csv',
                os.path.join(os.path.dirname(__file__), '..', 'data', 'oil_prices.csv'),
                'oil_prices.csv'
            ]
            for c in candidates:
                if os.path.exists(c):
                    data_path = c
                    break
            if data_path is None:
                data_path = 'data/oil_prices.csv'
        self.data_path = data_path
        self.df = None
        self.df_clean = None
        self.X = None
        self.y = None
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()

    def load_data(self):
        """Load and parse the CSV data"""
        self.df = pd.read_csv(self.data_path, sep=';', decimal=',', encoding='latin1')
        self.df.columns = self.df.columns.str.strip()

        date_col = 'date' if 'date' in self.df.columns else 'Date'
        month_map = {
            "janv": "Jan", "févr": "Feb", "mars": "Mar",
            "avr": "Apr", "mai": "May", "juin": "Jun",
            "juil": "Jul", "août": "Aug", "sept": "Sep",
            "oct": "Oct", "nov": "Nov", "déc": "Dec"
        }
        for fr, en in month_map.items():
            self.df[date_col] = self.df[date_col].astype(str).str.replace(fr, en, regex=False)
        self.df[date_col] = pd.to_datetime(self.df[date_col], format='%b %d, %Y', errors='coerce')
        self.df.sort_values(date_col, inplace=True)
        self.df.set_index(date_col, inplace=True)
        return self.df

    def clean_data(self):
        """Select core predictors and impute missing values"""
        if self.df is None:
            self.load_data()
        key_cols = [
            'RBOB',
            'Ultra-Low Sulfur CARB Diesel',
            'Kerosene-Type Jet Fuel',
            'Conventional Gasoline',
            'Heating Oil',
            'crude oil ( WTI)'
        ]
        available_cols = [c for c in key_cols if c in self.df.columns]
        self.df_clean = self.df[available_cols].copy()
        self.df_clean = self.df_clean.ffill().bfill()
        return self.df_clean

    def engineer_features(self):
        """Construct temporal, lag, and spread features without lookahead leakage"""
        if self.df_clean is None:
            self.clean_data()
        df = self.df_clean.copy()

        # Temporal features
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['day_of_week'] = df.index.dayofweek
        df['day_of_year'] = df.index.dayofyear

        # Lag-1 features (yesterday's prices)
        cols_to_lag = [c for c in self.df_clean.columns if c != 'RBOB']
        for col in cols_to_lag:
            df[f'{col}_lag1'] = df[col].shift(1)
        df['RBOB_lag1'] = df['RBOB'].shift(1)

        # Rolling 7-day statistics on lagged inputs (no lookahead)
        for col in cols_to_lag:
            df[f'{col}_ma7'] = df[col].rolling(7).mean()
            df[f'{col}_std7'] = df[col].rolling(7).std()

        # Price returns
        for col in cols_to_lag:
            df[f'{col}_return'] = df[col].pct_change() * 100

        # Spreads
        if 'crude oil ( WTI)' in df.columns:
            df['RBOB_WTI_spread'] = df['RBOB'] - (df['crude oil ( WTI)'] / 42.0)
        if 'Conventional Gasoline' in df.columns:
            df['RBOB_Conv_spread'] = df['RBOB'] - df['Conventional Gasoline']

        # Seasonality
        df['is_summer'] = ((df['month'] >= 5) & (df['month'] <= 8)).astype(int)
        df['is_winter'] = ((df['month'] == 12) | (df['month'] <= 2)).astype(int)

        # Drop NaN caused by shift/rolling
        df = df.dropna()

        # Target: Tomorrow's RBOB price (shift(-1))
        df['RBOB_target'] = df['RBOB'].shift(-1)
        df = df.dropna()

        feature_cols = [c for c in df.columns if c != 'RBOB_target']
        self.X = df[feature_cols]
        self.y = df['RBOB_target']
        self.df_modeled = df
        return df

    def prepare_train_test(self, target_col='RBOB_target'):
        if self.X is None:
            self.engineer_features()
        return self.X, self.y

    def train_models(self):
        """Train models using 5-fold TimeSeriesSplit walk-forward validation"""
        if self.X is None:
            self.engineer_features()

        models_config = {
            'Baseline (Persistence)': 'persistence',
            'Ridge Regression': Ridge(alpha=10.0),
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features='sqrt',
                random_state=42
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            )
        }

        tscv = TimeSeriesSplit(n_splits=5)
        results = {}

        for name, model in models_config.items():
            results[name] = {
                'mae_folds': [],
                'rmse_folds': [],
                'r2_folds': [],
                'predictions': [],
                'actuals': []
            }

            for train_idx, test_idx in tscv.split(self.X):
                X_tr, X_te = self.X.iloc[train_idx], self.X.iloc[test_idx]
                y_tr, y_te = self.y.iloc[train_idx], self.y.iloc[test_idx]

                if model == 'persistence':
                    y_pred = X_te['RBOB'].values
                else:
                    X_tr_sc = self.scaler.fit_transform(X_tr)
                    X_te_sc = self.scaler.transform(X_te)
                    model.fit(X_tr_sc, y_tr)
                    y_pred = model.predict(X_te_sc)

                results[name]['mae_folds'].append(mean_absolute_error(y_te, y_pred))
                results[name]['rmse_folds'].append(np.sqrt(mean_squared_error(y_te, y_pred)))
                results[name]['r2_folds'].append(r2_score(y_te, y_pred))
                results[name]['predictions'].extend(y_pred)
                results[name]['actuals'].extend(y_te.values)

            if model != 'persistence':
                X_all_sc = self.scaler.fit_transform(self.X)
                model.fit(X_all_sc, self.y)
                self.models[name] = model

            results[name]['mae'] = np.mean(results[name]['mae_folds'])
            results[name]['mae_std'] = np.std(results[name]['mae_folds'])
            results[name]['rmse'] = np.mean(results[name]['rmse_folds'])
            results[name]['r2'] = np.mean(results[name]['r2_folds'])

        self.results = results
        return results

    def predict(self, model_name='Ridge Regression'):
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")
        X_all_sc = self.scaler.transform(self.X)
        return self.models[model_name].predict(X_all_sc)

    def get_feature_importance(self, model_name='Random Forest', top_n=10):
        if model_name not in self.models:
            return None
        model = self.models[model_name]
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
        else:
            return None
        fi = pd.DataFrame({
            'feature': self.X.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        return fi.head(top_n)
