"""
RBOB Gasoline Price Predictor
Backend module for data processing and model training
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class GasolinePredictor:
    """Main predictor class for RBOB gasoline price forecasting"""
    
    def __init__(self, data_path=None):
        import os
        if data_path is None:
            if os.path.exists('data/oil_prices.csv'):
                data_path = 'data/oil_prices.csv'
            elif os.path.exists('../data/oil_prices.csv'):
                data_path = '../data/oil_prices.csv'
            else:
                data_path = 'oil_prices.csv'
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.y_train = None
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Load and parse the CSV data"""
        # Load with correct separator and decimal
        self.df = pd.read_csv(self.data_path, sep=';', decimal=',', encoding='latin1')
        
        # French month mapping
        month_map = {
            "janv": "Jan", "févr": "Feb", "mars": "Mar",
            "avr": "Apr", "mai": "May", "juin": "Jun",
            "juil": "Jul", "août": "Aug", "sept": "Sep",
            "oct": "Oct", "nov": "Nov", "déc": "Dec"
        }
        
        # Replace French months with English
        for fr, en in month_map.items():
            self.df['date'] = self.df['date'].str.replace(fr, en)
        
        # Parse dates
        self.df['date'] = pd.to_datetime(self.df['date'], format="%b %d, %Y", errors="coerce")
        self.df.set_index('date', inplace=True)
        self.df.sort_index(inplace=True)
        
        return self.df
    
    def clean_data(self):
        """Clean and prepare the data"""
        # Convert object columns to numeric
        for col in self.df.columns:
            if self.df[col].dtype == "object":
                self.df[col] = (
                    self.df[col].astype(str)
                    .str.replace(r"[^\d\.\-]", "", regex=True)
                    .replace("", np.nan)
                )
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
        
        # Fill missing values
        self.df.fillna(method='ffill', inplace=True)
        self.df.fillna(method='bfill', inplace=True)
        
        return self.df
    
    def engineer_features(self):
        """Create features for modeling"""
        # Create lag features
        for col in self.df.columns:
            if col != ' RBOB':
                self.df[f'{col}_lag1'] = self.df[col].shift(1)
        
        # Create rolling statistics
        for col in [' RBOB', 'crude oil ( WTI)']:
            self.df[f'{col}_rolling_mean_7'] = self.df[col].rolling(window=7).mean()
            self.df[f'{col}_rolling_std_7'] = self.df[col].rolling(window=7).std()
        
        # Drop rows with NaN after feature engineering
        self.df.dropna(inplace=True)
        
        return self.df
    
    def prepare_train_test(self, target_col=' RBOB'):
        """Prepare training data"""
        # Separate features and target
        y = self.df[target_col]
        X = self.df.drop(columns=[target_col])
        
        self.X_train = X
        self.y_train = y
        
        return X, y
    
    def train_models(self):
        """Train multiple models"""
        # Define models
        models_config = {
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
            ),
            'Ridge': Ridge(alpha=10.0)
        }
        
        # Train with cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        for name, model in models_config.items():
            mae_scores = []
            rmse_scores = []
            r2_scores = []
            
            for train_idx, test_idx in tscv.split(self.X_train):
                X_tr, X_te = self.X_train.iloc[train_idx], self.X_train.iloc[test_idx]
                y_tr, y_te = self.y_train.iloc[train_idx], self.y_train.iloc[test_idx]
                
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_te)
                
                mae_scores.append(mean_absolute_error(y_te, y_pred))
                rmse_scores.append(np.sqrt(mean_squared_error(y_te, y_pred)))
                r2_scores.append(r2_score(y_te, y_pred))
            
            self.results[name] = {
                'MAE': np.mean(mae_scores),
                'RMSE': np.mean(rmse_scores),
                'R2': np.mean(r2_scores),
                'MAE_std': np.std(mae_scores)
            }
            
            # Train final model on all data
            model.fit(self.X_train, self.y_train)
            self.models[name] = model
        
        return self.results
    
    def predict(self, model_name='Random Forest'):
        """Make predictions using trained model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        predictions = self.models[model_name].predict(self.X_train)
        return predictions
    
    def get_feature_importance(self, model_name='Random Forest', top_n=10):
        """Get feature importance from tree-based models"""
        if model_name not in self.models:
            return None
        
        model = self.models[model_name]
        if not hasattr(model, 'feature_importances_'):
            return None
        
        importance_df = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        
        return importance_df
