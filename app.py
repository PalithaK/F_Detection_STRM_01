import numpy as np
import pandas as pd
import pickle
import json
import time
import joblib
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve
)
from xgboost import XGBClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


# Configuration & Constants

@dataclass
class SystemConfig:
    """Central configuration for the fraud detection system"""

    n_transactions: int = 50000
    fraud_rate: float = 0.015

    xgb_weight: float = 0.6
    nn_weight: float = 0.4
    threshold: float = 0.5

    test_size: float = 0.3
    val_size: float = 0.2
    random_seed: int = 42

    false_positive_cost: float = 10.0
    false_negative_cost: float = 100.0

    target_latency_ms: float = 100.0

config = SystemConfig()


# Data Generator Class

class FraudDataGenerator:
    """
    Simulates realistic credit card transaction data with fraud patterns.
    Includes temporal patterns, device fingerprints, and behavioral anomalies.
    """

    def __init__(self, n_transactions: int = 100000, fraud_rate: float = 0.015):
        self.n_transactions = n_transactions
        self.fraud_rate = fraud_rate
        self.n_fraud = int(n_transactions * fraud_rate)
        self.n_legit = n_transactions - self.n_fraud

    def generate_transactions(self) -> pd.DataFrame:
        """Generate synthetic transaction dataset with realistic patterns"""
        np.random.seed(42)

        n = self.n_transactions
        base_time = datetime(2024, 1, 1)
        timestamps = [base_time + timedelta(
            minutes=np.random.randint(0, 30*24*60)
        ) for _ in range(n)]
        customer_ids = np.random.randint(1, 5000, n)
        segments = np.random.choice(['premium', 'standard', 'basic'], n, p=[0.2, 0.5, 0.3])
        amounts = np.random.lognormal(mean=3.5, sigma=1.2, size=n)
        amounts = np.clip(amounts, 0.5, 10000)
        tx_types = np.random.choice(
            ['online', 'pos', 'atm', 'mobile'],
            n,
            p=[0.4, 0.3, 0.2, 0.1]
        )
        devices = np.random.choice(
            ['ios', 'android', 'windows', 'unknown'],
            n,
            p=[0.35, 0.35, 0.2, 0.1]
        )
        distance_from_home = np.random.exponential(scale=50, size=n)
        hour_of_day = np.random.randint(0, 24, n)
        is_night = (hour_of_day < 6) | (hour_of_day > 22)
        prev_tx_count = np.random.poisson(lam=15, size=n)
        prev_tx_avg_amount = np.random.exponential(scale=80, size=n)
        tx_velocity_1h = np.random.poisson(lam=1.5, size=n)

        df = pd.DataFrame({
            'timestamp': timestamps,
            'customer_id': customer_ids,
            'segment': segments,
            'amount': amounts,
            'tx_type': tx_types,
            'device': devices,
            'distance_from_home': distance_from_home,
            'hour_of_day': hour_of_day,
            'is_night': is_night.astype(int),
            'prev_tx_count': prev_tx_count,
            'prev_tx_avg_amount': prev_tx_avg_amount,
            'tx_velocity_1h': tx_velocity_1h,
            'is_fraud': 0
        })

        fraud_indices = self._generate_realistic_fraud_patterns(df)
        df.loc[fraud_indices, 'is_fraud'] = 1
        df['is_foreign'] = (df['distance_from_home'] > 500).astype(int)
        df['is_high_amount'] = (df['amount'] > df['amount'].quantile(0.95)).astype(int)
        df['is_high_velocity'] = (df['tx_velocity_1h'] > 10).astype(int)
        return df

    def _generate_realistic_fraud_patterns(self, df: pd.DataFrame) -> List[int]:
        """Inject fraud patterns that mimic real fraud scenarios"""
        fraud_indices = []
        n_fraud = self.n_fraud

        n1 = int(n_fraud * 0.3)
        candidates = df[
            (df['amount'] > df['amount'].quantile(0.85)) &
            (df['distance_from_home'] > 200) &
            (df['is_night'] == 1)
        ].index
        if len(candidates) >= n1:
            fraud_indices.extend(np.random.choice(candidates, n1, replace=False))

        n2 = int(n_fraud * 0.25)
        candidates = df[
            (df['tx_velocity_1h'] > 8) &
            (df['amount'] < df['amount'].quantile(0.3))
        ].index
        if len(candidates) >= n2:
            fraud_indices.extend(np.random.choice(candidates, n2, replace=False))

        n3 = int(n_fraud * 0.2)
        candidates = df[
            (df['device'] == 'unknown') &
            (df['amount'] > df['amount'].quantile(0.9))
        ].index
        if len(candidates) >= n3:
            fraud_indices.extend(np.random.choice(candidates, n3, replace=False))

        n4 = int(n_fraud * 0.15)
        candidates = df[
            (df['segment'] == 'premium') &
            (df['prev_tx_avg_amount'] < 50) &
            (df['amount'] > 500)
        ].index
        if len(candidates) >= n4:
            fraud_indices.extend(np.random.choice(candidates, n4, replace=False))

        n5 = n_fraud - len(fraud_indices)
        if n5 > 0:
            remaining = [i for i in df.index if i not in fraud_indices]
            if len(remaining) >= n5:
                fraud_indices.extend(np.random.choice(remaining, n5, replace=False))

        return list(fraud_indices)


# Feature Engineering

class FeatureEngineer:
    """Feature engineering pipeline with encoding and scaling"""

    def __init__(self):
        self.encoders = {}
        self.scaler = RobustScaler()
        self.feature_names = []
        self.config = {
            'numerical_features': ['amount', 'distance_from_home', 'prev_tx_count',
                                  'prev_tx_avg_amount', 'tx_velocity_1h'],
            'categorical_features': ['segment', 'tx_type', 'device'],
            'temporal_features': ['hour_of_day', 'is_night'],
            'interaction_features': ['is_foreign', 'is_high_amount', 'is_high_velocity']
        }

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit encoders and transform data"""
        encoded_dfs = []

        for cat_feat in self.config['categorical_features']:
            if cat_feat in df.columns:
                dummies = pd.get_dummies(df[cat_feat], prefix=cat_feat, drop_first=True)
                encoded_dfs.append(dummies)
                self.encoders[cat_feat] = list(dummies.columns)

        num_df = df[self.config['numerical_features']].copy()
        temp_df = df[self.config['temporal_features']].copy()
        inter_df = df[self.config['interaction_features']].copy()

        X = pd.concat([num_df, temp_df, inter_df] + encoded_dfs, axis=1)
        self.feature_names = X.columns.tolist()

        num_cols = self.config['numerical_features'] + self.config['temporal_features']
        num_cols = [c for c in num_cols if c in X.columns]
        X[num_cols] = self.scaler.fit_transform(X[num_cols])

        return X.values

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted encoders"""
        encoded_dfs = []

        for cat_feat in self.config['categorical_features']:
            if cat_feat in df.columns:
                dummies = pd.get_dummies(df[cat_feat], prefix=cat_feat, drop_first=True)
                for col in self.encoders.get(cat_feat, []):
                    if col not in dummies.columns:
                        dummies[col] = 0
                encoded_dfs.append(dummies[self.encoders[cat_feat]])

        num_df = df[self.config['numerical_features']].copy()
        temp_df = df[self.config['temporal_features']].copy()
        inter_df = df[self.config['interaction_features']].copy()

        X = pd.concat([num_df, temp_df, inter_df] + encoded_dfs, axis=1)

        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0

        X = X[self.feature_names]

        num_cols = self.config['numerical_features'] + self.config['temporal_features']
        num_cols = [c for c in num_cols if c in X.columns]
        X[num_cols] = self.scaler.transform(X[num_cols])

        return X.values


# Model Definition

class _NeuralNetwork(nn.Sequential): # Changed to inherit directly from nn.Sequential
    def __init__(self, n_features):
        super().__init__( # Pass layers directly to nn.Sequential's constructor
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1) # Removed nn.Sigmoid() here as BCEWithLogitsLoss handles it
        )

    # The forward method is automatically handled by nn.Sequential


class FraudDetectionEnsemble:
    """Ensemble of XGBoost and Neural Network for fraud detection"""

    def __init__(self, n_features: int, threshold: float = 0.5):
        self.n_features = n_features
        self.threshold = threshold
        self.xgb_model = None
        self.nn_model = None
        self.xgb_weight = config.xgb_weight
        self.nn_weight = config.nn_weight

    def build_neural_network(self) -> _NeuralNetwork:
        """Build a deep neural network for fraud detection"""
        return _NeuralNetwork(self.n_features)

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None):
        """Train both models in the ensemble"""
        print("\n" + "="*60)
        print("TRAINING ENSEMBLE MODEL")
        print("="*60)

        fraud_ratio = y.mean()
        scale_pos_weight = (1 - fraud_ratio) / fraud_ratio

        # Train XGBoost
        print("\n[1/2] Training XGBoost...")
        self.xgb_model = XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=config.random_seed,
            n_jobs=-1,
            eval_metric='aucpr'
        )

        if X_val is not None:
            self.xgb_model.fit(
                X, y,
                eval_set=[(X, y), (X_val, y_val)],
                verbose=50
            )
        else:
            self.xgb_model.fit(X, y)

        # Train Neural Network with PyTorch
        print("\n[2/2] Training Neural Network (PyTorch)...")
        self.nn_model = self.build_neural_network()
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([scale_pos_weight]))
        optimizer = optim.Adam(self.nn_model.parameters(), lr=0.001)

        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y).float().unsqueeze(1)
        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

        if X_val is not None:
            X_val_tensor = torch.from_numpy(X_val).float()
            y_val_tensor = torch.from_numpy(y_val).float().unsqueeze(1)
        else:
            # If no explicit validation set, use a split from training data
            # This part will need careful adjustment for consistent splitting if no X_val is provided.
            # For now, let's assume X_val is always provided or handle it outside this block for simplicity.
            X_val_tensor = X_tensor[:int(len(X_tensor)*0.2)] # Example placeholder
            y_val_tensor = y_tensor[:int(len(y_tensor)*0.2)] # Example placeholder
        best_val_auc = -np.inf
        patience_counter = 0
        epochs = 100 # Max epochs

        for epoch in range(epochs):
            self.nn_model.train()
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = self.nn_model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            self.nn_model.eval()
            with torch.no_grad():
                val_outputs = self.nn_model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
                val_proba = torch.sigmoid(val_outputs).cpu().numpy().flatten()
                val_auc = roc_auc_score(y_val, val_proba)

            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val AUC: {val_auc:.4f}")

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                # Save best model state (optional, for restoration)
                # torch.save(self.nn_model.state_dict(), 'best_nn_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= 20: # Early stopping patience
                    print("Early stopping triggered")
                    break

        print("\n✅ Ensemble training complete!")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get ensemble probability predictions"""
        xgb_proba = self.xgb_model.predict_proba(X)[:, 1]
        
        self.nn_model.eval() # Set model to evaluation mode
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float()
            nn_outputs = self.nn_model(X_tensor)
            nn_proba = torch.sigmoid(nn_outputs).cpu().numpy().flatten()
        
        ensemble_proba = (self.xgb_weight * xgb_proba +
                          self.nn_weight * nn_proba)
        return ensemble_proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Get binary predictions"""
        proba = self.predict_proba(X)
        return (proba >= self.threshold).astype(int)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Comprehensive model evaluation"""
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)

        # Individual model predictions for comparison
        xgb_proba = self.xgb_model.predict_proba(X)[:, 1]
        xgb_pred = (xgb_proba >= self.threshold).astype(int)

        self.nn_model.eval() # Set model to evaluation mode
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float()
            nn_outputs = self.nn_model(X_tensor)
            nn_proba = torch.sigmoid(nn_outputs).cpu().numpy().flatten()
        nn_pred = (nn_proba >= self.threshold).astype(int)

        results = {
            'ensemble': {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred),
                'recall': recall_score(y, y_pred),
                'f1': f1_score(y, y_pred),
                'auc_roc': roc_auc_score(y, y_proba),
                'confusion_matrix': confusion_matrix(y, y_pred).tolist()
            },
            'xgb': {
                'accuracy': accuracy_score(y, xgb_pred),
                'precision': precision_score(y, xgb_pred),
                'recall': recall_score(y, xgb_pred),
                'f1': f1_score(y, xgb_pred),
                'auc_roc': roc_auc_score(y, xgb_proba)
            },
            'nn': {
                'accuracy': accuracy_score(y, nn_pred),
                'precision': precision_score(y, nn_pred),
                'recall': recall_score(y, nn_pred),
                'f1': f1_score(y, nn_pred),
                'auc_roc': roc_auc_score(y, nn_proba)
            }
        }

        return results


# Real-Time Inference Engine

class FraudDetectionAPI:
    """Real-time fraud detection inference engine"""

    def __init__(self, model: FraudDetectionEnsemble,
                 feature_engineer: FeatureEngineer):
        self.model = model
        self.feature_engineer = feature_engineer
        self.metrics = {
            'total_predictions': 0,
            'fraud_alerts': 0,
            'avg_inference_time_ms': 0,
            'latency_breakdown': []
        }
        self.prediction_log = []

    def predict_transaction(self, transaction: Dict) -> Dict:
        """Real-time prediction for a single transaction"""
        start_time = time.time()

        # Convert to DataFrame
        df = pd.DataFrame([transaction])

        # Feature engineering
        X = self.feature_engineer.transform(df)

        # Get prediction
        proba = self.model.predict_proba(X)[0]
        fraud_score = float(proba)
        is_fraud = int(fraud_score >= self.model.threshold)

        # Calculate inference time
        inference_time_ms = (time.time() - start_time) * 1000

        # Update metrics
        self.metrics['total_predictions'] += 1
        self.metrics['fraud_alerts'] += is_fraud
        self.metrics['avg_inference_time_ms'] = (
            (self.metrics['avg_inference_time_ms'] *
             (self.metrics['total_predictions'] - 1) + inference_time_ms) /
            self.metrics['total_predictions']
        )
        self.metrics['latency_breakdown'].append(inference_time_ms)

        # Log prediction
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'transaction_id': transaction.get('transaction_id', 'N/A'),
            'amount': transaction.get('amount', 0),
            'fraud_score': fraud_score,
            'prediction': is_fraud,
            'inference_time_ms': inference_time_ms
        }
        self.prediction_log.append(log_entry)

        # Response
        response = {
            'transaction_id': transaction.get('transaction_id', 'N/A'),
            'prediction': '🚨 FRAUD' if is_fraud else '✅ LEGIT',
            'fraud_score': fraud_score,
            'confidence': self._get_confidence_bucket(fraud_score),
            'threshold': self.model.threshold,
            'inference_time_ms': round(inference_time_ms, 2),
            'timestamp': datetime.now().isoformat()
        }

        return response

    def _get_confidence_bucket(self, score: float) -> str:
        """Categorize confidence level"""
        if score > 0.9: return '🔴 VERY HIGH'
        elif score > 0.7: return '🟡 HIGH'
        elif score > 0.5: return '🟢 MEDIUM'
        else: return '⚪ LOW'

    def batch_predict(self, transactions: List[Dict]) -> List[Dict]:
        """Batch prediction for multiple transactions"""
        return [self.predict_transaction(tx) for tx in transactions]

    def get_metrics(self) -> Dict:
        """Get API performance metrics"""
        return self.metrics

    def get_latency_stats(self) -> Dict:
        """Get latency statistics"""
        latencies = self.metrics['latency_breakdown']
        if not latencies:
            return {}
        return {
            'min_ms': min(latencies),
            'max_ms': max(latencies),
            'mean_ms': np.mean(latencies),
            'median_ms': np.median(latencies),
            'p95_ms': np.percentile(latencies, 95)
        }


# Load Production System

def load_fraud_detection_system():
    """Load saved models for production inference"""
    # Load config first to get n_features for _NeuralNetwork
    with open('models/fraud_ensemble_config.json', 'r') as f:
        config_dict = json.load(f)

    feature_engineer = joblib.load('models/fraud_feature_engineer.pkl')
    xgb_model = joblib.load('models/fraud_xgb_model.pkl')
    
    nn_model_state_dict = torch.load('models/fraud_nn_model.pt')
    nn_model = _NeuralNetwork(config_dict['n_features'])
    nn_model.load_state_dict(nn_model_state_dict)
    nn_model.eval() # Set to evaluation mode

    class LoadedEnsemble:
        def __init__(self, xgb, nn, config_dict):
            self.xgb_model = xgb
            self.nn_model = nn
            self.threshold = config_dict['threshold']
            self.xgb_weight = config_dict['xgb_weight']
            self.nn_weight = config_dict['nn_weight']
            self.nn_model.eval() # Ensure NN model is in eval mode upon loading

        def predict_proba(self, X):
            xgb_proba = self.xgb_model.predict_proba(X)[:, 1]
            
            with torch.no_grad():
                X_tensor = torch.from_numpy(X).float()
                nn_outputs = self.nn_model(X_tensor)
                nn_proba = torch.sigmoid(nn_outputs).cpu().numpy().flatten()
            
            return self.xgb_weight * xgb_proba + self.nn_weight * nn_proba

        def predict(self, X):
            return (self.predict_proba(X) >= self.threshold).astype(int)

    ensemble = LoadedEnsemble(xgb_model, nn_model, config_dict)
    api = FraudDetectionAPI(ensemble, feature_engineer)

    return api


st.set_page_config(layout="wide")

st.title("💳 Real-Time Fraud Detection System")

@st.cache_resource
def get_fraud_detection_api():
    return load_fraud_detection_system()

api = get_fraud_detection_api()

st.write("""\nThis application demonstrates a real-time fraud detection system.\nIt loads pre-trained models (XGBoost + Neural Network ensemble) and simulates real-time transaction processing.\n\n**System Status:** ✅ Online\n**How it works:**\n1.  A pre-trained ensemble model (XGBoost + Neural Network) is loaded.\n2.  You can input transaction details, and the system will predict the fraud likelihood.\n3.  The system also reports inference latency.\n""")

st.sidebar.header("Transaction Input")

with st.sidebar.form("transaction_form"):
    st.subheader("Transaction Details")
    transaction_id = st.text_input("Transaction ID", f"TX_{\nnp.random.randint(10000, 99999)}")
    amount = st.number_input("Amount ($)", min_value=0.01, value=50.00, step=10.0)
    segment = st.selectbox("Customer Segment", ['basic', 'standard', 'premium'])
    tx_type = st.selectbox("Transaction Type", ['pos', 'online', 'atm', 'mobile'])
    device = st.selectbox("Device", ['ios', 'android', 'windows', 'unknown'])
    distance_from_home = st.slider("Distance from Home (km)", min_value=0.0, max_value=1000.0, value=15.0)
    hour_of_day = st.slider("Hour of Day (0-23)", min_value=0, max_value=23, value=datetime.now().hour)
    is_night = st.checkbox("Is Night Time?", value=False)
    prev_tx_count = st.number_input("Previous Transaction Count", min_value=0, value=10)
    prev_tx_avg_amount = st.number_input("Previous Transaction Avg Amount ($)", min_value=0.0, value=75.0)
    tx_velocity_1h = st.number_input("Transaction Velocity (1h)", min_value=0, value=2)
    is_foreign = st.checkbox("Is Foreign Transaction?", value=False)
    is_high_amount = st.checkbox("Is High Amount Transaction?", value=False)
    is_high_velocity = st.checkbox("Is High Velocity Transaction?", value=False)

    submitted = st.form_submit_button("Predict Fraud")

if submitted:
    transaction = {
        'transaction_id': transaction_id,
        'amount': amount,
        'segment': segment,
        'tx_type': tx_type,
        'device': device,
        'distance_from_home': distance_from_home,
        'hour_of_day': hour_of_day,
        'is_night': int(is_night),
        'prev_tx_count': prev_tx_count,
        'prev_tx_avg_amount': prev_tx_avg_amount,
        'tx_velocity_1h': tx_velocity_1h,
        'is_foreign': int(is_foreign),
        'is_high_amount': int(is_high_amount),
        'is_high_velocity': int(is_high_velocity)
    }

    st.subheader("Prediction Results: ")
    prediction_result = api.predict_transaction(transaction)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Prediction", prediction_result['prediction'])
    with col2:
        st.metric("Fraud Score", f"\{prediction_result['fraud_score']:.4f}")

    st.info(f"**Confidence:** \{prediction_result['confidence']}\n    | **Latency:** \{prediction_result['inference_time_ms']:.2f} ms")

    st.write("---")
    st.subheader("Transaction Details Submitted: ")
    st.json(transaction)

# Display historical predictions (optional)
if st.expander("View Recent Predictions"):
    if api.prediction_log:
        log_df = pd.DataFrame(api.prediction_log).sort_values('timestamp', ascending=False)
        st.dataframe(log_df)
    else:
        st.write("No recent predictions yet.")
