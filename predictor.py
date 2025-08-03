import numpy as np
import pandas as pd
import requests
import json
import os
import time
import joblib
from threading import Thread
from datetime import datetime
from collections import deque
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from typing import Dict, List, Tuple, Optional
import xgboost as xgb

class NumberPredictionAI:
    def __init__(self, config: Dict):
        """Initialize prediction system with configuration"""
        self.config = config
        self.api_headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config['api_token']}"
        }
        self.number_scaler = MinMaxScaler()
        self.color_scaler = MinMaxScaler()
        self.number_encoder = LabelEncoder()
        self.color_encoder = LabelEncoder()
        self.number_model = None
        self.color_model = None
        self.xgb_number_model = None
        self.xgb_color_model = None
        self.history = pd.DataFrame(columns=['roundId', 'runnerName', 'color', 'timestamp'])
        self.number_sequence = deque(maxlen=config.get('seq_length', 5))
        self.color_sequence = deque(maxlen=config.get('seq_length', 5))
        self._running = False
        self._update_thread = None
        self.market_info = None
        self.color_map = {
            '0': 'VIOLET', '1': 'GREEN', '2': 'RED', '3': 'GREEN',
            '4': 'RED', '5': 'VIOLET', '6': 'RED', '7': 'GREEN',
            '8': 'RED', '9': 'GREEN'
        }
        # Store feature column names for consistency
        self.number_feature_cols = None
        self.color_feature_cols = None
        self._prepare_directories()
        self._initialize_encoders()
        self._load_resources()

    def _prepare_directories(self):
        """Ensure required directories exist"""
        os.makedirs(self.config['model_dir'], exist_ok=True)

    def _initialize_encoders(self):
        """Initialize encoders with all possible classes"""
        # Initialize number encoder with all possible numbers
        self.number_encoder.fit(list(self.color_map.keys()))
        # Initialize color encoder with all possible colors
        self.color_encoder.fit(['RED', 'GREEN', 'VIOLET'])

    def _load_resources(self):
        """Load pre-trained models and preprocessors if available"""
        try:
            number_model_path = os.path.join(self.config['model_dir'], 'number_model.keras')
            color_model_path = os.path.join(self.config['model_dir'], 'color_model.keras')
            xgb_number_model_path = os.path.join(self.config['model_dir'], 'xgb_number_model.pkl')
            xgb_color_model_path = os.path.join(self.config['model_dir'], 'xgb_color_model.pkl')
            number_scaler_path = os.path.join(self.config['model_dir'], 'number_scaler.pkl')
            color_scaler_path = os.path.join(self.config['model_dir'], 'color_scaler.pkl')
            number_encoder_path = os.path.join(self.config['model_dir'], 'number_encoder.pkl')
            color_encoder_path = os.path.join(self.config['model_dir'], 'color_encoder.pkl')

            if os.path.exists(number_model_path):
                self.number_model = load_model(number_model_path)
            if os.path.exists(color_model_path):
                self.color_model = load_model(color_model_path)
            if os.path.exists(xgb_number_model_path):
                self.xgb_number_model = joblib.load(xgb_number_model_path)
            if os.path.exists(xgb_color_model_path):
                self.xgb_color_model = joblib.load(xgb_color_model_path)
            if os.path.exists(number_scaler_path):
                self.number_scaler = joblib.load(number_scaler_path)
            if os.path.exists(color_scaler_path):
                self.color_scaler = joblib.load(color_scaler_path)
            if os.path.exists(number_encoder_path):
                self.number_encoder = joblib.load(number_encoder_path)
            if os.path.exists(color_encoder_path):
                self.color_encoder = joblib.load(color_encoder_path)
                
            # Load feature column names if available
            number_features_path = os.path.join(self.config['model_dir'], 'number_features.json')
            color_features_path = os.path.join(self.config['model_dir'], 'color_features.json')
            if os.path.exists(number_features_path):
                with open(number_features_path, 'r') as f:
                    self.number_feature_cols = json.load(f)
            if os.path.exists(color_features_path):
                with open(color_features_path, 'r') as f:
                    self.color_feature_cols = json.load(f)
                    
        except Exception as e:
            print(f"⚠️ Failed loading resources: {str(e)}")

    def _call_api(self, endpoint: str, payload: Dict) -> Dict:
        """Generic API call method with retry logic"""
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))
        try:
            response = session.post(
                f"{self.config['api_base_url']}/{endpoint}",
                headers=self.api_headers,
                json=payload,
                timeout=15
            )
            response.raise_for_status()
            print(f"✅ API call to {endpoint} successful")
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"⚠️ API call to {endpoint} failed: {str(e)}")
            return None

    def fetch_markets(self) -> Dict:
        """Fetch current market data with validation"""
        payload = {
            "token": self.config['api_token'],
            "operatorId": self.config['operator_id'],
            "partnerId": self.config['partner_id'],
            "providerId": self.config['provider_id'],
            "gameId": self.config['game_id'],
            "tableId": self.config['table_id']
        }
        response = self._call_api("getmarkets", payload)
        if response and response.get('status') == "RS_OK":
            self.market_info = response
            print(f"✅ Fetched market data: {len(response.get('table', {}).get('markets', []))} markets")
            return response
        print("⚠️ Failed to fetch valid market data")
        return None

    def fetch_results(self, count: int = 1) -> pd.DataFrame:
        """Fetch historical results with validation and timestamp"""
        payload = {
            "token": self.config['api_token'],
            "providerId": self.config['provider_id'],
            "tableId": self.config['table_id'],
            "resultsCount": min(count, 50)
        }
        response = self._call_api("get-last-results", payload)
        if not response or 'result' not in response or not response['result']:
            print("⚠️ No valid results in API response")
            return pd.DataFrame(columns=['roundId', 'runnerName', 'color', 'timestamp'])
        
        results = pd.DataFrame(response['result'])
        if not {'roundId', 'runnerName'}.issubset(results.columns):
            print("⚠️ Missing required columns in results")
            return pd.DataFrame(columns=['roundId', 'runnerName', 'color', 'timestamp'])
            
        results['runnerName'] = results['runnerName'].astype(str)
        valid_runners = [str(k) for k in self.color_map.keys()]
        results = results[results['runnerName'].isin(valid_runners)]
        if results.empty:
            print("⚠️ No valid runnerName values in results")
            return pd.DataFrame(columns=['roundId', 'runnerName', 'color', 'timestamp'])
            
        results['color'] = results['runnerName'].map(self.color_map)
        results['timestamp'] = pd.to_datetime(datetime.now())
        print(f"✅ Fetched {len(results)} results")
        return results

    def _extract_features_simple(self, sequence: List[str], target_type: str) -> np.ndarray:
        """Extract simple features from a sequence for prediction - matching training feature size"""
        if target_type == 'number':
            possible_values = list(self.color_map.keys())
        else:
            possible_values = ['RED', 'GREEN', 'VIOLET']
        
        features = []
        
        # Basic frequency features for the sequence
        for val in possible_values:
            freq = sequence.count(val) / len(sequence)
            features.append(freq)
        
        # Position-based features
        for i, val in enumerate(sequence):
            one_hot = [1.0 if val == pv else 0.0 for pv in possible_values]
            features.extend(one_hot)
        
        return np.array(features)

    def _extract_features(self, data: pd.DataFrame, target_type: str) -> Tuple[pd.DataFrame, List[str]]:
        """Extract consistent features for pattern analysis"""
        if data.empty:
            print(f"⚠️ Empty data for {target_type} feature extraction")
            return pd.DataFrame(), []
        
        features = data.copy()
        if target_type == 'number':
            target_col = 'runnerName'
            possible_values = list(self.color_map.keys())
        else:
            target_col = 'color'
            possible_values = ['RED', 'GREEN', 'VIOLET']
        
        # Validate target column
        if target_col not in features.columns or features[target_col].isna().all():
            print(f"⚠️ Invalid or missing {target_col} in features")
            return pd.DataFrame(), []
        
        # Ensure target column has valid values
        features[target_col] = features[target_col].astype(str)
        features = features[features[target_col].isin(possible_values)]
        if features.empty:
            print(f"⚠️ No valid {target_col} values in features")
            return pd.DataFrame(), []
        
        # Create feature columns with consistent naming
        feature_cols = []
        
        # Basic value encoding (one-hot style)
        for val in possible_values:
            col_name = f'{target_col}_{val}'
            features[col_name] = (features[target_col] == val).astype(float)
            feature_cols.append(col_name)
        
        # Rolling statistics with fixed window
        window = min(5, len(features))
        
        # Frequency features
        for val in possible_values:
            col_name = f'{target_col}_freq_{val}'
            features[col_name] = features[f'{target_col}_{val}'].rolling(window=window, min_periods=1).mean()
            feature_cols.append(col_name)
        
        # Streak features
        for val in possible_values:
            col_name = f'{target_col}_streak_{val}'
            # Calculate consecutive occurrences
            mask = features[target_col] == val
            features[col_name] = mask.groupby((~mask).cumsum()).cumsum().astype(float)
            feature_cols.append(col_name)
        
        # Time since last occurrence
        for val in possible_values:
            col_name = f'{target_col}_last_{val}'
            mask = features[target_col] == val
            features[col_name] = features.index.to_series().where(mask).ffill()
            features[col_name] = features.index.to_series() - features[col_name]
            features[col_name] = features[col_name].fillna(len(features)).astype(float)
            feature_cols.append(col_name)
        
        # Ensure all feature columns are numeric and handle NaN values
        for col in feature_cols:
            features[col] = pd.to_numeric(features[col], errors='coerce').fillna(0)
        
        # Drop rows with any NaN values in feature columns
        features = features.dropna(subset=feature_cols)
        
        if features.empty:
            print(f"⚠️ Feature extraction for {target_type} resulted in empty DataFrame")
            return pd.DataFrame(), []
        
        print(f"✅ Extracted {len(feature_cols)} features for {target_type}")
        return features, feature_cols

    def preprocess_data(self, data: pd.DataFrame, target_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for model training with consistent features"""
        try:
            min_data = self.config['seq_length'] * 2
            if data.empty or len(data) < min_data:
                print(f"⚠️ Insufficient {target_type} data for preprocessing: {len(data)} rows, need {min_data}")
                return np.array([]), np.array([])
                
            if target_type == 'number':
                encoder = self.number_encoder
                scaler = self.number_scaler
                target_col = 'runnerName'
                num_classes = len(self.color_map)
                possible_values = list(self.color_map.keys())
            else:
                encoder = self.color_encoder
                scaler = self.color_scaler
                target_col = 'color'
                num_classes = 3
                possible_values = ['RED', 'GREEN', 'VIOLET']
            
            encoded = encoder.transform(data[target_col])
            
            # Prepare sequences using simple feature extraction
            sequences = []
            labels = []
            seq_length = self.config['seq_length']
            
            for i in range(len(data) - seq_length):
                seq_data = data[target_col].iloc[i:i+seq_length].tolist()
                if len(seq_data) == seq_length:
                    # Extract simple features for the sequence
                    seq_features = self._extract_features_simple(seq_data, target_type)
                    sequences.append(seq_features)
                    labels.append(encoded[i+seq_length])
                
            if not sequences:
                print(f"⚠️ No {target_type} sequences created")
                return np.array([]), np.array([])
                
            X = np.array(sequences)
            y = to_categorical(np.array(labels), num_classes=num_classes)
            
            # Store the feature size for consistency
            if target_type == 'number':
                self.number_feature_size = X.shape[1]
            else:
                self.color_feature_size = X.shape[1]
            
            # Reshape for LSTM (samples, timesteps=1, features)
            X = X.reshape(X.shape[0], 1, X.shape[1])
            
            # Fit and transform scaler
            scaler.fit(X.reshape(-1, X.shape[-1]))
            X = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
            
            print(f"✅ Preprocessed {target_type} data: X shape {X.shape}, y shape {y.shape}")
            return X, y
            
        except Exception as e:
            print(f"⚠️ {target_type} preprocessing error: {str(e)}")
            import traceback
            traceback.print_exc()
            return np.array([]), np.array([])

    def build_lstm_model(self, input_shape: Tuple, output_size: int) -> Sequential:
        """Build LSTM model"""
        try:
            model = Sequential([
                Input(shape=input_shape),
                LSTM(32, return_sequences=False, recurrent_dropout=0.2),
                BatchNormalization(),
                Dense(16, activation='relu'),
                Dropout(0.3),
                Dense(output_size, activation='softmax')
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.0005),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            print(f"✅ Built LSTM model with input shape {input_shape} and output size {output_size}")
            return model
        except Exception as e:
            print(f"⚠️ LSTM model building failed: {str(e)}")
            return None

    def build_xgb_model(self, output_size: int) -> GradientBoostingClassifier:
        """Build XGBoost model for ensemble prediction"""
        try:
            model = xgb.XGBClassifier(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                objective='multi:softmax',
                num_class=output_size,
                random_state=42
            )
            print(f"✅ Built XGBoost model with output size {output_size}")
            return model
        except Exception as e:
            print(f"⚠️ XGBoost model building failed: {str(e)}")
            return None

    def train(self, X: np.ndarray, y: np.ndarray, model_type: str, epochs: int = 15) -> Optional[Dict]:
        """Train both LSTM and XGBoost models"""
        if X.size == 0 or y.size == 0:
            print(f"⚠️ No {model_type} data to train on")
            return None
            
        # Train LSTM model
        lstm_model = self.number_model if model_type == 'number' else self.color_model
        callbacks = [
            EarlyStopping(patience=3, restore_best_weights=True, monitor='val_loss'),
            ModelCheckpoint(
                os.path.join(self.config['model_dir'], f'{model_type}_model.keras'),
                save_best_only=True,
                monitor='val_loss'
            )
        ]
        
        try:
            history = lstm_model.fit(
                X, y,
                epochs=epochs,
                batch_size=4,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=1
            )
            
            # Save scaler and encoder
            scaler = self.number_scaler if model_type == 'number' else self.color_scaler
            encoder = self.number_encoder if model_type == 'number' else self.color_encoder
            
            joblib.dump(scaler, os.path.join(self.config['model_dir'], f'{model_type}_scaler.pkl'))
            joblib.dump(encoder, os.path.join(self.config['model_dir'], f'{model_type}_encoder.pkl'))
            
            print(f"✅ Trained LSTM {model_type} model")
            
            # Train XGBoost model
            xgb_model = self.xgb_number_model if model_type == 'number' else self.xgb_color_model
            # Flatten the data for XGBoost
            X_xgb = X.reshape(X.shape[0], -1)
            y_xgb = np.argmax(y, axis=1)
            xgb_model.fit(X_xgb, y_xgb)
            joblib.dump(xgb_model, os.path.join(self.config['model_dir'], f'xgb_{model_type}_model.pkl'))
            print(f"✅ Trained XGBoost {model_type} model")
            
            return history.history
        except Exception as e:
            print(f"⚠️ {model_type} training failed: {str(e)}")
            return None

    def update_model(self, new_data_count: int = 50) -> bool:
        """Update both LSTM and XGBoost models with fresh data"""
        if self.fetch_markets() is None:
            print("⚠️ Failed to fetch markets in update_model")
            return False
            
        new_data = self.fetch_results(new_data_count)
        if len(new_data) < self.config['seq_length'] * 2:
            print(f"⚠️ Not enough data for training: {len(new_data)} rows, need {self.config['seq_length'] * 2}")
            return False
            
        # Update history
        if self.history.empty:
            self.history = new_data
        else:
            new_data = new_data[self.history.columns]
            self.history = pd.concat([new_data, self.history], ignore_index=True).drop_duplicates(subset=['roundId']).head(100)
        
        print(f"✅ Updated history with {len(self.history)} unique records")
        
        # Update number model
        X_num, y_num = self.preprocess_data(self.history, 'number')
        if X_num.size == 0 or y_num.size == 0:
            print("⚠️ No valid number training data")
            return False
            
        if self.number_model is None:
            self.number_model = self.build_lstm_model(
                input_shape=(X_num.shape[1], X_num.shape[2]),
                output_size=y_num.shape[1]
            )
            if self.number_model is None:
                return False
        if self.xgb_number_model is None:
            self.xgb_number_model = self.build_xgb_model(output_size=y_num.shape[1])
            if self.xgb_number_model is None:
                return False
                
        self.train(X_num, y_num, 'number')
        
        # Update color model
        X_col, y_col = self.preprocess_data(self.history, 'color')
        if X_col.size == 0 or y_col.size == 0:
            print("⚠️ No valid color training data")
            return False
            
        if self.color_model is None:
            self.color_model = self.build_lstm_model(
                input_shape=(X_col.shape[1], X_col.shape[2]),
                output_size=y_col.shape[1]
            )
            if self.color_model is None:
                return False
        if self.xgb_color_model is None:
            self.xgb_color_model = self.build_xgb_model(output_size=y_col.shape[1])
            if self.xgb_color_model is None:
                return False
                
        self.train(X_col, y_col, 'color')
        
        # Initialize sequences
        if len(self.history) >= self.config['seq_length']:
            self.number_sequence.clear()
            self.color_sequence.clear()
            self.number_sequence.extend(self.history['runnerName'].tail(self.config['seq_length']).tolist())
            self.color_sequence.extend(self.history['color'].tail(self.config['seq_length']).tolist())
            print(f"✅ Initialized sequences: {len(self.number_sequence)} numbers, {len(self.color_sequence)} colors")
        
        return True

    def predict_next(self, sequence: List[str], model_type: str) -> Dict:
        """Generate ensemble prediction with consistent feature extraction"""
        if len(sequence) < self.config['seq_length']:
            print(f"⚠️ Insufficient {model_type} sequence length: {len(sequence)} (expected {self.config['seq_length']})")
            return {}
            
        try:
            lstm_model = self.number_model if model_type == 'number' else self.color_model
            xgb_model = self.xgb_number_model if model_type == 'number' else self.xgb_color_model
            encoder = self.number_encoder if model_type == 'number' else self.color_encoder
            scaler = self.number_scaler if model_type == 'number' else self.color_scaler
            
            # Extract simple features for the sequence
            seq_features = self._extract_features_simple(sequence[-self.config['seq_length']:], model_type)
            
            # Prepare LSTM input
            lstm_input = seq_features.reshape(1, 1, -1)
            lstm_input = scaler.transform(lstm_input.reshape(-1, lstm_input.shape[-1])).reshape(lstm_input.shape)
            lstm_probs = lstm_model.predict(lstm_input, verbose=0)[0]
            
            # Prepare XGBoost input
            xgb_input = seq_features.reshape(1, -1)
            xgb_probs = xgb_model.predict_proba(xgb_input)[0]
            
            # Ensemble prediction
            ensemble_probs = 0.6 * lstm_probs + 0.4 * xgb_probs
            print(f"✅ Generated {model_type} ensemble prediction")
            
            # Prepare market predictions
            predictions = {}
            if self.market_info is None:
                self.fetch_markets()
            
            market_type = "MATCH_ODDS" if model_type == 'number' else "COLOR"
            
            if self.market_info and 'table' in self.market_info:
                target_market = None
                for market in self.market_info['table']['markets']:
                    if market['marketType'] == market_type:
                        target_market = market
                        break
                
                if target_market:
                    for runner in target_market['runners']:
                        if runner['runnerName'] in encoder.classes_:
                            idx = list(encoder.classes_).index(runner['runnerName'])
                            prob = ensemble_probs[idx]
                            
                            back_price = None
                            if runner['backPrices']:
                                back_price = runner['backPrices'][0]['price']
                            
                            expected_value = prob * (back_price - 1) if back_price else -1
                            
                            predictions[runner['runnerName']] = {
                                'probability': float(prob),
                                'back_price': float(back_price) if back_price else None,
                                'runner_id': runner['runnerId'],
                                'market_id': target_market['marketId'],
                                'market_type': target_market['marketType'],
                                'confidence': float(prob * 100),
                                'expected_value': float(expected_value)
                            }
            
            return predictions
            
        except Exception as e:
            print(f"⚠️ {model_type} prediction error: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}

    def analyze_patterns(self):
        """Analyze historical data for statistical anomalies"""
        if self.history.empty:
            print("⚠️ No historical data for pattern analysis")
            return
        
        print("\n📈 Pattern Analysis:")
        # Color frequency
        color_counts = self.history['color'].value_counts(normalize=True) * 100
        print("Color Frequencies:")
        for color, freq in color_counts.items():
            expected = 33.33 if color in ['RED', 'GREEN', 'VIOLET'] else 10.0
            print(f"{color}: {freq:.2f}% (Expected: {expected:.2f}%)")
        
        # Streak analysis
        streaks = (self.history['color'] == self.history['color'].shift(1)).cumsum()
        streak_counts = streaks.value_counts()
        print("\nStreak Lengths (Consecutive Same Colors):")
        for streak, count in streak_counts.items():
            print(f"Streak {streak}: {count} occurrences")
        
        # Chi-squared test for randomness
        try:
            from scipy.stats import chi2_contingency
            observed = self.history['color'].value_counts().reindex(['RED', 'GREEN', 'VIOLET'], fill_value=0).values
            expected = np.array([len(self.history) / 3] * 3)
            chi2, p_value = chi2_contingency([observed, expected])[:2]
            print(f"\nChi-Squared Test for Randomness: p-value = {p_value:.4f}")
            if p_value < 0.05:
                print("⚠️ Possible non-random pattern detected (p < 0.05)")
        except ImportError:
            print("\n⚠️ scipy not available for chi-squared test")

    def start_real_time_prediction(self):
        """Start continuous prediction service"""
        print("⏳ Initializing with historical data...")
        try:
            if self.fetch_markets() is None:
                raise ConnectionError("Failed to fetch market information")
            
            if not self.update_model(self.config['initial_data_count']):
                raise ValueError("Failed to initialize models")
                
        except Exception as e:
            print(f"❌ Initial training failed: {str(e)}")
            return
            
        self._running = True
        self._update_thread = Thread(
            target=self._continuous_update,
            daemon=True
        )
        self._update_thread.start()
        print(f"🚀 Real-time prediction active (updating every {self.config['update_interval']}s)")

    def _continuous_update(self):
        """Background update thread with incremental training and pattern analysis"""
        update_count = 0
        while self._running:
            try:
                if update_count % 10 == 0:
                    self.fetch_markets()
                
                new_data = self.fetch_results(1)
                
                if not new_data.empty:
                    latest_round = new_data.iloc[0]['roundId']
                    print(f"Checking new result: roundId {latest_round}")
                    if self.history.empty or latest_round not in self.history['roundId'].values:
                        self.history = pd.concat([new_data, self.history], ignore_index=True).drop_duplicates(subset=['roundId']).head(100)
                        self.number_sequence.append(new_data.iloc[0]['runnerName'])
                        self.color_sequence.append(new_data.iloc[0]['color'])
                        print(f"✅ Added new result: number {new_data.iloc[0]['runnerName']}, color {new_data.iloc[0]['color']}")
                        
                        self._online_learn()
                        if update_count % 20 == 0:
                            self.analyze_patterns()
                    
                    if len(self.number_sequence) >= self.config['seq_length']:
                        self._generate_real_time_prediction()
                    else:
                        print(f"⚠️ Waiting for sufficient sequence length: {len(self.number_sequence)}/{self.config['seq_length']}")
                
                else:
                    print("⚠️ No new data fetched")
                
                update_count += 1
            except Exception as e:
                print(f"⚠️ Update error: {str(e)}")
            
            time.sleep(self.config['update_interval'])

    def _online_learn(self):
        """Incremental model update with latest data"""
        try:
            print("🔁 Learning from recent patterns...")
            recent_data = self.history.head(100)
            if len(recent_data) < self.config['seq_length'] * 2:
                print(f"⚠️ Insufficient data for online learning: {len(recent_data)} rows")
                return
                
            # Update number model
            X_num, y_num = self.preprocess_data(recent_data, 'number')
            if X_num.size > 0 and y_num.size > 0 and self.number_model is not None:
                self.number_model.fit(X_num, y_num, epochs=2, batch_size=4, verbose=0)
                self.number_model.save(os.path.join(self.config['model_dir'], 'number_model.keras'))
                X_xgb = X_num.reshape(X_num.shape[0], -1)
                y_xgb = np.argmax(y_num, axis=1)
                if self.xgb_number_model is not None:
                    self.xgb_number_model.fit(X_xgb, y_xgb)
                    joblib.dump(self.xgb_number_model, os.path.join(self.config['model_dir'], 'xgb_number_model.pkl'))
                print("✅ Updated number models")
            
            # Update color model
            X_col, y_col = self.preprocess_data(recent_data, 'color')
            if X_col.size > 0 and y_col.size > 0 and self.color_model is not None:
                self.color_model.fit(X_col, y_col, epochs=2, batch_size=4, verbose=0)
                self.color_model.save(os.path.join(self.config['model_dir'], 'color_model.keras'))
                X_xgb = X_col.reshape(X_col.shape[0], -1)
                y_xgb = np.argmax(y_col, axis=1)
                if self.xgb_color_model is not None:
                    self.xgb_color_model.fit(X_xgb, y_xgb)
                    joblib.dump(self.xgb_color_model, os.path.join(self.config['model_dir'], 'xgb_color_model.pkl'))
                print("✅ Updated color models")
            
        except Exception as e:
            print(f"⚠️ Online learning failed: {str(e)}")

    def _generate_real_time_prediction(self):
        """Generate and display prediction with market details and expected value"""
        try:
            number_sequence = list(self.number_sequence)[-self.config['seq_length']:]
            color_sequence = list(self.color_sequence)[-self.config['seq_length']:]
            
            print(f"DEBUG: Predicting with number sequence: {number_sequence}")
            print(f"DEBUG: Predicting with color sequence: {color_sequence}")
            
            number_pred = self.predict_next(number_sequence, 'number')
            color_pred = self.predict_next(color_sequence, 'color')
            
            if not number_pred or not color_pred:
                print("⚠️ No predictions generated")
                return
                
            best_number, number_data = max(number_pred.items(), key=lambda x: x[1]['probability'])
            best_color, color_data = max(color_pred.items(), key=lambda x: x[1]['probability'])
            
            print(f"\n🕒 {datetime.now().strftime('%H:%M:%S')} Prediction Update")
            print(f"📊 Last Number Sequence: {' → '.join(number_sequence)}")
            print(f"📊 Last Color Sequence: {' → '.join(color_sequence)}")
            
            print("\nNumber Predictions:")
            for number, data in sorted(number_pred.items(), key=lambda x: x[1]['probability'], reverse=True):
                if data['probability'] > 0.05:
                    price_info = f"@ {data['back_price']}" if data['back_price'] else "(no price)"
                    ev_info = f"EV: {data['expected_value']:.2f}" if data['expected_value'] >= 0 else "EV: N/A"
                    print(f"🔢 {number}: {data['probability']:.1%} {price_info} | {ev_info} | ID: {data['runner_id']}")
            
            print("\nColor Predictions:")
            for color, data in sorted(color_pred.items(), key=lambda x: x[1]['probability'], reverse=True):
                if data['probability'] > 0.1:
                    price_info = f"@ {data['back_price']}" if data['back_price'] else "(no price)"
                    ev_info = f"EV: {data['expected_value']:.2f}" if data['expected_value'] >= 0 else "EV: N/A"
                    print(f"🎨 {color}: {data['probability']:.1%} {price_info} | {ev_info} | ID: {data['runner_id']}")
            
            print(f"\n💡 Recommended Bet: Number {best_number} ({number_data['probability']:.1%}, EV: {number_data['expected_value']:.2f}) | Color {best_color} ({color_data['probability']:.1%}, EV: {color_data['expected_value']:.2f})")
            print("━" * 50)
            
        except Exception as e:
            print(f"⚠️ Prediction display error: {str(e)}")
            import traceback
            traceback.print_exc()

    def stop_real_time_prediction(self):
        """Stop prediction service"""
        self._running = False
        if self._update_thread:
            self._update_thread.join()
        print("🛑 Prediction service stopped")