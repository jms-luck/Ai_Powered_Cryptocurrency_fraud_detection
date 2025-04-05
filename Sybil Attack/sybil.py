# Import Intel-optimized libraries
from sklearnex import patch_sklearn
patch_sklearn()  

import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load data with Intel-optimized pandas
file_path = "transaction_dataset.csv"
df = pd.read_csv(file_path)

# Drop irrelevant columns
df = df.drop(columns=['Unnamed: 0', 'Index', 'Address'])

# Define expected columns
expected_columns = [
    'Transaction Frequency', 'Average Sent Amount', 'Average Received Amount',
    'Unique Sent Addresses', 'Unique Received Addresses', 'Transaction Time Consistency',
    'Time Diff between Transactions (Minutes)', 'Account Age', 'Total Sent Transactions',
    'Total Received Transactions', 'Device Fingerprint', 'IP Address', 'Geolocation'
]

# Intel-optimized feature engineering
def feature_engineering(df):
    """Vectorized feature engineering using Intel-optimized numpy"""
    # Initialize with zeros (Intel-optimized)
    for col in expected_columns:
        if col not in df.columns:
            if col in ['Device Fingerprint', 'IP Address', 'Geolocation']:
                df[col] = 'Unknown'
            else:
                df[col] = np.zeros(len(df), dtype=np.float32)
    
    # Vectorized calculations
    if 'Sent tnx' in df.columns and 'Received Tnx' in df.columns:
        df['Transaction Frequency'] = df['Sent tnx'].values + df['Received Tnx'].values
        df['Total Sent Transactions'] = df['Sent tnx'].values
        df['Total Received Transactions'] = df['Received Tnx'].values
    
    if 'avg val sent' in df.columns:
        df['Average Sent Amount'] = df['avg val sent'].values
        
    if 'avg val received' in df.columns:
        df['Average Received Amount'] = df['avg val received'].values
    
    if 'Unique Sent To Addresses' in df.columns:
        df['Unique Sent Addresses'] = df['Unique Sent To Addresses'].values
        
    if 'Unique Received From Addresses' in df.columns:
        df['Unique Received Addresses'] = df['Unique Received From Addresses'].values
    
    if 'Time Diff between first and last (Mins)' in df.columns:
        df['Transaction Time Consistency'] = df['Time Diff between first and last (Mins)'].values
    
    return df

# Apply feature engineering
df = feature_engineering(df)
y = df['FLAG']
X = df[expected_columns]

# Identify column types
numerical_cols = [col for col in expected_columns if X[col].dtype in ['int64', 'float64']]
categorical_cols = [col for col in expected_columns if X[col].dtype == 'object']

# Intel-optimized preprocessing
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Intel-optimized
    ('scaler', StandardScaler())  # Uses Intel MKL
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Intel-optimized
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # Intel-optimized
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# Intel-optimized Random Forest with parallel processing
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        random_state=42,
        n_estimators=100,
        n_jobs=-1,  # Use all available cores
        max_depth=30,  # Deeper trees for better accuracy
        min_samples_split=5
    ))
])

# Train with Intel optimizations
print("Training with Intel oneAPI optimizations...")
start_time = time.time()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
pipeline.fit(X_train, y_train)
print(f"Training completed in {time.time() - start_time:.2f} seconds")

# Evaluate
y_pred = pipeline.predict(X_test)
print("\nTest Accuracy: {:.4f}".format(accuracy_score(y_test, y_pred)))
print("\nClassification Report:\n", classification_report(y_test, y_pred)))

# Real-Time Simulation with Intel optimizations
def simulate_real_time_transaction():
    """Use Intel-optimized numpy random number generation"""
    return {
        'Sent tnx': np.random.randint(1, 10),
        'Received Tnx': np.random.randint(1, 10),
        'avg val sent': np.random.uniform(0.01, 5.0),
        'avg val received': np.random.uniform(0.01, 5.0),
        'Unique Sent To Addresses': np.random.randint(1, 20),
        'Unique Received From Addresses': np.random.randint(1, 20),
        'Time Diff between first and last (Mins)': np.random.uniform(1, 120),
        'Timestamp': datetime.now(),
        'Device Fingerprint': np.random.choice(['DeviceA', 'DeviceB', 'DeviceC']),
        'IP Address': np.random.choice(['192.168.0.1', '192.168.0.2']),
        'Geolocation': np.random.choice(['US', 'EU', 'Asia']),
        'Account Age': np.random.randint(1, 100)
    }

def real_time_monitoring(num_transactions=10, delay=2):
    previous_data = None
    print("\n--- Intel-optimized Sybil Attack Detection ---")
    
    for i in range(num_transactions):
        # Generate transaction
        transaction_data = simulate_real_time_transaction()
        
        # Feature engineering
        features = {
            'Transaction Frequency': transaction_data['Sent tnx'] + transaction_data['Received Tnx'],
            'Average Sent Amount': transaction_data['avg val sent'],
            'Average Received Amount': transaction_data['avg val received'],
            'Unique Sent Addresses': transaction_data['Unique Sent To Addresses'],
            'Unique Received Addresses': transaction_data['Unique Received From Addresses'],
            'Transaction Time Consistency': transaction_data['Time Diff between first and last (Mins)'],
            'Time Diff between Transactions (Minutes)': (
                (transaction_data['Timestamp'] - previous_data['Timestamp']).total_seconds() / 60.0 
                if previous_data else 0.0
            ),
            'Account Age': transaction_data['Account Age'],
            'Total Sent Transactions': transaction_data['Sent tnx'],
            'Total Received Transactions': transaction_data['Received Tnx'],
            'Device Fingerprint': transaction_data['Device Fingerprint'],
            'IP Address': transaction_data['IP Address'],
            'Geolocation': transaction_data['Geolocation']
        }
        
        # Create aligned DataFrame
        features_df = pd.DataFrame([features])
        for col in expected_columns:
            if col not in features_df.columns:
                if col in numerical_cols:
                    features_df[col] = 0.0
                else:
                    features_df[col] = 'Unknown'
        features_df = features_df[expected_columns]
        
        # Intel-optimized prediction
        start_time = time.time()
        sybil_prob = pipeline.predict_proba(features_df)[0, 1]
        pred_time = time.time() - start_time
        
        print(f"\nTransaction {i+1}:")
        print(f"Sybil Probability: {sybil_prob:.4f} (Predicted in {pred_time:.6f}s)")
        print("ALERT! ⚠️" if pipeline.predict(features_df)[0] == 1 else "Status: Normal")
        
        previous_data = transaction_data
        time.sleep(delay)

# Run optimized simulation
real_time_monitoring()

# Save optimized model and features
joblib.dump(pipeline, 'sybil_detection_model_intel.pkl')
joblib.dump(expected_columns, 'feature_columns_intel.pkl')
print("Intel-optimized model saved successfully!")
