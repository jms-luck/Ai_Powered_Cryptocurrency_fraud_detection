# Import Intel-optimized libraries
from sklearnex import patch_sklearn
patch_sklearn()  # This must be called before importing scikit-learn components

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

# Feature Engineering with Intel-optimized numpy
def feature_engineering(df):
    expected_columns = [
        'Transaction Frequency', 'Average Sent Amount', 'Average Received Amount', 
        'Unique Sent Addresses', 'Unique Received Addresses', 'Transaction Time Consistency', 
        'Time Diff between Transactions (Minutes)', 'total Ether sent', 'total ether received', 
        'Number of Created Contracts', 'Received Tnx', 'min value received', 'avg val sent', 
        'max val sent', 'avg val received', 'max value received ', 'Sent tnx', 'Avg min between sent tnx',
        'total ether balance', 'Total ERC20 tnxs', 'ERC20 avg time between sent tnx', 'ERC20 uniq sent addr', 
        'ERC20 min val sent', 'ERC20 max val sent', 'ERC20 avg val sent', 'ERC20 avg time between rec tnx', 
        'ERC20 most sent token type', 'ERC20 total Ether sent contract', 'ERC20 uniq sent token name'
    ]
    
    # Initialize missing columns with zeros (Intel-optimized operations)
    for col in expected_columns:
        if col not in df.columns:
            df[col] = np.zeros(len(df), dtype=np.float64)  # Intel-optimized zeros

    # Vectorized operations using Intel-optimized numpy
    df['Transaction Frequency'] = df['Sent tnx'].values + df['Received Tnx'].values
    df['Average Sent Amount'] = df['avg val sent'].values
    df['Average Received Amount'] = df['avg val received'].values
    
    # Conditional column assignments
    if 'Unique Sent To Addresses' in df.columns:
        df['Unique Sent Addresses'] = df['Unique Sent To Addresses'].values
    else:
        df['Unique Sent Addresses'] = np.zeros(len(df), dtype=np.int64)

    if 'Unique Received From Addresses' in df.columns:
        df['Unique Received Addresses'] = df['Unique Received From Addresses'].values
    else:
        df['Unique Received Addresses'] = np.zeros(len(df), dtype=np.int64)

    if 'Time Diff between first and last (Mins)' in df.columns:
        df['Transaction Time Consistency'] = df['Time Diff between first and last (Mins)'].values
        
    return df

df = feature_engineering(df)
y = df['FLAG']
X = df.drop(columns=['FLAG'])

# Save training feature column order
training_columns = list(X.columns)

# Identify numerical and categorical columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# Split data with Intel-optimized operations
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Intel-optimized preprocessing pipelines
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

# Intel-optimized Random Forest
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        random_state=42,
        n_estimators=100,
        n_jobs=-1  # Enable parallel processing with Intel optimizations
    ))
])

# Train with Intel optimizations
print("Training with Intel oneAPI optimizations...")
start_time = time.time()
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
    }

def real_time_monitoring(num_transactions=10, delay=2):
    previous_data = None
    print("\n--- Intel-optimized Real-Time Monitoring ---")
    
    for i in range(num_transactions):
        transaction_data = simulate_real_time_transaction()
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
            'total Ether sent': 0.0,
            'total ether received': 0.0,
            'Number of Created Contracts': 0
        }
        
        # Create aligned DataFrame
        features_df = pd.DataFrame([features])
        for col in training_columns:
            if col not in features_df.columns:
                features_df[col] = 0.0
        features_df = features_df[training_columns]
        
        # Intel-optimized prediction
        start_time = time.time()
        fraud_prob = pipeline.predict_proba(features_df)[0, 1]
        pred_time = time.time() - start_time
        
        print(f"\nTransaction {i+1}:")
        print(f"Fraud Probability: {fraud_prob:.4f} (Predicted in {pred_time:.6f}s)")
        print("ALERT! ⚠️" if pipeline.predict(features_df)[0] == 1 else "Status: Normal")
        
        previous_data = transaction_data
        time.sleep(delay)

# Run optimized simulation
real_time_monitoring()

# Save optimized model
joblib.dump(pipeline, 'fraud_detection_model_intel.pkl')
joblib.dump(training_columns, 'feature_columns_intel.pkl')
