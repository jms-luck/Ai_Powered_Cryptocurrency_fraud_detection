# Import Intel-optimized libraries
from sklearnex import patch_sklearn
patch_sklearn()  # Patch scikit-learn to use Intel optimizations

import pandas as pd
import numpy as np
import random
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

# Load data
file_path = "transaction_dataset.csv"
df = pd.read_csv(file_path)

# Drop irrelevant columns
df = df.drop(columns=['Unnamed: 0', 'Index', 'Address'])

## Feature Engineering on Historical Data with Intel optimizations
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

def feature_engineering(df):
    """
    Generate new features from the dataset with Intel-optimized operations
    """
    # Ensure all expected columns are present; fill missing ones with 0
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0

    # Transaction Frequency (using Intel-optimized numpy operations)
    df['Transaction Frequency'] = df['Sent tnx'].values + df['Received Tnx'].values
    
    # Average Sent and Received Amounts
    df['Average Sent Amount'] = df['avg val sent']
    df['Average Received Amount'] = df['avg val received']
    
    # Diversity of Interactions
    if 'Unique Sent To Addresses' in df.columns:
        df['Unique Sent Addresses'] = df['Unique Sent To Addresses']
    elif 'Unique Sent Addresses' not in df.columns:
        df['Unique Sent Addresses'] = 0

    if 'Unique Received From Addresses' in df.columns:
        df['Unique Received Addresses'] = df['Unique Received From Addresses']
    elif 'Unique Received Addresses' not in df.columns:
        df['Unique Received Addresses'] = 0

    # Transaction Time Consistency
    if 'Time Diff between first and last (Mins)' in df.columns:
        df['Transaction Time Consistency'] = df['Time Diff between first and last (Mins)']
    
    return df

df = feature_engineering(df)
y = df['FLAG']
X = df.drop(columns=['FLAG'])

# Save training feature column order (this is what the pipeline expects)
training_columns = list(X.columns)

# Identify numerical and categorical columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

## Split Data and Build Pipeline with Intel optimizations
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Preprocessing pipelines with Intel optimizations
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())  # Uses Intel MKL acceleration
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # Intel-optimized
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# Machine Learning Pipeline with Intel-optimized Random Forest
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))  # Uses Intel Extension for Scikit-learn
])

# Train the model with Intel optimizations
print("Training model with Intel optimizations...")
start_time = time.time()
pipeline.fit(X_train, y_train)
training_time = time.time() - start_time
print(f"Model training completed in {training_time:.2f} seconds")

# Evaluate the model
y_pred = pipeline.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print("\nTest Accuracy: {:.4f}".format(test_accuracy))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

## Real-Time Prediction Simulation with Intel optimizations
def align_features(df_sim):
    """
    Ensure that the simulation DataFrame has exactly the same columns (and order)
    as the training data. Missing columns are filled with default value 0.
    """
    for col in training_columns:
        if col not in df_sim.columns:
            df_sim[col] = 0
    # Reorder the columns to match training data
    return df_sim[training_columns]

def simulate_real_time_transaction():
    """
    Simulate a single real-time transaction using Intel-optimized random numbers
    """
    transaction_data = {
        'Sent tnx': np.random.randint(1, 10),
        'Received Tnx': np.random.randint(1, 10),
        'avg val sent': np.random.uniform(0.01, 5.0),
        'avg val received': np.random.uniform(0.01, 5.0),
        'Unique Sent To Addresses': np.random.randint(1, 20),
        'Unique Received From Addresses': np.random.randint(1, 20),
        'Time Diff between first and last (Mins)': np.random.uniform(1, 120),
        'Timestamp': datetime.now(),
    }
    return transaction_data

def feature_engineering_for_real_time(transaction_data, previous_data=None):
    """
    Generate features for real-time transaction data with Intel optimizations
    """
    # Calculate basic features using numpy operations (Intel-optimized)
    transaction_frequency = transaction_data['Sent tnx'] + transaction_data['Received Tnx']
    average_sent_amount = transaction_data['avg val sent']
    average_received_amount = transaction_data['avg val received']
    unique_sent_addresses = transaction_data['Unique Sent To Addresses']
    unique_received_addresses = transaction_data['Unique Received From Addresses']
    time_diff_first_last = transaction_data['Time Diff between first and last (Mins)']
    
    # Calculate time difference between transactions
    if previous_data is not None:
        previous_time = previous_data['Timestamp']
        time_diff = (transaction_data['Timestamp'] - previous_time).total_seconds() / 60.0
    else:
        time_diff = 0

    # Build feature dictionary
    features = {
        'Transaction Frequency': transaction_frequency,
        'Average Sent Amount': average_sent_amount,
        'Average Received Amount': average_received_amount,
        'Unique Sent Addresses': unique_sent_addresses,
        'Unique Received Addresses': unique_received_addresses,
        'Transaction Time Consistency': time_diff_first_last,
        'Time Diff between Transactions (Minutes)': time_diff,
        'total Ether sent': 0,
        'total ether received': 0,
        'Number of Created Contracts': 0,
    }
    
    # Create DataFrame and perform feature engineering
    features_df = pd.DataFrame([features])
    features_df = feature_engineering(features_df)
    return align_features(features_df)

def real_time_monitoring(num_transactions=10, delay=2):
    """
    Simulate real-time monitoring with Intel-optimized predictions
    """
    previous_data = None
    print("\n--- Starting Real-Time Transaction Monitoring Simulation (Intel-optimized) ---")
    
    for i in range(num_transactions):
        transaction_data = simulate_real_time_transaction()
        transaction_df = feature_engineering_for_real_time(transaction_data, previous_data)
        
        # Predict fraud probability using the trained model
        start_pred_time = time.time()
        fraud_prob = pipeline.predict_proba(transaction_df)[0, 1]
        prediction = pipeline.predict(transaction_df)[0]
        pred_time = time.time() - start_pred_time
        
        print(f"\nTransaction {i + 1}:")
        print(f"Predicted Fraud Probability: {fraud_prob:.4f}")
        print(f"Prediction Time: {pred_time:.6f} seconds")
        
        if prediction == 1:
            print("⚠️ Fraudulent Transaction Detected! Stopping transaction.")
        else:
            print("Transaction appears legitimate. Proceeding with processing.")
        
        previous_data = transaction_data
        time.sleep(delay)
    
    print("\n--- Simulation Completed ---")

# Run the optimized real-time monitoring simulation
real_time_monitoring(num_transactions=10, delay=2)

# Save the optimized model
joblib.dump(pipeline, 'fraud_detection_model_intel.pkl')
joblib.dump(X.columns.tolist(), 'feature_columns_intel.pkl')
