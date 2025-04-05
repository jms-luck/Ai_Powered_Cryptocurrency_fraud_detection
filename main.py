# Import Intel-optimized libraries
import pandas as pd
import numpy as np
from sklearnex import patch_sklearn  # Intel Extension for Scikit-learn
patch_sklearn()  # Patch scikit-learn to use Intel optimizations

import random
import time
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
import joblib
from web3 import Web3
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
import dash
from dash import Dash, dcc, html, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import networkx as nx
import matplotlib.pyplot as plt
import socket
import signal
import sys
import logging
import os
import io
import base64
from PIL import Image
import google.generativeai as genai

# Setup logging for tracking anomalies
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

API_KEY = ""
genai.configure(api_key=API_KEY)

# Initialize Intel oneAPI components
from daal4py.linear_model import LinearRegression as DAALLinearRegression
from daal4py.sklearn.ensemble import RandomForestClassifier as DAALRandomForestClassifier

# Blockchain Configuration
BLOCKCHAIN_API_KEY = "e65ee1b207274346b6a586c24e43bb18"
BLOCKCHAIN_URL = f"https://mainnet.infura.io/v3/{BLOCKCHAIN_API_KEY}"
w3 = Web3(Web3.HTTPProvider(BLOCKCHAIN_URL))

class BlockchainLedger:
    def __init__(self):
        self.verified_addresses = set([
            '0x742d35Cc6634C0532925a3b844Bc454e4438f44e', 
            '0x1aD91ee08f21bE3dE0BA2ba6918E714dA6B45836'
        ])
        self.w3 = w3
        
    def is_connected(self):
        return True

    def verify_address(self, address):
        return Web3.is_address(address) and address in self.verified_addresses

    def is_verified(self, address):
        return self.verify_address(address)

ledger = BlockchainLedger()

# Data Storage and Blocked Accounts
transaction_history = []
alerts = []
blocked_accounts = set()
anomalies_detected = []
llm_insights = ""

# Function to automatically block/freeze accounts
def block_account(address):
    global blocked_accounts
    if address not in blocked_accounts:
        blocked_accounts.add(address)
        logger.warning(f"Account {address} has been automatically blocked/frozen.")

# Simulate Real-Time Transaction with Intel-optimized random number generation
def simulate_real_time_transaction():
    # Using numpy's random which is optimized by Intel oneMKL
    address = np.random.choice([
        Web3.to_checksum_address(f"0x{np.random.bytes(20).hex()}"),
        np.random.choice([
            '0x742d35Cc6634C0532925a3b844Bc454e4438f44e', 
            '0x1aD91ee08f21bE3dE0BA2ba6918E714dA6B45836'
        ])
    ])
    
    transaction = {
        'Sent tnx': np.random.randint(1, 10),
        'Received Tnx': np.random.randint(1, 10),
        'avg val sent': np.random.uniform(0.01, 5.0),
        'avg val received': np.random.uniform(0.01, 5.0),
        'Unique Sent To Addresses': np.random.randint(1, 20),
        'Unique Received From Addresses': np.random.randint(1, 20),
        'Time Diff between first and last (Mins)': np.random.uniform(1, 120),
        'Timestamp': datetime.now(),
        'Account Age': np.random.randint(1, 100),
        'Device Fingerprint': np.random.choice(['DeviceA', 'DeviceB', 'DeviceC']),
        'IP Address': np.random.choice(['192.168.0.1', '192.168.0.2']),
        'Geolocation': np.random.choice(['US', 'EU', 'Asia']),
        'Address': address
    }
    transaction_history.append(transaction)
    return transaction

# Feature Engineering Functions with Intel optimizations
def feature_engineering_real_time_fake(transaction_data, previous_data=None):
    time_diff = (transaction_data['Timestamp'] - previous_data['Timestamp']).total_seconds() / 60.0 if previous_data else 0
    return pd.DataFrame([{
        'Transaction Frequency': transaction_data['Sent tnx'] + transaction_data['Received Tnx'],
        'Average Sent Amount': transaction_data['avg val sent'],
        'Average Received Amount': transaction_data['avg val received'],
        'Unique Sent Addresses': transaction_data['Unique Sent To Addresses'],
        'Unique Received Addresses': transaction_data['Unique Received From Addresses'],
        'Transaction Time Consistency': transaction_data['Time Diff between first and last (Mins)'],
        'Time Diff between Transactions (Minutes)': time_diff
    }])

def feature_engineering_real_time_sybil(transaction_data, previous_data=None):
    time_diff = (transaction_data['Timestamp'] - previous_data['Timestamp']).total_seconds() / 60.0 if previous_data else 0
    return pd.DataFrame([{
        'Transaction Frequency': transaction_data['Sent tnx'] + transaction_data['Received Tnx'],
        'Average Sent Amount': transaction_data['avg val sent'],
        'Average Received Amount': transaction_data['avg val received'],
        'Unique Sent Addresses': transaction_data['Unique Sent To Addresses'],
        'Unique Received Addresses': transaction_data['Unique Received From Addresses'],
        'Transaction Time Consistency': transaction_data['Time Diff between first and last (Mins)'],
        'Time Diff between Transactions (Minutes)': time_diff,
        'Account Age': transaction_data['Account Age'],
        'Total Sent Transactions': transaction_data['Sent tnx'],
        'Total Received Transactions': transaction_data['Received Tnx'],
        'Device Fingerprint': transaction_data['Device Fingerprint'],
        'IP Address': transaction_data['IP Address'],
        'Geolocation': transaction_data['Geolocation']
    }])

def feature_engineering_real_time_bot(transaction_data, previous_data=None):
    time_diff = (transaction_data['Timestamp'] - previous_data['Timestamp']).total_seconds() / 60.0 if previous_data else 0
    # Using numpy's std which is optimized by Intel oneMKL
    variance = np.std([transaction_data['avg val sent'], transaction_data['avg val received']])
    frequency = transaction_data['Sent tnx'] + transaction_data['Received Tnx']
    return pd.DataFrame([{
        'Transaction Time Diff': transaction_data['Time Diff between first and last (Mins)'],
        'Transaction Amount Variance': variance,
        'Unique Sent Addresses': transaction_data['Unique Sent To Addresses'],
        'Bot Activity Indicator': frequency * variance * transaction_data['Unique Sent To Addresses']
    }])

# Load models using Intel-optimized scikit-learn
try:
    pipeline_fake = joblib.load('fake_identities_model.pkl')
    pipeline_sybil = joblib.load('sybil_attacks_model.pkl')
    pipeline_bot = joblib.load('bot_activity_model.pkl')
except:
    logger.warning("Could not load pre-trained models. Training new ones...")
    # Fallback to training new models with Intel optimizations
    from sklearnex.ensemble import RandomForestClassifier as IntelexRandomForest
    pipeline_fake = Pipeline([
        ('classifier', IntelexRandomForest(n_estimators=100))
    ])
    pipeline_sybil = Pipeline([
        ('classifier', IntelexRandomForest(n_estimators=100))
    ])
    pipeline_bot = Pipeline([
        ('classifier', IntelexRandomForest(n_estimators=100))
    ])

# Alert Detection Logic with Automatic Blocking
def detect_alerts(transaction_data, previous_data=None):
    global alerts, blocked_accounts
    current_alerts = []
    address = transaction_data['Address']
    
    if address in blocked_accounts:
        current_alerts.append(f"Account {address} is already blocked.")
        alerts.extend(current_alerts)
        return

    df = pd.DataFrame(transaction_history)
    
    # Using Intel-optimized pandas operations
    if len(df) > 1:
        similar_ip = df[df['IP Address'] == transaction_data['IP Address']]
        similar_device = df[df['Device Fingerprint'] == transaction_data['Device Fingerprint']]
        if len(similar_ip) > 3 or len(similar_device) > 3:
            current_alerts.append(f"Sybil Attack detected - Account {address} has been automatically blocked.")
            block_account(address)
    
    features_fake = feature_engineering_real_time_fake(transaction_data, previous_data)
    features_sybil = feature_engineering_real_time_sybil(transaction_data, previous_data)
    features_bot = feature_engineering_real_time_bot(transaction_data, previous_data)
    
    freq = features_fake['Transaction Frequency'].iloc[0]
    avg_sent = features_fake['Average Sent Amount'].iloc[0]
    time_diff = features_fake['Time Diff between Transactions (Minutes)'].iloc[0]
    if freq > 5 and avg_sent < 1.0 and (time_diff < 5 or time_diff == 0):
        current_alerts.append(f"Fake Identity detected for account {address}. Investigation recommended.")
    
    bot_indicator = features_bot['Bot Activity Indicator'].iloc[0]
    if bot_indicator > 50:
        current_alerts.append(f"Bot Activity detected - Account {address} has been automatically blocked.")
        block_account(address)
    
    if pipeline_fake and pipeline_sybil and pipeline_bot:
        if pipeline_fake.predict(features_fake)[0] == 1:
            current_alerts.append(f"Fake Identity (Model): Investigation recommended for account {address}.")
        if pipeline_sybil.predict(features_sybil)[0] == 1:
            current_alerts.append(f"Sybil Attack (Model) detected - Account {address} has been automatically blocked.")
            block_account(address)
        if pipeline_bot.predict(features_bot)[0] == 1:
            current_alerts.append(f"Bot Activity (Model) detected - Account {address} has been automatically blocked.")
            block_account(address)
    
    if not ledger.is_verified(address):
        current_alerts.append(f"Unverified Address: Blockchain check failed for account {address}. Follow-up action recommended.")
    
    if current_alerts:
        alerts.extend(current_alerts)

# Graph-based Anomaly Detection with Intel optimizations
def create_wallet_transactions():
    transactions = []
    for tx in transaction_history:
        src = tx['Address']
        for _ in range(min(tx['Sent tnx'], 3)):
            dst = f"wallet_{np.random.randint(1000, 9999)}"
            amount = tx['avg val sent']
            transactions.append((src, dst, amount))
    wallets = list(set([t[0] for t in transactions] + [t[1] for t in transactions]))
    return wallets, transactions

def build_wallet_graph(transactions):
    G = nx.DiGraph()
    for src, dst, amount in transactions:
        if G.has_edge(src, dst):
            G[src][dst]['weight'] += amount
        else:
            G.add_edge(src, dst, weight=amount)
    return G

def extract_features(G):
    features = {}
    for node in G.nodes():
        in_deg = G.in_degree(node)
        out_deg = G.out_degree(node)
        in_amount = sum(G[u][node]['weight'] for u in G.predecessors(node)) if in_deg > 0 else 0
        out_amount = sum(G[node][v]['weight'] for v in G.successors(node)) if out_deg > 0 else 0
        features[node] = [in_deg, out_deg, in_amount, out_amount]
    return features

def detect_graph_anomalies(features, contamination=0.1):
    global anomalies_detected
    nodes = list(features.keys())
    data = np.array(list(features.values()))
    
    if len(data) == 0:
        logger.warning("No data to analyze for anomalies")
        return []
    
    if np.std(data, axis=0).min() > 0:
        data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    
    # Using Intel-optimized Isolation Forest
    from sklearnex.ensemble import IsolationForest as IntelexIsolationForest
    clf = IntelexIsolationForest(random_state=42, contamination=contamination)
    clf.fit(data)
    preds = clf.predict(data)
    anomalies = [nodes[i] for i, pred in enumerate(preds) if pred == -1]
    
    for addr in anomalies:
        if addr not in anomalies_detected:
            anomalies_detected.append(addr)
            alerts.append(f"Graph Anomaly detected for account {addr}. Automatically blocked.")
            block_account(addr)
    
    return anomalies

def visualize_graph(G, anomalies):
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    
    normal_nodes = [node for node in G.nodes() if node not in anomalies]
    nx.draw_networkx_nodes(G, pos, nodelist=normal_nodes, node_color='green', alpha=0.8, node_size=300)
    
    if anomalies:
        nx.draw_networkx_nodes(G, pos, nodelist=anomalies, node_color='red', alpha=0.8, node_size=300)
    
    edge_widths = [min(G[u][v]['weight'] * 0.5, 3) for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, arrows=True)
    
    if len(G.nodes()) < 30:
        nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title("Wallet Network with Detected Anomalies")
    plt.axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    
    graph_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{graph_base64}"

def get_llm_insights(alerts_list, anomalies_list):
    global llm_insights
    
    try:
        prompt = "You are a blockchain security analyst. Analyze the following alerts and detected anomalies, and provide insights and recommendations.\n\n"
        prompt += "Alerts:\n" + "\n".join(str(a) for a in alerts_list) + "\n\n"
        prompt += "Detected Anomalies (Blocked Accounts): " + ", ".join(str(a) for a in anomalies_list) + "\n\n"
        prompt += "Provide a brief summary and recommendations:"
        
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        llm_insights = response.text
        return llm_insights
    except Exception as e:
        logger.error(f"Error getting LLM insights: {e}")
        return f"Error getting AI insights: {str(e)}"

# Dash Application Setup (unchanged)
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    # ... (same layout code as before)
])

@app.callback(
    [Output('heatmap', 'figure'),
     Output('network-graph', 'figure'),
     Output('time-series', 'figure'),
     Output('alerts-list', 'children'),
     Output('blocked-accounts', 'children'),
     Output('graph-visualization', 'src')],
    [Input('interval-component', 'n_intervals')]
)
def update_dashboard(n):
    global transaction_history, alerts, blocked_accounts, anomalies_detected
    previous_data = transaction_history[-1] if transaction_history else None
    
    if n > 0:
        transaction_data = simulate_real_time_transaction()
        detect_alerts(transaction_data, previous_data)
        
        if n % 5 == 0 and len(transaction_history) > 10:
            wallets, transactions = create_wallet_transactions()
            G = build_wallet_graph(transactions)
            features = extract_features(G)
            anomalies = detect_graph_anomalies(features)
            graph_img = visualize_graph(G, anomalies)
        else:
            graph_img = "" if n == 0 else dash.no_update

    df = pd.DataFrame(transaction_history)
    
    if len(df) == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="No data yet")
        return empty_fig, empty_fig, empty_fig, "No alerts yet", "No blocked accounts", ""

    # Heatmap using Intel-optimized pandas
    heatmap_data = df.groupby('Geolocation').agg({
        'Sent tnx': 'sum',
        'Received Tnx': 'sum'
    }).reset_index()
    heatmap_data['Total Tnx'] = heatmap_data['Sent tnx'] + heatmap_data['Received Tnx']
    heatmap_fig = px.density_heatmap(
        heatmap_data, 
        x='Geolocation', 
        y='Total Tnx', 
        title='Transaction Activity Heatmap',
        color_continuous_scale='Viridis'
    )
    heatmap_fig.update_layout(height=300)

    # Network graph
    G = nx.Graph()
    for i, row in df.iterrows():
        G.add_node(row['Address'])
        if i > 0 and np.random.random() < 0.3:
            G.add_edge(df.iloc[i-1]['Address'], row['Address'])
    
    pos = nx.spring_layout(G)
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    node_x, node_y = [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
    
    network_fig = go.Figure()
    network_fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.5, color='gray')))
    node_colors = ['red' if node in blocked_accounts else 'blue' for node in G.nodes()]
    network_fig.add_trace(go.Scatter(
        x=node_x, y=node_y, 
        mode='markers', 
        marker=dict(size=10, color=node_colors),
        text=list(G.nodes()),
        hoverinfo='text'
    ))
    network_fig.update_layout(title='User Interaction Network', showlegend=False, height=300)

    # Time series with Intel-optimized numpy
    df['Total Tnx'] = df['Sent tnx'] + df['Received Tnx']
    time_series_fig = go.Figure()
    time_series_fig.add_trace(go.Scatter(
        x=df['Timestamp'], 
        y=df['Total Tnx'], 
        mode='lines+markers', 
        name='Transaction Frequency'
    ))
    time_series_fig.add_trace(go.Scatter(
        x=df['Timestamp'], 
        y=df['avg val sent'], 
        mode='lines+markers', 
        name='Avg Sent Amount',
        yaxis='y2'
    ))
    time_series_fig.update_layout(
        title='Transaction Metrics Over Time',
        yaxis=dict(title='Transaction Frequency'),
        yaxis2=dict(title='Amount (ETH)', overlaying='y', side='right'),
        height=300
    )

    alert_items = [html.P(alert, className="alert alert-danger") for alert in alerts[-10:]]
    blocked_list = [html.P(acc, className="alert alert-dark") for acc in sorted(blocked_accounts)]
    
    if n == 0 or len(transaction_history) < 5:
        graph_img = ""
    elif n % 5 == 0 or graph_img == "":
        wallets, transactions = create_wallet_transactions()
        G = build_wallet_graph(transactions)
        features = extract_features(G)
        anomalies = detect_graph_anomalies(features)
        graph_img = visualize_graph(G, anomalies)
    
    return heatmap_fig, network_fig, time_series_fig, alert_items, blocked_list, graph_img

@app.callback(
    Output('ai-insights', 'children'),
    [Input('run-ai-analysis', 'n_clicks')],
    [State('alerts-list', 'children')]
)
def update_ai_insights(n_clicks, alert_list):
    if n_clicks is None:
        return html.P("Click 'Run AI Analysis' to get security insights from the LLM.")
    
    alerts_text = [a['props']['children'] for a in alert_list] if alert_list else []
    anomalies_text = list(blocked_accounts)
    
    insights = get_llm_insights(alerts_text, anomalies_text)
    insight_paragraphs = [html.P(p) for p in insights.split('\n\n')]
    
    return insight_paragraphs

# Initialize some transactions for the first load
for _ in range(10):
    tx = simulate_real_time_transaction()
    previous = transaction_history[-2] if len(transaction_history) >= 2 else None
    detect_alerts(tx, previous)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        manual_port = None  
        port = manual_port if manual_port else find_free_port(start_port=8050)
        print(f"Starting Dash server on port {port}")
        print("Visit http://127.0.0.1:{port} to view the dashboard")
        app.run_server(debug=True, port=port, use_reloader=False)
    except OSError as e:
        print(f"Failed to start server: {e}")
        print("Please try one of the following:")
        print("1. Run the script again to try a different port.")
        print("2. Manually specify a port by setting 'manual_port' (e.g., 8051).")
        print("3. Stop existing Dash servers using the steps below:")
        print("   - Windows: Task Manager -> End 'python.exe' processes")
        print("   - Linux/Mac: 'lsof -i :8050' then 'kill -9 <PID>'")
