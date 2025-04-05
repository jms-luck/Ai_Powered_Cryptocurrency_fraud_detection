# Blockchain Transaction Monitoring System with Intel oneAPI Optimization

## Overview

This application provides real-time monitoring and anomaly detection for blockchain transactions, leveraging Intel oneAPI toolkit for optimized performance. The system detects various types of suspicious activities including fake identities, Sybil attacks, and bot activity.

## Key Features

- **Real-time transaction monitoring** with simulated blockchain data
- **Anomaly detection** using machine learning models
- **Graph-based analysis** of wallet interactions
- **Automated account blocking** for suspicious activity
- **AI-powered insights** from Google Gemini
- **Intel-optimized performance** for faster processing

## Technology Stack

- **Intel oneAPI Components**:
  - Intel® Extension for Scikit-learn
  - Intel® Distribution for Python
  - Intel® oneAPI Data Analytics Library (oneDAL)
  - Intel® oneAPI Math Kernel Library (oneMKL)

- **Other Technologies**:
  - Python 3.8+
  - Dash/Plotly for visualization
  - Web3.py for blockchain interaction
  - Google Gemini API for AI insights
  - NetworkX for graph analysis

## Installation

### Prerequisites

1. Install Intel oneAPI Base Toolkit: [Download Here](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html)
2. Python 3.8 or later

### Setup

```bash
# Clone the repository
[git clone https://github.com/yourusername/blockchain-monitoring.git](https://github.com/jms-luck/Ai_Powered_Cryptocurrency_fraud_detection)
cd blockchain-monitoring

# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate    # Windows

# Install requirements
pip install -r requirements.txt

# Install Intel optimizations
pip install scikit-learn-intelex
```

## Configuration

1. Create a `.env` file with your API keys:
```
BLOCKCHAIN_API_KEY=your_infura_api_key
GEMINI_API_KEY=your_google_gemini_api_key
```

2. Pre-trained models:
- Place your model files (`fake_identities_model.pkl`, `sybil_attacks_model.pkl`, `bot_activity_model.pkl`) in the root directory
- If models are not available, the system will train basic models on first run

## Running the Application

```bash
python app.py
```

The dashboard will be available at: `http://127.0.0.1:8050`

## Usage

1. **Dashboard Overview**:
   - Real-time transaction visualization
   - Alert notifications
   - Blocked accounts list
   - Network graph visualization

2. **Features**:
   - Automatic anomaly detection every 5 seconds
   - Manual AI analysis trigger
   - Interactive visualizations

3. **Alert Types**:
   - Fake identity detection
   - Sybil attack patterns
   - Bot activity
   - Graph-based anomalies
   - Unverified addresses

## Performance Optimization

The system leverages Intel oneAPI for:
- Faster machine learning inference
- Optimized numerical computations
- Efficient data processing
- Parallel execution of detection algorithms

## Troubleshooting

1. **Port conflicts**:
   - Edit the port number in `app.py` if default port (8050) is busy

2. **Model loading errors**:
   - Delete model files to force retraining of basic models

3. **API errors**:
   - Verify your API keys in `.env` file
   - Check your internet connection

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please open an issue or pull request for any improvements.

## Contact

For questions or support, please contact: [meenachisundaresan24@gmail.com](mailto:meenachisundaresan24@gmail.com)
