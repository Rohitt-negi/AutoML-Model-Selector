#!/usr/bin/env bash
# AutoML Model Selector — Quick Startup Script
set -e

echo "=========================================================="
echo "🤖 AutoML Model Selector — Final Year B.Tech Project"
echo "=========================================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 is not installed or not in PATH."
    exit 1
fi

# Create virtual environment if it does not exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment 'venv'..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Checking and installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Launch Streamlit app
echo "🚀 Launching AutoML Model Selector..."
streamlit run app.py

