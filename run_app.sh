#!/bin/bash
# Startup script for Solar Regatta Gradio App

echo "=========================================="
echo "Solar Regatta - Starting Application"
echo "=========================================="
echo ""

# Use the correct Python
PYTHON="/opt/anaconda3/bin/python3"

echo "Python version:"
$PYTHON --version
echo ""

echo "Checking dependencies..."
$PYTHON -c "import gradio; print('✓ Gradio:', gradio.__version__)" || {
    echo "✗ Gradio not found. Installing..."
    pip install -r requirements.txt
}

echo ""
echo "Starting Gradio app..."
echo "Once started, open http://localhost:7860 in your browser"
echo "Press Ctrl+C to stop"
echo ""

$PYTHON app.py
