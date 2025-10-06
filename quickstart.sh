#!/bin/bash

# Frame2KG Evaluation Quickstart Script

echo "Frame2KG Evaluation Toolkit - Quick Start"
echo "=========================================="
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | grep -Po '(?<=Python )[\d.]+')
echo "✓ Python version: $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Install package
echo ""
echo "Installing frame2kg-eval..."
pip install --upgrade pip
pip install -e .

echo ""
echo "✓ Installation complete!"
echo ""
echo "Available commands:"
echo "  frame2kg-eval     - Main evaluation command"
echo "  frame2kg-sweep    - Parameter sweep"
echo "  frame2kg-aggregate - Aggregate multiple runs"
echo "  frame2kg-doctor   - Sanity checker"
echo ""
echo "Example usage:"
echo "  frame2kg-eval --pred-dir ./predictions --gt hf:lewiswatson/Frame2KG-YC2:validation_dev --out results.csv"
echo ""
echo "To activate the environment in the future, run:"
echo "  source .venv/bin/activate"
