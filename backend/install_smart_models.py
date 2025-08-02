#!/usr/bin/env python3
"""
Install Smart Models for Resume Processing
This script installs the best available models for document processing
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def install_smart_models():
    """Install all smart models and dependencies"""
    
    print("🚀 Installing Smart Models for Resume Processing")
    print("=" * 50)
    
    # 1. Install Anthropic (Claude 3.5 Sonnet)
    success = run_command(
        "pip install anthropic",
        "Installing Anthropic SDK for Claude 3.5 Sonnet"
    )
    
    # 2. Install Marker (PDF to Markdown)
    success = run_command(
        "pip install marker-pdf",
        "Installing Marker for PDF to Markdown conversion"
    )
    
    # 3. Install Surya (OCR and Layout Analysis)
    success = run_command(
        "pip install surya-ocr",
        "Installing Surya for OCR and layout analysis"
    )
    
    # 4. Install additional dependencies
    dependencies = [
        "torch",
        "torchvision", 
        "transformers",
        "pillow",
        "opencv-python",
        "numpy"
    ]
    
    for dep in dependencies:
        run_command(
            f"pip install {dep}",
            f"Installing {dep}"
        )
    
    print("\n" + "=" * 50)
    print("📋 Setup Instructions:")
    print("=" * 50)
    
    print("\n1. 🔑 Set up API Keys:")
    print("   Add to your .env file:")
    print("   ANTHROPIC_API_KEY=your_claude_api_key_here")
    print("   OPENAI_API_KEY=your_openai_api_key_here")
    
    print("\n2. 🧪 Test the installation:")
    print("   python test_smart_models.py")
    
    print("\n3. 🚀 Run the smart API:")
    print("   python smart_models_api.py")
    
    print("\n📚 Model Information:")
    print("   - Claude 3.5 Sonnet: Best for text optimization")
    print("   - Marker: High-quality PDF to Markdown")
    print("   - Surya: Open source OCR + layout analysis")
    print("   - Fallbacks: GPT-4o and PyMuPDF if others fail")

if __name__ == "__main__":
    install_smart_models()
