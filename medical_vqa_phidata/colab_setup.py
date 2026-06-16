#!/usr/bin/env python3
"""
colab_setup.py — One-shot Google Colab setup.

Run this first in your Colab notebook:
    !python colab_setup.py
"""
import subprocess, sys, os

def run(cmd):
    print(f"\n>>> {cmd}")
    subprocess.run(cmd, shell=True)

print("="*55)
print("  Medical VQA PhiData — Colab Setup")
print("="*55)

run("pip install --upgrade pip -q")
run("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q")
run("pip install -r requirements.txt -q")
run("pip install bitsandbytes -q")

# Install Ollama
run("curl -fsSL https://ollama.com/install.sh | sh")

# Pull default model in background
run("ollama serve &")
import time; time.sleep(3)
run("ollama pull mistral")

# NLTK data
run('python -c "import nltk; nltk.download(\'punkt\', quiet=True)"')

# Directories
for d in ["data/raw","data/processed","data/features",
          "artifacts/checkpoints","artifacts/evaluation","logs"]:
    os.makedirs(d, exist_ok=True)

print("\n✅ Setup complete!")
print("\nRun pipeline:")
print('  !python main.py --image /content/xray.jpg --question "Is there pneumonia?"')
print("\nRun inference only:")
print('  !python inference.py --image /content/xray.jpg --question "..." --checkpoint ./artifacts/checkpoints/...')
