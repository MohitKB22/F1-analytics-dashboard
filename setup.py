"""
setup.py
Run `pip install -e .` from the project root once.
After that every script (train.py, predict.py, app.py, …) can
import `data.*` and `models.*` from any working directory.
"""

from setuptools import setup, find_packages

setup(
    name="f1_hybrid_ai",
    version="1.0.0",
    description="F1 Hybrid AI Prediction System – XGBoost × LSTM × Weather (1950-2026)",
    packages=find_packages(),          # finds data/ and models/ automatically
    python_requires=">=3.9,<3.14",     # Python 3.14 has upstream xgboost issues
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.3",
        "joblib>=1.3.0",
        "streamlit>=1.32.0",
        "plotly>=5.18.0",
    ],
    extras_require={
        "lstm": ["tensorflow>=2.15.0"],   # optional full LSTM
    },
)
