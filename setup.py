"""
Setup script for Intervention Search package
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="intervention-search",
    version="2.0.0",
    author="Causal AI Team",
    author_email="",
    description="Production-Grade Causal Intervention System with Proper Uncertainty Quantification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/intervention-search",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.1.0",
        "networkx>=2.5.0",
        "scikit-learn>=0.24.0",
        "scipy>=1.5.0",
    ],
    extras_require={
        "ml": [
            "xgboost>=1.3.0",
            "lightgbm>=3.0.0",
            "catboost>=0.24.0",
        ],
        "viz": [
            "matplotlib>=3.3.0",
            "seaborn>=0.11.0",
            "plotly>=4.14.0",
        ],
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.0",
            "flake8>=3.9.0",
        ],
        "all": [
            "xgboost>=1.3.0",
            "lightgbm>=3.0.0",
            "catboost>=0.24.0",
            "matplotlib>=3.3.0",
            "seaborn>=0.11.0",
            "plotly>=4.14.0",
        ],
    },
    entry_points={
        "console_scripts": [
            # Add CLI tools if needed in future
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "causal-inference",
        "intervention-search",
        "causal-discovery",
        "uncertainty-quantification",
        "monte-carlo",
        "bayesian-optimization",
        "root-cause-analysis",
    ],
)
