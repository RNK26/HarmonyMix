from setuptools import setup, find_packages

setup(
    name="harmonymix",
    version="0.1.0",
    description="A modular music recommendation engine using Content-Based, Collaborative Filtering, and Hybrid algorithms.",
    author="RNK26",
    url="https://github.com/RNK26/HarmonyMix",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "streamlit>=1.10.0",
        "dvc>=2.8.0",
    ],
    python_requires=">=3.9",
)
