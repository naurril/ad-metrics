from setuptools import setup, find_packages

setup(
    name="admetrics",
    version="0.1.0",
    description="Comprehensive metrics library for autonomous driving: detection, tracking, prediction, and localization",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
    ],
    extras_require={
        "viz": [
            "matplotlib>=3.3.0",
            "open3d>=0.13.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
    },
    python_requires=">=3.8",
)
