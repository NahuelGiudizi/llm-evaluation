from setuptools import setup, find_packages

setup(
    name="llm-benchmark-toolkit",
    version="2.4.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "ollama>=0.1.0",
        "datasets>=2.14.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "plotly>=5.17.0",
        "scipy>=1.11.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "llm-eval=llm_evaluator.cli:cli",
            "llm-dashboard=llm_evaluator.dashboard:main",
        ],
    },
    python_requires=">=3.11",
)
