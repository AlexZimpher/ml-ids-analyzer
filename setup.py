from setuptools import setup, find_packages

setup(
    name="ml-ids-analyzer",
    version="1.0.0",
    description=(
        "ML-powered IDS alert validation engine: "
        "batch & streaming inference, hyperparameter search, explainability"
    ),
    author="Alexander Zimpher & Spencer Hendren",
    python_requires=">=3.8",
    packages=find_packages(
        include=["ml_ids_analyzer", "ml_ids_analyzer.*"]
    ),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "xgboost>=1.6.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "joblib>=1.1.0",
        "PyYAML>=5.4.0",
        "shap>=0.40.0",
    ],
    entry_points={
        "console_scripts": [
            "mlids-preprocess = ml_ids_analyzer.preprocessing.preprocess:main",
            "mlids-train      = ml_ids_analyzer.modeling.train:main",
            "mlids-predict    = ml_ids_analyzer.inference.predict:main",
        ]
    },
    include_package_data=True,
)
