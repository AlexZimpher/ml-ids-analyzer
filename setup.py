# setup.py

from setuptools import setup, find_packages

setup(
    # THIS MUST MATCH what your entry-points use below
    name="ml-ids-analyzer",
    version="1.0.0",
    # Tell setuptools where your modules live
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "matplotlib",
        "seaborn",
        "joblib",
        "PyYAML",
        "shap",
    ],
    entry_points={
        "console_scripts": [
            "mlids-preprocess=src.preprocess:main",
            "mlids-train=src.model:train_model",
            "mlids-predict=src.predict:predict_new_data",
        ]
    },
)
