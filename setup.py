from setuptools import setup, find_packages
from pathlib import Path

# read the long description from README.md
here = Path(__file__).parent
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="ml-ids-analyzer",
    version="1.0.0",
    description=(
        "ML-powered IDS alert validation engine: "
        "batch & streaming inference, hyperparameter search, explainability"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Alexander Zimpher & Spencer Hendren",
    python_requires=">=3.8",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
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
            "mlids-train      = ml_ids_analyzer.model:main",
            "mlids-predict    = ml_ids_analyzer.inference.predict:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
