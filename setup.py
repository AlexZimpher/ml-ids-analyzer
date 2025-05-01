from setuptools import setup, find_packages

setup(
    name="ml-ids-analyzer",
    version="0.1.0",
    author="Your Name",
    description="ML pipeline for IDS alert analysis",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "PyYAML",
        "joblib",
        "click",
        "fastapi",
        "uvicorn[standard]",
        "pydantic",
        "httpx",
    ],
    entry_points={
        "console_scripts": [
            "mlids-preprocess = ml_ids_analyzer.preprocessing.preprocess:main",
            "mlids-train      = ml_ids_analyzer.model:main",
            "mlids-predict    = ml_ids_analyzer.inference.predict:main",
        ],
    },
)
