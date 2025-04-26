from setuptools import setup, find_packages

setup(
    name='ml_ids_analyzer',
    version='1.0.0',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy', 'pandas', 'scikit-learn', 'xgboost',
        'matplotlib', 'seaborn', 'joblib', 'PyYAML', 'shap'
    ],
     entry_points={
     'console_scripts': [
         'mlids-preprocess=preprocess:main',
         'mlids-train=model:train_model',
         'mlids-predict=predict:predict_new_data',
     ]
     'console_scripts': [
         'mlids-preprocess=src.preprocess:main',
         'mlids-train=src.model:train_model',
         'mlids-predict=src.predict:predict_new_data',
     ]
 }
