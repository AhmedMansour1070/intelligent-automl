#!/usr/bin/env python

from setuptools import setup, find_packages
import os

# Read version from version.py
version_file = os.path.join('intelligent_automl', 'version.py')
version = {}
with open(version_file) as f:
    exec(f.read(), version)

# Read README for long description
try:
    with open('README.md', 'r', encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "Intelligent AutoML Framework - The world's most intelligent automated machine learning framework"

# Read requirements
try:
    with open('requirements.txt', 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
except FileNotFoundError:
    requirements = [
        'pandas>=1.3.0',
        'numpy>=1.21.0',
        'scikit-learn>=1.1.0',
        'matplotlib>=3.5.0',
        'seaborn>=0.11.0',
        'joblib>=1.1.0',
        'tqdm>=4.62.0',
    ]

setup(
    name='intelligent-automl',
    version=version['__version__'],
    author='Your Name',
    author_email='your.email@example.com',
    description='The world\'s most intelligent automated machine learning framework',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/intelligent-automl',
    project_urls={
        'Documentation': 'https://intelligent-automl.readthedocs.io',
        'Source': 'https://github.com/yourusername/intelligent-automl',
        'Tracker': 'https://github.com/yourusername/intelligent-automl/issues',
    },
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'black>=22.0.0',
            'flake8>=5.0.0',
            'mypy>=0.991',
            'pre-commit>=2.20.0',
        ],
        'docs': [
            'sphinx>=5.0.0',
            'sphinx-rtd-theme>=1.0.0',
            'sphinx-autodoc-typehints>=1.19.0',
        ],
        'all': [
            'optuna>=3.0.0',
            'scikit-optimize>=0.9.0',
            'xgboost>=1.6.0',
            'lightgbm>=3.3.0',
            'catboost>=1.1.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'intelligent-automl=intelligent_automl.cli:main',
        ],
    },
    keywords='machine learning, automl, data preprocessing, artificial intelligence, feature engineering',
    include_package_data=True,
    zip_safe=False,
)