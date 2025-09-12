from setuptools import setup, find_packages

setup(
    name="ids5g",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'click',
        'torch',
        'pandas',
        'scikit-learn',
        'joblib',
        'tqdm',
        'torchattacks',
        'seaborn',
        'matplotlib'
    ],
    entry_points={
        'console_scripts': [
            'ids5g = src.cli:cli',
        ],
    },
)
