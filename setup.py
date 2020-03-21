from setuptools import setup, find_packages

setup(
    name="bigearthnet",
    version="1.0.0",
    description="Big Earth Net Image Segmentation Transfer Learning",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy>=1.2.0",
    ],
)
