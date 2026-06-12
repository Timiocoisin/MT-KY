from setuptools import find_packages, setup

setup(
    name="boilerwear",
    version="1.0.0",
    description="BoilerWear-190: Fine-Grained Ordinal Wear Estimation",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "timm>=0.9.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "scikit-image>=0.21.0",
        "Pillow>=10.0.0",
        "PyYAML>=6.0",
        "tqdm>=4.65.0",
        "pandas>=2.0.0",
    ],
)
