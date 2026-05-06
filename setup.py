from setuptools import setup, find_packages

setup(
    name="helga",
    version="0.1.0",
    description="Humanistic Environmental Learning and Generating Agent",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "pyro-ppl>=1.8.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "pytest>=7.4.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "sympy>=1.12.0",
    ],
)