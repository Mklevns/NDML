# setup.py - NDML Package Installation
from setuptools import setup, find_packages
import os

# Read version from __init__.py
def get_version():
    with open(os.path.join("ndml", "__init__.py"), "r") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    return "1.0.0"

# Read long description from README
def get_long_description():
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return """
        NDML - Neuromorphic Distributed Memory Layer
        
        A sophisticated AI memory system inspired by biological neural networks,
        featuring multi-timescale dynamics, distributed consensus, and intelligent
        memory lifecycle management.
        """

setup(
    name="ndml",
    version=get_version(),
    author="NDML Team",
    author_email="contact@ndml.ai",
    description="Neuromorphic Distributed Memory Layer for AI Systems",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/ndml-team/ndml",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9", 
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.21.0",
        "faiss-cpu>=1.7.0",  # or faiss-gpu for GPU support
        "transformers>=4.20.0",
        "asyncio",
        "aiohttp>=3.8.0",
        "PyYAML>=6.0",
        "logging",
        "pathlib",
        "dataclasses",
        "typing_extensions>=4.0.0",
        "scipy>=1.8.0",
    ],
    extras_require={
        "gpu": ["faiss-gpu>=1.7.0"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.20.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
            "coverage>=6.0.0",
        ],
        "deployment": [
            "kubernetes>=24.0.0",
            "docker>=6.0.0",
            "prometheus-client>=0.15.0",
        ],
        "examples": [
            "jupyter>=1.0.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "pandas>=1.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ndml=ndml.main:main",
            "ndml-test=ndml.main:main",
            "ndml-consensus=ndml.deployment.consensus_node:main",
        ],
    },
    include_package_data=True,
    package_data={
        "ndml": [
            "config/*.yaml",
            "deployment/*.yaml",
            "deployment/kubernetes/*.yaml",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/ndml-team/ndml/issues",
        "Source": "https://github.com/ndml-team/ndml",
        "Documentation": "https://ndml.readthedocs.io/",
    },
)
