# setup.py
from setuptools import setup, find_packages

# Read README.md for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hybf",
    version="0.1.0",
    author="Marty Blaber",
    author_email="your.email@example.com",
    description="A hybrid binary format for efficient storage of tabular data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/martyblaber/hybf",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "pytest-xdist>=2.0",
            "black>=22.0",
            "isort>=5.0",
            "flake8>=4.0",
            "mypy>=0.900",
        ],
        "test": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "pytest-xdist>=2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hybf=hybf.cli:main",  # If you add CLI tools later
        ],
    },
)