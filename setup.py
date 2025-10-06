"""Setup script for Frame2KG evaluation toolkit."""

from setuptools import setup, find_packages

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="frame2kg-eval",
    version="1.0.0",
    author="Frame2KG Team",
    description="Frame2KG benchmark evaluation toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/frame2kg/evaluation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "datasets>=2.0.0",
        "Pillow>=8.0.0",
        "PyYAML>=6.0",
        "tqdm>=4.62.0",
        "click>=8.0.0",
        "sentence-transformers>=2.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "frame2kg-eval=frame2kg_eval.cli.evaluate:main",
            "frame2kg-sweep=frame2kg_eval.cli.sweep:main",
            "frame2kg-aggregate=frame2kg_eval.cli.aggregate:main",
            "frame2kg-doctor=frame2kg_eval.cli.doctor:main",
        ],
    },
    package_data={
        "frame2kg_eval": ["config/*.yaml"],
    },
    include_package_data=True,
)
