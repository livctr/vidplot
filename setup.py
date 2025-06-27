from setuptools import setup, find_packages

setup(
    name="vidplot",
    version="0.1.0",
    description="A video annotation visualizer for computer vision and data science workflows.",
    author="Victor Li",
    author_email="vhl2022@nyu.edu",
    packages=find_packages(),
    install_requires=[
        "opencv-python>=4.5.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "tqdm>=4.60.0",
        "matplotlib>=3.4.0",
    ],
    extras_require={
        "decord": ["decord>=0.6.0"],
        "av": ["av>=10.0.0"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8-pyproject==1.2.3",
            "flake8>=4.0.0",
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "ruff>=0.4.0",
        ],
    },
    python_requires=">=3.8",
    include_package_data=True,
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/vidplot",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
