import torchlight
from setuptools import setup, find_packages

version = torchlight.__version__

setup(
    name="torchlight",
    version=version,
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision==0.19.1",
        "numpy",
        "tqdm",
        "pillow",
        "matplotlib",
    ],
    author="Harikesh Kushwaha",
    author_email="harikeshkumar0926@gmail.com",
    url="https://github.com/Hari31416/torchlight",
    keywords=[
        "pytorch",
        "mechanistic interpretability",
        "machine learning",
        "neural networks",
        "convolutional neural networks",
        "feature visualization",
        "optimization",
    ],
    license="MIT",
    description="A library to visualize features learned by CNNs",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
