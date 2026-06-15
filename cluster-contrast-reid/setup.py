from setuptools import setup, find_packages


setup(
    name="clustercontrast-sg",
    version="1.0.0",
    description="SG cluster contrast for unsupervised person re-identification",
    install_requires=[
        "numpy",
        "torch",
        "torchvision",
        "six",
        "h5py",
        "Pillow",
        "scipy",
        "scikit-learn",
        "metric-learn",
        "faiss-gpu",
        "yacs",
    ],
    packages=find_packages(),
    keywords=[
        "Unsupervised Learning",
        "Contrastive Learning",
        "Person Re-Identification",
    ],
)
