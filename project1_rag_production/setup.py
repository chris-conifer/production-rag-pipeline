"""
Setup file for RAG Production Pipeline
"""

from setuptools import setup, find_packages

setup(
    name="rag-production",
    version="1.0.0",
    author="Chris",
    description="Production-ready RAG pipeline with comprehensive evaluation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        # Core dependencies loaded from requirements.txt
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)



