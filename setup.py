"""Setup configuration for AI-Aware PCB package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ai-aware-pcb",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-Aware Predictive Collision Boundary framework for autonomous driving safety",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/AI-Aware-PCB",
    packages=find_packages(exclude=["tests*", "*.tests"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    include_package_data=True,
    zip_safe=False,
)
