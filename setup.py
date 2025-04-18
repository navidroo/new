from setuptools import setup, find_packages

with open("vit_tactile/README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("vit_tactile/requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="vit_tactile",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="ViT-Tiny tactile encoder with Performer attention and dynamic early-exit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/vit-tactile",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
)