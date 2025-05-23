from setuptools import setup, find_packages

with open("vit_tactile/README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("vit_tactile/requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line for line in fh.read().splitlines() if not line.startswith('#')]

# Ensure PyTorch is installed first
install_requires = [req for req in requirements if req.startswith("torch")]
install_requires.extend([req for req in requirements if not req.startswith("torch")])

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
    install_requires=install_requires,
)