from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ez4d",
    version="0.1.0",
    author="Yan Xia",
    author_email="yan.xia@utexas.edu",
    description="EZ4D - Easy 4D processing library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Core dependencies
        "torch>=1.9.0",
        "numpy>=1.19.0",

        # Visualization
        "wis3d",
        "trimesh",
        "opencv-python",
        "imageio",
        "imageio-ffmpeg",
        "matplotlib",
        "pillow",

        # Utilities
        "tqdm",
        "einops",
        "colorlog",
        "rich",

        # Optional but recommended for video loading
        "decord",
    ],
    extras_require={
        "pytorch3d": [
            "pytorch3d",
        ],
        'pyrender': [
            "pyrender",
        ],
    },
)

