"""CartoPalette — Context-Aware Color Palette Generation for Thematic Cartography."""
from setuptools import setup, find_packages

setup(
    name="cartopalette",
    version="0.1.0",
    description="Basemap-adaptive color palette generation for thematic cartography",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "Pillow>=9.0.0",
        "scipy>=1.10.0",
    ],
    extras_require={
        "web": ["streamlit>=1.28.0"],
        "research": ["scikit-learn>=1.2.0", "PyYAML>=6.0"],
    },
)
