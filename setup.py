from setuptools import setup, find_packages

setup(
    name="picoolfx",
    version="0.1.0",
    description="A library for applying cool effects to images",
    author="Nicola Cerutti",
    author_email="nc@nicores.de",
    packages=find_packages(),
    install_requires=[
        "<geopandas>",
        "<matplotlib>",
        "<noise>",
        "<numpy>",
        "<pandas>",
        "<Pillow> >= 9.0.0",
        "<rasterio> >= 1.0.0",
        "<Shapely>",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
