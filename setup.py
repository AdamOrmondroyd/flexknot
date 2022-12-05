import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="flexknot",
    version="1.0.0",
    author="Adam Neil Ormondroyd",
    author_email="adam.ormondroyd@gmail.com",
    description="Flex-Knot",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ormorod/flexknot",
    packages=["flexknot"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Licence :: TODO",
        "Operating System :: Linux",
    ],
    install_requires=[
        "numpy",
        "scipy",
        "pypolychord",
    ],
)
