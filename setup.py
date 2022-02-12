import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="linf",
    version="0.0",
    author="Adam Neil Ormondroyd",
    author_email="adam.ormondroyd@gmail.com",
    description="Linear INterpolation Functions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ormorod/linf",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Licence :: TODO",
        "Operating System :: OS Independent",
    ],
)
