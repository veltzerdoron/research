import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="quant-veltzer-doron",
    version="0.0.1",
    author="Veltzer Doron",
    author_email="veltzerdoron@gmail.com",
    description="ANN based learning of quantifiers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/quant",
    packages = setuptools.find_packages(
        where = 'src',
        include = ['pkg*',],
        exclude = ['tests',]
    ),
    package_dir = {"":"src"},
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)

