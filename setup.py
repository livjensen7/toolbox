import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="toolbox",
    version="0.0.1",
    author="Sy Redding",
    description="Redding Lab analysis tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/github/ReddingLab/toolbox",
    packages=setuptools.find_packages(),
    package_data={'toolbox': ['testdata/*.tif']},
    include_package_data=True,
    install_requires=[
          'numpy','scipy','scikit-image','matplotlib'
      ],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)