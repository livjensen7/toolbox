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
    package_data={'toolbox/testdata': ['1_align.tif', '2_align.tif', '3_align.tif', '4_align.tif', '5_align.tif', '6_align.tif',
                              '7_align.tif', '8_align.tif', '9_align.tif', '10_align.tif', '11_align.tif']},

    include_package_data=True,
    install_requires=[
          'numpy','scipy','scikit-image'
      ],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)