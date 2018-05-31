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
    include_package_data=True,
    data_files=[('tif', ['1_align.tif', '2_align.tif',
                         '3_align.tif', '4_align.tif',
                         '5_align.tif', '6_align.tif',
                         '7_align.tif', '8_align.tif',
                         '9_align.tif', '10_align.tif',
                         '11_align.tif'])]
    install_requires=[
          'numpy','scipy','scikit-image'
      ],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)