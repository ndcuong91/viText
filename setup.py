import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="viText",
    version="0.0.1",
    author="cuongnd",
    author_email="titikid@gmail.com",
    long_description = long_description,
    description="OCR",
    install_requires=[],
    url="https://github.com/titikid/viText",
    packages=setuptools.find_packages(),
    #package_data={'classifier_crnn': []},
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",

    ],
    python_requires = '>=3.6',
)
