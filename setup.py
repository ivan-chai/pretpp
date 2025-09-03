import os
import setuptools

with open("README.md") as fp:
    long_description = fp.read()

setuptools.setup(
    name="pretpp",
    version="0.0.1",
    author="Ivan Karpukhin",
    author_email="karpuhini@yandex.ru",
    description="Advanced pretraining for TPP, MTPP and Event Sequences.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(include=["pretpp", "pretpp.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "hotpp-benchmark>=0.5.9"
    ]
)
