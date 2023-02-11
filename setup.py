from setuptools import setup, find_packages

setup(
    name='bioval',
    version='0.1.0',
    description='A package for Bioinformatics Validation',
    author='Ihab Bendidi, Ethan Cohen, Auguste Genovesio',
    packages=find_packages(),
    install_requires=[
        'pynvml',
        'numpy',
        'torch',
        'torchvision',
        'Pillow',
        'nvidia-ml-py',
        'poetry',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)
