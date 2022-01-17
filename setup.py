# coding='utf-8'

from setuptools import setup

setup(
    name='lytools',
    version='0.0.21',
    author='Yang Lee',
    author_email='leeyang1991@gmail.com',
    packages=['lytools'],
    url='https://github.com/leeyang1991/lytools',
    python_requires='>=3',
    install_requires=[
    'matplotlib',
    'numpy',
    'scipy',
    'gdal',
    'tqdm',
    'pandas',
    'seaborn',
    'sklearn',
    'requests',
    ],
)