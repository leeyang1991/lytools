# coding='utf-8'

from setuptools import setup
long_description = open('README.md').read()
setup(
    name='lytools',
    version='0.0.120',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Yang Li',
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
    'scikit-learn',
    'outdated',
    'basemap',
    'netCDF4',
    'psutil',
    ],
)
