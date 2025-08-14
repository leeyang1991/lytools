# coding='utf-8'

from setuptools import setup

def get_version():
    init_f = './lytools/__init__.py'
    with open(init_f) as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[-1].strip().strip("'")
version = get_version()

long_description = open('README.md').read()
setup(
    name='lytools',
    version=version,
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
    'basemap',
    'netCDF4',
    'psutil',
    'openpyxl',
    ],
)
