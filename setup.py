from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='samplify',
    version='0.1.0',
    author='Kevin Decker',
    author_email='kevindckr@gmail.com',
    description='Sampling algorithm implementation and testing for small remotely sensed data sets.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/kevindckr/samplify',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'numba',
        'scipy',
        'scikit-learn',
        'spectral',
        'matplotlib',
        'pandas',
        'geopandas',
        'esda',
        'libpysal',
        'psutil',
        'tqdm'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: GIS',
    ],
    python_requires='>=3.8',
    keywords='sampling, remote sensing, hyperspectral'
)

