from setuptools import find_packages, setup


setup(
    name='fer',
    version='0.0.1',
    description='Simple package for facial emotion recognition with threading.',
    author='Ozge Nilay Yalcin',
    url="https://github.com/o-n-yalcin/fer",
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    packages=find_packages(exclude=['test*']),
    python_requires='==3.6.*',
    install_requires=[
        'numpy',
        'opencv-python>=3.2',
        'tensorflow==1.12.*',
        'keras>=2.0,<2.3'
    ],
    tests_require=['pytest']
)
