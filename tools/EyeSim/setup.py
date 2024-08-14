from setuptools import setup


setup(
    packages=['EyeSim'],  # same as name
    name='EyeSim',
    version='0.1',
    install_requires=[
        'gymnasium',
        'box2d_py',
        'numpy',
        'matplotlib',
        'scikit-image',
    ],
)
