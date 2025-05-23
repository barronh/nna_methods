from setuptools import setup
import os
import shutil

with open('README.md') as f:
    longdescstr = f.read()

with open('nna_methods/__init__.py') as f:
    for l in f.readlines():
        if '__version__' in l:
            versionstr = eval(l.strip().split('=')[-1].strip())
            break
    else:
        versionstr = '-9.0.0'

setup(
    name='nna_methods',
    version=versionstr,
    description='Nearest Neighbor Averaging methods',
    long_description=longdescstr,
    long_description_content_type='text/markdown',
    author='Barron H. Henderson',
    author_email='barronh@gmail.com',
    packages=['nna_methods'],
    install_requires=['numpy', 'scipy', 'pandas', 'scikit-learn'],
)
