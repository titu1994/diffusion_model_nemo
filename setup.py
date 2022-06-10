from setuptools import setup, find_packages

with open('requirements.txt') as f:
    req = f.readlines()


setup(
    name='diffusion_model_nemo',
    version='0.1.0',
    # url='https://github.com/mypackage.git',
    author='Somshubra Majumdar',
    author_email='titu1994@gmail.com',
    description='Port of HF DDPM implementation',
    packages=find_packages(),
    install_requires=req,
)
