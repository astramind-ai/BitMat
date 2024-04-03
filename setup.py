from setuptools import setup, find_packages

setup(
    name='bitmat-tl',
    version='0.2.2',
    author='Marco Lironi',
    author_email='marcolironi@astramind.ai',
    description='An efficent implementation for the paper: "The Era of 1-bit LLMs"',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/astramind-ai/BitMat/tree/main',
    packages=find_packages(),
    install_requires=[
        'torch',
        'triton',
        'transformers',
        'bitsandbytes',
        'numpy',
        'tqdm'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
