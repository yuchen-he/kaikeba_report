try:
    from pip._internal.req import parse_requirements
except ImportError:
    from pip.req import parse_requirements

from setuptools import (
    find_packages,
    setup,
)

requirements = [str(ir.req) for ir in parse_requirements("requirements.txt", session=False)]

setup(
    name='vector',
    version='0.0.1',
    description='vector',
    packages=find_packages(exclude=[]),
    author='rambo',
    author_email='xxx@gmail.com',
    license='Apache License v2',
    package_data={'': ['*.*']},
    url='',
    install_requires=requirements,
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: Unix',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
