from setuptools import find_namespace_packages, setup

setup(
    name='cr-validation',
    version='0.0.1',
    description='Credit risk validation routines @ CR Consulting',
    # packages=find_namespace_packages(include='betteruse.*'), <-- Only used if we want to join multiple repo's into 1 CR package
    packages=['cr'],
    install_requires=(
        'numpy>=1.0.0',
        'pandas>=1.0.0',
        'plotly>=5.3.1',
        'kaleido>=0.2.1', # plotly IO
    ),
    setup_requires=(
        'pytest-runner',
    )
)