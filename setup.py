from setuptools import setup, find_packages

# Read the contents of the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mattervial",  # Package name
    version="0.1.3",  # Version number
    author="Rogério A. Gouvêa",  # Add your name or the authorship group
    author_email="rogeriog.em@gmail.com",  # Add your email
    description="A package that uses pretrained graph-neural network models and symbolic regression formulas on material descriptors as featurizers for interpretable predictions in materials science.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rogeriog/MatterVial",  # URL to the project
    packages=find_packages(include=['mattervial', 'mattervial.*', 'mattervial.packages.*', 'mattervial.packages.roost.*', 'mattervial.featurizers.formulas', 'mattervial.interpreter' ]),    
    include_package_data=True,  # Include non-Python files specified in MANIFEST.in
    license="MIT License",  # License type
    classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',  # Specify Python version compatibility
    install_requires=[
    "tensorflow>=2.0",
    "scikit-learn>=0.24.0",
    "h5py>=2.10.0",
    "pandas>=1.1.0",
    "numpy>=1.18.0",
    "megnet>=1.0.0",
    "keras>=2.3.0",
    "contextlib2",  # for managing stdout redirection
    "pickle-mixin",  # for loading and saving scalers
    "pymatgen>=2022.0.0",  # for handling crystal structures
    "torch",
    "torch_scatter",
   ],
    package_data={
        'mattervial': [
            'packages/roost/**/*',  # Include all files and subdirectories in mattervial/packages/roost
            'packages/roost/CITATION.cff',
            'packages/roost/LICENSE',
            'packages/roost/README.md',
            'featurizers/custom_models/*.h5', 'featurizers/custom_models/*.pkl',
            'featurizers/custom_models/*.json', 'featurizers/custom_models/*.tar',
            'featurizers/formulas/*.txt'],
        'mattervial.interpreter': [
            'formulas/*.json',
            'shap_plots/**/*',
            'shap_values/**/*',
            'data/**/*',
        ],
    },
    entry_points={
        'console_scripts': [
            # Add any scripts you want to run directly from the command line
        ],
    },
)
