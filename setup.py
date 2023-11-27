from setuptools import setup
# tkinter might be needed to install separately
setup(
    name='SteelSegmentation',
    version='0.1',
    scripts=['SteelSegmentation.py'],
    packages=[],
    install_requires=[
        'numpy',
        'tkinter',
        'tensorflow',
        'Pillow',
        'matplotlib',
        'keras',
    ],
    entry_points={
        'console_scripts': [
            # Replace with your actual script and module
            'steel_segmentation=SteelSegmentation:main',
        ],
    },
)
