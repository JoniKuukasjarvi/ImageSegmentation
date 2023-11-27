from setuptools import setup
# tkinter might be needed to install separately
setup(
    name='ImageSegmentation',
    version='0.1',
    scripts=['ImageSegmentation.py'],
    packages=[],
    install_requires=[
        'numpy',
        'tk',
        'tensorflow',
        'Pillow',
        'matplotlib',
        'keras',
    ],
    entry_points={
        'console_scripts': [
            # Replace with your actual script and module
            'image_segmentation=ImageSegmentation:main',
        ],
    },
)
