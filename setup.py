import setuptools

setuptools.setup(
    name='gnies',
    version='0.0.2',
    author='Juan L. Gamella',
    author_email='juangamella@gmail.com',
    packages=['gnies'],
    scripts=[],
    url='https://github.com/juangamella/gnies',
    license='BSD 3-Clause License',
    description='',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    install_requires=['numpy>=1.17.0', 'ges>=1.1.0']
)
