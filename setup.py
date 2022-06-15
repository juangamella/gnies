import setuptools

setuptools.setup(
    name='gnies',
    version='0.0.1',
    author='Juan L. Gamella',
    author_email='juangamella@gmail.com',
    packages=['template'],
    scripts=[],
    url='https://github.com/juangamella/gnies',
    license='BSD 3-Clause License',
    description='',
    long_description=open('README_pypi.md').read(),
    long_description_content_type="text/markdown",
    install_requires=['numpy>=1.17.0', 'ges>=1.1.0']
)
