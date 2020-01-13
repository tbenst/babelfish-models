from setuptools import setup, find_packages

setup(name='babelfish-models',
      version='0.1',
      description='I can speak fish!',
      url='https://github.com/tbenst/babelfish-models',
      author='Tyler Benster',
      author_email='github@tylerbenster.com',
      # license='None',
      packages=find_packages(),
      zip_safe=False,
      classifiers=[
          # How mature is this project? Common values are
          #   3 - Alpha
          #   4 - Beta
          #   5 - Production/Stable
          'Development Status :: 3 - Alpha',

          # Indicate who your project is intended for
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering',

          # Pick your license as you wish (should match "license" above)

          # Specify the Python versions you support here. In particular, ensure
          # that you indicate whether you support Python 2, Python 3 or both.
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
      ],
      keywords='neuroscience 2P',
      setup_requires=['pytest-runner'],
      tests_require=['pytest'],
      )
