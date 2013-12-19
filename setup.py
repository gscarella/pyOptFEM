#!/usr/bin/env python
try:
    from setuptools import setup, find_packages
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()
    from setuptools import setup, find_packages

import sys

Release=sys.version.split(' ')[0]
major,minor,micro=Release.split('.')


setup_defaults = {  
   'name'        : 'pyOptFEM',
   'description' : 'Python library for benchmarking Sparse Finite-Element matrices assembly',
   'version'     : '0.0.7',
   'url'         : 'http://www.math.univ-paris13.fr/~cuvelier',
   'author'      : 'Francois Cuvelier',
   'author_email': 'cuvelier@math.univ-paris13.fr',
   'license'     : 'BSD',
   'packages'    : ['pyOptFEM',
                    'pyOptFEM/FEM2D',
                    'pyOptFEM/FEM3D','pyOptFEM/valid2D','pyOptFEM/valid3D'
                   ],
   'classifiers':['Topic :: Scientific/Engineering :: Mathematics'],
   }    

# Python 2.7.3
if major=='2' and eval(minor)==7 and eval(micro)==3:
  setup(name=setup_defaults['name'],
        description = setup_defaults['description'],
        version=setup_defaults['version'],
        url=setup_defaults['url'],
        author=setup_defaults['author'],
        author_email=setup_defaults['author_email'],
        license = setup_defaults['license'],
        packages=setup_defaults['packages'],
        classifiers=setup_defaults['classifiers'],
        install_requires=['numpy >= 1.7.0','scipy >= 0.9.0','sympy >= 0.7.1.rc1','matplotlib >= 1.1.1rc']      
       ) 
  sys.exit()
  
#Python 2.7.5
if major=='2' and eval(minor)==7 and eval(micro)==5:
  setup(name=setup_defaults['name'],
        description = setup_defaults['description'],
        version=setup_defaults['version'],
        url=setup_defaults['url'],
        author=setup_defaults['author'],
        author_email=setup_defaults['author_email'],
        license = setup_defaults['license'],
        packages=setup_defaults['packages'],
        classifiers=setup_defaults['classifiers'],
        install_requires=['numpy >= 1.7.1','scipy >= 0.12.0','sympy >= 0.7.1.rc1','matplotlib >= 1.3.0']      
       ) 
  sys.exit()

# Python 3.3.2
if major=='3' and eval(minor)==3 and eval(micro)==2: 
  setup(name=setup_defaults['name'],
        description = setup_defaults['description'],
        version=setup_defaults['version'],
        url=setup_defaults['url'],
        author=setup_defaults['author'],
        author_email=setup_defaults['author_email'],
        license = setup_defaults['license'],
        packages=setup_defaults['packages'],
        classifiers=setup_defaults['classifiers'],
        install_requires=['numpy >= 1.7.1','scipy >= 0.12.0','sympy >= 0.7.3','matplotlib >= 1.3.0']     
       )
  sys.exit()
  
# Python 3.2.5
if major=='3' and eval(minor)==2 and eval(micro)==5: 
  setup(name=setup_defaults['name'],
        description = setup_defaults['description'],
        version=setup_defaults['version'],
        url=setup_defaults['url'],
        author=setup_defaults['author'],
        author_email=setup_defaults['author_email'],
        license = setup_defaults['license'],
        packages=setup_defaults['packages'],
        classifiers=setup_defaults['classifiers'],
        install_requires=['scipy >= 0.12.0','sympy >= 0.7.3','matplotlib >= 1.3.0','numpy >= 1.7.1, < 1.8']     
       )
  sys.exit()
  
  
# Python 3.1.5
if major=='3' and eval(minor)==1 and eval(micro)==5: 
  setup(name=setup_defaults['name'],
        description = setup_defaults['description'],
        version=setup_defaults['version'],
        url=setup_defaults['url'],
        author=setup_defaults['author'],
        author_email=setup_defaults['author_email'],
        license = setup_defaults['license'],
        packages=setup_defaults['packages'],
        classifiers=setup_defaults['classifiers'],
        install_requires=['matplotlib < 1.3.1','numpy < 1.8']     
       )
  sys.exit()
   
setup(name=setup_defaults['name'],
        description = setup_defaults['description'],
        version=setup_defaults['version'],
        url=setup_defaults['url'],
        author=setup_defaults['author'],
        author_email=setup_defaults['author_email'],
        license = setup_defaults['license'],
        packages=setup_defaults['packages'],
        classifiers=setup_defaults['classifiers'],
        install_requires=['numpy >= 1.7.1,<1.8.0','scipy >= 0.12.0','sympy >= 0.7.3','matplotlib >= 1.1.1']
     )
