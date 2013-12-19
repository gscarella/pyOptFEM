**pyOptFEM** is a Python module which aims at measuring and comparing 
the performance of three programming techniques for **assembling finite element matrices in 2D and 3D**.
For each of the matrices studied, three assembly versions are available: ``base``, ``OptV1`` and ``OptV2``

Three matrices are currently implemented : **Mass matrix**, **Stiffness matrix** and **Elastic stiffness matrix**.

More details are given in html or pdf documentations : 
  - http://www.math.univ-paris13.fr/~cuvelier/software/docs/Software/FEM/pyOptFEM/<VERSION>/pyOptFEM_<VERSION>.pdf
  - http://www.math.univ-paris13.fr/~cuvelier/software/docs/Software/FEM/pyOptFEM/<VERSION>/html/index.html
  
1) Requirements
  **pyOptFEM** works on Python3 or Python2 and needs ``numpy``, ``scipy``, ``sympy`` and ``matplotlib`` modules.
  
2) Testing and Working
  - Python 3.3.2 under Ubuntu 12.04 LTS (x86_64) with 16Go RAM and :
    * ``numpy`` (1.7.1)
    * ``scipy`` (0.12.0)
    * ``sympy`` (0.7.3)
    * ``matplotlib`` (1.3.0)
  - Python 2.7.5 under Ubuntu 12.04 LTS (x86_64) with 16Go RAM and :
    * ``numpy`` (1.7.1)
    * ``scipy`` (0.12.0) or (0.13.0)
    * ``sympy`` (0.7.1) or (0.7.1)
    * ``matplotlib`` (1.3.0) or (1.3.1)
  - Python 2.7.5 under Windows 7 (64bit), Anaconda 1.7.0 distribution (http://continuum.io/downloads) with 4Go RAM and :
    * ``numpy`` (1.7.1)
    * ``scipy`` (0.12.0)
    * ``sympy`` (0.7.3)
    * ``matplotlib`` (1.3.0)