import numpy as np
from scipy.spatial import Delaunay
from .toolsVec import ComputeAreaOpt
from .toolsVec import ComputeAreaOpt2
import matplotlib.pyplot as plt


class getMesh:
   """
   Reads a *FreeFEM++* mesh from file ``meshfile``. Class attributes are :
   
     - **nq**, total number of mesh vertices (points), also denoted :math:`\\nq`.
     - **nme**, total number of mesh elements (triangles in 2d),
     - **version**, mesh structure version,
     - **q**, *Numpy* array of vertices coordinates, dimension ``(nq,2)`` *(version 0)* or ``(2,nq)`` *(version 1)*. 
     
       ``q[j]``  *(version 0)* or ``q[:,j]`` *(version 1)* are the two coordinates of the :math:`j`-th vertex, :math:`j\in\{0,..,nq-1\}`
       
     - **me**, *Numpy* connectivity array,  dimension ``(nme,3)`` *(version 0)* or ``(3,nme)`` *(version 1)*. 
       
       ``me[k]``  *(version 0)* or ``me[:,k]`` *(version 1)* are the storage index of the three vertices of the :math:`k`-th triangle in the array ``q`` of vertices coordinates, :math:`k\in\{0,...,nme-1\}`.
 
     - **areas**, Array of mesh elements areas, ``(nme,)`` *Numpy* array.
     
       ``areas[k]`` is the area of :math:`k`-th triangle, ``k in range(0,nme)``
       
   :param    N: number of points on each side of the square
   
   **optional parameter** : ``version=0`` or ``version=1``
       
   >>> from pyOptFEM.FEM2D import *
   >>> Th=getMesh('mesh/disk4-1-5.msh')
   >>> PlotMesh(Th)
   
   .. figure::  images/PlotMesh_disk4.png
     :width: 400px
     :scale: 100%
     :align:   center
     
     Visualisation of a *FreeFEM++* mesh (disk unit)
   """
   def __init__(self, meshfile,**kwargs):
      version=kwargs.get('version', 0)
      fp = open(meshfile, 'rt') 
      self.nq, self.nme, self.nbe = np.fromfile(fp, sep=" ", dtype=np.int32, count=3)
      data = np.fromfile(fp, sep=" ", dtype=np.float64, count=3*self.nq)
      data.shape = (self.nq,3)
      self.q=data[:,[0,1]]
      self.ql=np.int32(data[:,2])
      data = np.fromfile(fp, sep=" ", dtype=np.int32, count=4*self.nme)
      data.shape=(self.nme,4)
      self.me=data[:,[0,1,2]]-1
      self.mel=data[:,3]
      data = np.fromfile(fp, sep=" ", dtype=np.int32, count=3*self.nbe)
      data.shape=(self.nbe,3)
      self.be=data[:,[0,1]]-1
      self.bel=data[:,2]
      self.areas=ComputeAreaOpt(self.q,self.me)
      self.version=version
      if version==1:
        self.q=self.q.T
        self.me=self.me.T
      
class SquareMesh:
   """  Creates meshes of the unit square :math:`[0,1]\\times [0,1]`. Class attributes are :
   
     - **nq**, total number of mesh vertices (points), also denoted :math:`\\nq`.
     - **nme**, total number of mesh elements (triangles in 2d),
     - **version**, mesh structure version,
     - **q**, *Numpy* array of vertices coordinates, dimension ``(nq,2)`` *(version 0)* or ``(2,nq)`` *(version 1)*. 
     
       ``q[j]``  *(version 0)* or ``q[:,j]`` *(version 1)* are the two coordinates of the :math:`j`-th vertex, :math:`j\in\{0,..,nq-1\}`
       
     - **me**, *Numpy* connectivity array,  dimension ``(nme,3)`` *(version 0)* or ``(3,nme)`` *(version 1)*. 
       
       ``me[k]``  *(version 0)* or ``me[:,k]`` *(version 1)* are the storage index of the three vertices of the :math:`k`-th triangle in the array ``q`` of vertices coordinates, :math:`k\in\{0,...,nme-1\}`.
 
     - **areas**, Array of mesh elements areas, ``(nme,)`` *Numpy* array.
     
       ``areas[k]`` is the area of :math:`k`-th triangle, ``k in range(0,nme)``
       
   :param    N: number of points on each side of the square
   
   **optional parameter** : ``version=0`` or ``version=1``
   
   >>> from pyOptFEM.FEM2D import *
   >>> Th=SquareMesh(3)
   >>> Th.nme,Th.nq
   (18, 16)
   >>> Th.q
   array([[ 0.        ,  0.        ],
          [ 0.33333333,  0.        ],
          [ 0.66666667,  0.        ],
          [ 1.        ,  0.        ],
          [ 0.        ,  0.33333333],
          [ 0.33333333,  0.33333333],
          [ 0.66666667,  0.33333333],
          [ 1.        ,  0.33333333],
          [ 0.        ,  0.66666667],
          [ 0.33333333,  0.66666667],   
          [ 0.66666667,  0.66666667],
          [ 1.        ,  0.66666667],
          [ 0.        ,  1.        ],
          [ 0.33333333,  1.        ],
          [ 0.66666667,  1.        ],
          [ 1.        ,  1.        ]])
   >>> PlotMesh(Th)

   .. figure::  images/PlotMesh_SquareMesh.png
     :width: 400px
     :scale: 100%
     :align:   center
     
     SquareMesh(3) visualisation
   """    
   def __init__(self, N,**kwargs):
      version=kwargs.get('version', 0)
      t=np.linspace(0, 1, N+1)
      x,y= np.meshgrid(t, t)
      x.shape = ((N+1)*(N+1))
      y.shape = ((N+1)*(N+1))
      self.q=np.array([x[:],y[:]]).T
      self.nq=self.q.shape[0]
      tri=Delaunay(self.q)
      self.me=tri.vertices
      self.nme=self.me.shape[0]
      self.areas=ComputeAreaOpt(self.q,self.me)
      self.version=version
      if version==1:
        self.q=self.q.T
        self.me=self.me.T
      
def PlotMesh(M):
  if M.version==0:
    plt.triplot(M.q[:,0],M.q[:,1],M.me,'bo-')
  elif M.version==1:
    plt.triplot(M.q[0],M.q[1],M.me,'bo-')
  plt.axis('equal')
  plt.axis('off')
  plt.show()
