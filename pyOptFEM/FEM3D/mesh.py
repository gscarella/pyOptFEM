import numpy as np
from scipy import sparse
from scipy.spatial import Delaunay
from .toolsVec import *


class getMesh:
   """ Reads a *medit* mesh from file ``meshfile``. Class attributes are :
   
     - **nq**, total number of mesh vertices (points), also denoted :math:`\\nq`.
     - **nme**, total number of mesh elements (tetrahedra in 3d),
     - **version**, mesh structure version,
     - **q**, *Numpy* array of vertices coordinates, dimension ``(nq,3)`` *(version 0)* or ``(3,nq)`` *(version 1)*. 
     
       ``q[j]``  *(version 0)* or ``q[:,j]`` *(version 1)* are the three coordinates of the :math:`j`-th vertex, :math:`j\in\{0,..,nq-1\}`
       
     - **me**, *Numpy* connectivity array,  dimension ``(nme,4)`` *(version 0)* or ``(4,nme)`` *(version 1)*. 
       
       ``me[k]``  *(version 0)* or ``me[:,k]`` *(version 1)* are the storage index of the four vertices of the :math:`k`-th tetrahedron in the array ``q`` of vertices coordinates, :math:`k\in\{0,...,nme-1\}`.
 
     - **volumes**, Array of mesh elements volumes, ``(nme,)`` *Numpy* array.
     
       ``volumes[k]`` is the volume of :math:`k`-th tetrahedron, ``k in range(0,nme)``
       
   :param    meshfile: *medit* mesh file
   
   **optional parameter** : ``version=0`` or ``version=1``
   """
   def __init__(self, filename,**kwargs):
      version=kwargs.get('version', 0)
      fp = open(filename, 'rt')
      line=''
      while (line.find('Vertices')==-1):
        line = fp.readline()
      self.nq = np.fromfile(fp, sep=" ", dtype=np.int32, count=1)[0]
      data = np.fromfile(fp, sep=" ", dtype=np.float64, count=4*self.nq)
      data.shape = (self.nq,4)
      self.q=data[:,[0,1,2]]
      self.ql=np.int32(data[:,3])
      line=''
      while (line.find('Triangles')==-1):
        line = fp.readline()
      self.nf = np.fromfile(fp, sep=" ", dtype=np.int32, count=1)[0]
      data = np.fromfile(fp, sep=" ", dtype=np.int32, count=4*self.nf)
      data.shape = (self.nf,4)
      self.f=data[:,[0,1,2]]-1
      self.fl=data[:,3]
      line=''
      while (line.find('Tetrahedra')==-1):
        line = fp.readline()
      self.nme = np.fromfile(fp, sep=" ", dtype=np.int32, count=1)[0]
      data = np.fromfile(fp, sep=" ", dtype=np.int32, count=5*self.nme)
      data.shape = (self.nme,5)
      self.me=data[:,[0,1,2,3]]-1
      self.mel=data[:,4]
      fp.close()
      self.volumes=ComputeVolumesOpt(self.q,self.me)
      self.version=version
      if version==1:
        self.q=self.q.T
        self.me=self.me.T
         
class CubeMesh:
   """ Creates meshes of the unit cube :math:`[0,1]^3`. Class attributes are :
   
     - **nq**, total number of mesh vertices (points), also denoted :math:`\\nq`.
     - **nme**, total number of mesh elements (tetrahedra in 3d),
     - **version**, mesh structure version,
     - **q**, *Numpy* array of vertices coordinates, dimension ``(nq,3)`` *(version 0)* or ``(3,nq)`` *(version 1)*. 
     
       ``q[j]``  *(version 0)* or ``q[:,j]`` *(version 1)* are the three coordinates of the :math:`j`-th vertex, :math:`j\in\{0,..,nq-1\}`
       
     - **me**, *Numpy* connectivity array,  dimension ``(nme,4)`` *(version 0)* or ``(4,nme)`` *(version 1)*. 
       
       ``me[k]``  *(version 0)* or ``me[:,k]`` *(version 1)* are the storage index of the four vertices of the :math:`k`-th tetrahedron in the array ``q`` of vertices coordinates, :math:`k\in\{0,...,nme-1\}`.
 
     - **volumes**, Array of mesh elements volumes, ``(nme,)`` *Numpy* array.
     
       ``volumes[k]`` is the volume of :math:`k`-th tetrahedron, ``k in range(0,nme)``
       
   :param    N: number of points on each edge of the cube
   
   **optional parameter** : ``version=0`` or ``version=1``
   """  
   def __init__(self, N,**kwargs):
      version=kwargs.get('version', 0)
      t=np.linspace(0, 1, N+1)
      x,y,z= np.meshgrid(t, t, t)
      x.shape = ((N+1)*(N+1)*(N+1))
      y.shape = ((N+1)*(N+1)*(N+1))
      z.shape = ((N+1)*(N+1)*(N+1))
      self.q=np.array([x[:],z[:],y[:]]).T
      self.nq=self.q.shape[0]
      tri=Delaunay(self.q)
      self.me=tri.vertices
      self.nme=self.me.shape[0]
      self.volumes=ComputeVolumesOpt(self.q,self.me)
      Ix=np.where(self.volumes==0)[0]
      if Ix.shape[0]>0:
        Ii=np.setdiff1d(range(0,self.nme),Ix)
        self.me=self.me[Ii]
        self.nme=self.me.shape[0]
        self.volumes=self.volumes[Ii]
      self.version=version
      if version==1:
        self.q=self.q.T
        self.me=self.me.T

# 4) Plot functions
  
def PlotTetra(ql):
  import matplotlib as mpl
  from mpl_toolkits.mplot3d import Axes3D
  import numpy as np
  import matplotlib.pyplot as plt
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  Q=ql.T
  ax.plot(Q[0], Q[1], Q[2])
  Q=array([ql[3],ql[1]]).T
  ax.plot(Q[0], Q[1], Q[2])
  Q=array([ql[0],ql[2]]).T
  ax.plot(Q[0], Q[1], Q[2])
  Q=array([ql[0],ql[3]]).T
  ax.plot(Q[0], Q[1], Q[2])
  plt.show()
  
def PlotTetraAx(ax,ql,color):
  Q=ql.T
  ax.plot(Q[0], Q[1], Q[2],color)
  Q=array([ql[3],ql[1]]).T
  ax.plot(Q[0], Q[1], Q[2],color)
  Q=array([ql[0],ql[2]]).T
  ax.plot(Q[0], Q[1], Q[2],color)
  Q=array([ql[0],ql[3]]).T
  ax.plot(Q[0], Q[1], Q[2],color)
  
def PlotMesh3D(Th,kk):
  import matplotlib as mpl
  from mpl_toolkits.mplot3d import Axes3D
  import numpy as np
  import matplotlib.pyplot as plt
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  nme=Th.nme
  for k in range(0,kk):
    PlotTetraAx(ax,Th.q[Th.me[k]],'b')
  for k in range(kk+1,nme):
    PlotTetraAx(ax,Th.q[Th.me[k]],'b')
  PlotTetraAx(ax,Th.q[Th.me[kk]],'r')
  plt.show()
