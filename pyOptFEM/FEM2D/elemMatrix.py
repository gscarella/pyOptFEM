import numpy as np

# Function ElemMassMat2DP1 perform 3.5 faster with global MassMat
MassMat = np.array([[2.,1.,1.],[1.,2.,1.],[1.,1.,2.]])/12.0

def ElemMassMat2DP1(area):
  """  Computes the element mass matrix :math:`\\mathbb{M}^e(T)` for a given triangle :math:`T` of area :math:`|T|`
  
  :param area: area of the triangle.
  :type area: float
  :returns: :math:`3 \\times 3` *numpy* array of floats.
  """
  return area*MassMat

def ElemStiffMat2DP1(q1,q2,q3,area):
  """ Computes the element stiffness matrix :math:`\\mathbb{S}^e(T)` for a given triangle :math:`T`
  
  :param q1,q2,q3: the three vertices of the triangle,
  :type q1,q2,q3: :math:`2 \\times 1` *numpy* array
  :param area: area of the triangle.
  :type area: float
  :returns: 
  :type: :math:`3 \\times 3` *numpy* array of floats.
  """
  M=np.array([q2-q3, q3-q1, q1-q2])
  return (1/(4*area))*np.dot(M,M.T)
  

def Hooke(la,mu):
  """ Returns the elasticity tensor, :math:`\\mathbb{H}`,  obtained from Hooke's law with an isotropic material. 
  It's defined with the Lame parameters :math:`\\lambda` and :math:`\\mu` by
  
  .. math::
    \\mathbb{H} =\\begin{pmatrix} 
                   \\lambda+2\\mu & \\lambda & 0\\\\ 
                   \\lambda & \\lambda+2\\mu & 0\\\\ 
                   0 & 0 & \\mu 
                 \\end{pmatrix}
    
  :param la: the  :math:`\\lambda` Lame parameter,
  :type la: float
  :param mu: the  :math:`\\mu` Lame parameter.
  :type mu: float
  :returns: Elasticity tensor, :math:`\\mathbb{H}`,
  :type: :math:`3 \\times 3` *numpy* array of floats.
  """
  return np.array([[la+2*mu,la,0],[la,la+2*mu,0],[0,0,mu]]);

  
def ElemStiffElasMat2DP1Ba(ql,area,H):
  """ Returns the element elastic stiffness matrix :math:`\\mathbb{K}^e(T)` 
  for a given triangle :math:`T`  in the local *alternate* basis :math:`\\mathcal{B}_a`
  
  :param ql: contains the three vertices of the triangle : ``ql[0]``, ``ql[1]`` and  ``ql[2]``,
  :type ql: :math:`3 \\times 2` *numpy* array
  :param area: area of the triangle ,
  :type area: float
  :param H: Elasticity tensor, :math:`\\mathbb{H}`.
  :type H: :math:`3 \\times 3` *numpy* array
  :returns: :math:`\\mathbb{K}^e(T)` in :math:`\\mathcal{B}_a` basis.
  :type: :math:`6 \\times 6` *numpy* array of floats.
  """
  u=ql[1]-ql[2]
  v=ql[2]-ql[0] 
  w=ql[0]-ql[1]
  B=np.array([[u[1],0,v[1],0,w[1],0],
             [0,-u[0],0,-v[0],0,-w[0]],
             [-u[0],u[1],-v[0],v[1],-w[0],w[1]]])
  return np.dot(B.T,np.dot(H,B))/(4*area)
  
def ElemStiffElasMat2DP1Bb(ql,area,H):
  """ Returns the element elastic stiffness matrix :math:`\\mathbb{K}^e(T)` 
  for a given triangle :math:`T`  in the local *block* basis :math:`\\mathcal{B}_b`
  
  :param ql: contains the three vertices of the triangle : ``ql[0]``, ``ql[1]`` and  ``ql[2]``,
  :type ql: :math:`3 \\times 2` *numpy* array
  :param area: area of the triangle, 
  :type area: float
  :param H: Elasticity tensor, :math:`\\mathbb{H}`.
  :type H: :math:`3 \\times 3` *numpy* array
  :returns: :math:`\\mathbb{K}^e(T)` in :math:`\\mathcal{B}_b` basis.
  :type: :math:`6 \\times 6` *numpy* array of floats
  """
  u=ql[1]-ql[2]
  v=ql[2]-ql[0] 
  w=ql[0]-ql[1]
  B=np.array([[u[1],v[1],w[1],0,0,0],
             [0,0,0,-u[0],-v[0],-w[0]],
             [-u[0],-v[0],-w[0],u[1],v[1],w[1]]])
  return np.dot(B.T,np.dot(H,B))/(4*area)

