import numpy as np

# Function ElemMassMat3DP1 perform ? faster with global MassMat
MassMat3D = np.array([[2.,1.,1.,1.],[1.,2.,1.,1.],[1.,1.,2.,1.],[1.,1.,1.,2.]])/20.0

# 1) Mass Matrix Assembly
def ElemMassMat3DP1(V):
  """  Computes  the element mass matrix :math:`\\mathbb{M}^e(T)` for a given tetrahedron :math:`T` of volume :math:`|T|`
  
  :param V: volume of the tetrahedron.
  :type V: float
  :returns: :math:`4 \\times 4` *numpy* array of floats.
  """
  return V*MassMat3D
  
# 2) Stiff Matrix Assembly
def ComputeGradient(ql):
  D12=ql[0]-ql[1];D13=ql[0]-ql[2];D14=ql[0]-ql[3]
  D23=ql[1]-ql[2];D24=ql[1]-ql[3];D34=ql[2]-ql[3]
  G=np.zeros((3,4))
  G[0,0]=-D23[1]*D24[2] + D23[2]*D24[1]
  G[1,0]= D23[0]*D24[2] - D23[2]*D24[0]
  G[2,0]=-D23[0]*D24[1] + D23[1]*D24[0]
  G[0,1]= D13[1]*D14[2] - D13[2]*D14[1]
  G[1,1]=-D13[0]*D14[2] + D13[2]*D14[0]
  G[2,1]= D13[0]*D14[1] - D13[1]*D14[0]
  G[0,2]=-D12[1]*D14[2] + D12[2]*D14[1]
  G[1,2]= D12[0]*D14[2] - D12[2]*D14[0]
  G[2,2]=-D12[0]*D14[1] + D12[1]*D14[0]
  G[0,3]= D12[1]*D13[2] - D12[2]*D13[1]
  G[1,3]=-D12[0]*D13[2] + D12[2]*D13[0]
  G[2,3]= D12[0]*D13[1] - D12[1]*D13[0]
  return G
  

def ElemStiffMat3DP1(ql,volume):
  """ Computes the element stiffness matrix :math:`\\mathbb{S}^e(T)` for a given tetrahedron :math:`T`
  
  :param ql: the four vertices of the tetrahedron,
  :type ql: :math:`2 \\times 4` *numpy* array
  :param volume: volume of the tetrahedron.
  :type volume: float
  :returns: 
  :type: :math:`4 \\times 4` *numpy* array of floats.
  """
  G=ComputeGradient(ql)
  return (1/(36*volume))*np.dot(G.T,G)     
       
# 3) Stiff Elas Matrix Assembly
def ElemStiffElasMatBa3DP1(ql,V,C):
  """ Returns the element elastic stiffness matrix :math:`\\mathbb{K}^e(T)` 
  for a given tetrahedron :math:`T`  in the local *alternate* basis :math:`\\mathcal{B}_a`
  
  :param ql: contains the four vertices of the tetrahedron,
  :type ql: :math:`4 \\times 2` *numpy* array
  :param V: volume of the tetrahedron
  :type V: float
  :param H: Elasticity tensor, :math:`\\mathbb{H}`.
  :type H: :math:`6 \\times 6` *numpy* array
  :returns: :math:`\\mathbb{K}^e(T)` in :math:`\\mathcal{B}_a` basis.
  :type: :math:`12 \\times 12` *numpy* array of floats.
  """
  G=ComputeGradient(ql)
  B=np.zeros((12,6))
  k=0;
  for il in range(0,4):
    B[  k,[0,3,4]]=G[:,il]
    B[1+k,[3,1,5]]=G[:,il]
    B[2+k,[4,5,2]]=G[:,il]
    k+=3
  return np.dot(np.dot(B,C),B.T)/(36*V)  
  
def ElemStiffElasMatBb3DP1(ql,V,C):
  """ Returns the element elastic stiffness matrix :math:`\\mathbb{K}^e(T)` 
  for a given tetrahedron :math:`T`  in the local *block* basis :math:`\\mathcal{B}_b`
  
  :param ql: contains the four vertices of the tetrahedron,
  :type ql: :math:`4 \\times 2` *numpy* array
  :param V: volume of the tetrahedron
  :type V: float
  :param H: Elasticity tensor, :math:`\\mathbb{H}`.
  :type H: :math:`6 \\times 6` *numpy* array
  :returns: :math:`\\mathbb{K}^e(T)` in :math:`\\mathcal{B}_b` basis.
  :type: :math:`12 \\times 12` *numpy* array of floats.
  """
  G=ComputeGradient(ql)
  B=np.zeros((12,6))
  for il in range(0,4):
    B[il  ,[0,3,4]]=G[:,il]
    B[il+4,[3,1,5]]=G[:,il]
    B[il+8,[4,5,2]]=G[:,il]
  return np.dot(np.dot(B,C),B.T)/(36*V) 

