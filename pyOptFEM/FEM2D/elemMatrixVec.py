import numpy as np

def ElemMassMat2DP1Vec(areas):
  """  Computes all the element Mass matrices :math:`\\mathbb{M}^e(T_k)` for :math:`k\\in\{0,\hdots,\\nme-1\\}`
  
  :param areas: areas of all the mesh elements.
  :type areas: :math:`\\nme` *numpy* array of floats
  :returns: a one dimensional *numpy* array of size :math:`9 \\nme`
  """
  nme=areas.shape[0]
  Kg=np.zeros((nme,9))
  Kg[:,0]=Kg[:,4]=Kg[:,8]=areas/6
  Kg[:,1]=Kg[:,3]=Kg[:,2]=Kg[:,6]=Kg[:,5]=Kg[:,7]=areas/12 
  return Kg.reshape((9*nme))

def ComputeGradientVec(q,me,areas):
  coef=0.5/sqrt(areas)
  u=q[me[:,1]]-q[me[:,2]]
  G1=np.array([u[:,1]*coef,-u[:,0]*coef]) 
  u=q[me[:,2]]-q[me[:,0]]
  G2=np.array([u[:,1]*coef,-u[:,0]*coef])
  u=q[me[:,0]]-q[me[:,1]]
  G3=np.array([u[:,1]*coef,-u[:,0]*coef])
  return [G1,G2,G3]
  
# q (2,nq), me (3,nme)
def ComputeGradientVecTr(q,me,areas):
  nme=me.shape[1]
  #coef=1/(2*areas)
  G=np.ndarray(shape=(3,2,nme))
  u=q[:,me[1]]-q[:,me[2]]
  G[0,0]=u[1]
  G[0,1]=-u[0]
  u=q[:,me[2]]-q[:,me[0]]
  G[1,0]=u[1]
  G[1,1]=-u[0]
  u=q[:,me[0]]-q[:,me[1]]
  G[2,0]=u[1]
  G[2,1]=-u[0]
  return G
  
# q (nq,2), me (nme,3) 
def ElemStiffMat2DP1Vec(nme,q,me,areas): 
  """  Computes all the element stiffness matrices :math:`\\mathbb{S}^e(T_k)` for :math:`k\\in\{0,\hdots,\\nme-1\\}`
  
  :param nme: number of mesh elements,
  :type nme: int
  :param q: mesh vertices,
  :type q: :math:`\\nq\\times 2` *numpy* array of floats
  :param me: mesh connectivity,
  :type me: :math:`\\nme\\times 3` *numpy* array of integers
  :param areas: areas of all the mesh elements.
  :type areas: :math:`\\nme` *numpy* array of floats
  :returns: a one dimensional *numpy* array of size :math:`9 \\nme`
  """
  q1=q[me[:,0]]
  q2=q[me[:,1]]
  q3=q[me[:,2]]
  u=q2-q3
  v=q3-q1
  w=q1-q2 
  Kg=np.empty((nme,9))
  areas4=4*areas
  Kg[:,0]        =np.sum(u*u,axis=1)/areas4
  Kg[:,1]=Kg[:,3]=np.sum(u*v,axis=1)/areas4
  Kg[:,2]=Kg[:,6]=np.sum(u*w,axis=1)/areas4
  Kg[:,4]        =np.sum(v*v,axis=1)/areas4
  Kg[:,5]=Kg[:,7]=np.sum(v*w,axis=1)/areas4
  Kg[:,8]        =np.sum(w*w,axis=1)/areas4
  return Kg.reshape((9*nme))  
  
# q (2,nq), me (3,nme) 
def ElemStiffElasMatBaVec2DP1(nme,q,me,areas,L,M,**kwargs):
  """  Computes all the element elastic stiffness matrices :math:`\\mathbb{K}^e(T_k)` for :math:`k\\in\{0,\hdots,\\nme-1\\}`   in local *alternate* basis.
  
  :param nme: number of mesh elements,
  :type nme: int
  :param q: mesh vertices,
  :type q: ``(2,nq)`` *numpy* array of floats
  :param me: mesh connectivity,
  :type me: ``(3,nme)`` *numpy* array of integers
  :param areas: areas of all the mesh elements.
  :type areas: ``(nme,)`` *numpy* array of floats
  :param L: the  :math:`\\lambda` Lame parameter,
  :type L: float
  :param M: the  :math:`\\mu` Lame parameter.
  :type M: float
  :returns: a ``(36*nme,)`` *numpy* array of floats.
  """  
  memory=kwargs.get('memory',False)
  ndf2=36;
  G=ComputeGradientVecTr(q,me,areas)
  coef=2*np.sqrt(areas)
  for il in range(0,3):
    for i in range(0,2):
      G[il,i]=G[il,i]/coef
  Kg=np.ndarray(shape=(ndf2,nme))  
  T1=G[0,0]**2
  T2=G[0,1]**2
  C=M*(T1+ T2)
  if memory:
    mem=G.nbytes+T1.nbytes+T2.nbytes+C.nbytes+coef.nbytes
    return mem
  Kg[0]=(L+M)*T1+C
  Kg[7]=(L+M)*T2+C
  
  T1=G[1,0]**2
  T2=G[1,1]**2
  C=M*(T1+ T2)
  Kg[14]=(L+M)*T1+C
  Kg[21]=(L+M)*T2+C
  
  T1=G[2,0]**2
  T2=G[2,1]**2
  C=M*(T1+ T2)
  Kg[28]=(L+M)*T1+C
  Kg[35]=(L+M)*T2+C
  
  T1=G[0,0]*G[1,0]
  T2=G[0,1]*G[1,1]
  C=M*(T1+ T2)
  Kg[2]=Kg[12]=(L+M)*T1+C
  Kg[9]=Kg[19]=(L+M)*T2+C
  
  T1=G[0,0]*G[1,1]
  T2=G[0,1]*G[1,0]
  Kg[3]=Kg[18]=L*T1+M*T2
  Kg[8]=Kg[13]=L*T2+M*T1
  
  T1=G[0,0]*G[2,0]
  T2=G[0,1]*G[2,1]
  C=M*(T1+ T2)
  Kg[4]=Kg[24]=(L+M)*T1+C
  Kg[11]=Kg[31]=(L+M)*T2+C
  
  T1=G[1,0]*G[2,0]
  T2=G[1,1]*G[2,1]
  C=M*(T1+ T2)
  Kg[16]=Kg[26]=(L+M)*T1+C
  Kg[23]=Kg[33]=(L+M)*T2+C
  
  Kg[1] =Kg[6] =(L+M)*G[0,0]*G[0,1] 
  Kg[15]=Kg[20]=(L+M)*G[1,0]*G[1,1]
  Kg[29]=Kg[34]=(L+M)*G[2,0]*G[2,1]
  
  T1=G[0,0]*G[2,1]
  T2=G[0,1]*G[2,0]
  Kg[5]=Kg[30]=L*T1+M*T2
  Kg[10]=Kg[25]=L*T2+M*T1
  
  T1=G[1,0]*G[2,1]
  T2=G[1,1]*G[2,0]
  Kg[17]=Kg[32]=L*T1+M*T2
  Kg[22]=Kg[27]=L*T2+M*T1

  return Kg.reshape((ndf2*nme))

  
def ElemStiffElasMatBbVec2DP1(nme,q,me,areas,L,M,**kwargs):
  """  Computes all the element elastic stiffness matrices :math:`\\mathbb{K}^e(T_k)` 
  for :math:`k\\in\{0,\hdots,\\nme-1\\}` in local *block* basis.
  
  :param nme: number of mesh elements,
  :type nme: int
  :param q: mesh vertices,
  :type q: ``(2,nq)`` *numpy* array of floats
  :param me: mesh connectivity,
  :type me: ``(3,nme)`` *numpy* array of integers
  :param areas: areas of all the mesh elements.
  :type areas: ``(nme,)`` *numpy* array of floats
  :param L: the  :math:`\\lambda` Lame parameter,
  :type L: float
  :param M: the  :math:`\\mu` Lame parameter.
  :type M: float
  :returns: a ``(36*nme,)`` *numpy* array of floats.
  """    
  memory=kwargs.get('memory',False)
  ndf2=36
  G=ComputeGradientVecTr(q,me,areas)
  coef=2*np.sqrt(areas)
  for il in range(0,3):
    for i in range(0,2):
      G[il,i]=G[il,i]/coef
  Kg=np.ndarray(shape=(ndf2,nme))   
  
  T1=G[0,0]**2;T2=G[0,1]**2;C=M*(T1+ T2)
  if memory:
    mem=G.nbytes+T1.nbytes+T2.nbytes+C.nbytes+coef.nbytes
    return mem
  
  Kg[0]=(L+M)*T1+C
  Kg[21]=(L+M)*T2+C
  
  T1=G[1,0]**2;T2=G[1,1]**2;C=M*(T1+ T2)
  Kg[7]=(L+M)*T1+C
  Kg[28]=(L+M)*T2+C
  
  T1=G[2,0]**2;T2=G[2,1]**2;C=M*(T1+ T2)
  Kg[14]=(L+M)*T1+C
  Kg[35]=(L+M)*T2+C
  
  T1=G[0,0]*G[1,0];T2=G[0,1]*G[1,1];C=M*(T1+ T2)
  Kg[1]=Kg[6]=(L+M)*T1+C
  Kg[22]=Kg[27]=(L+M)*T2+C
  
  T1=G[0,0]*G[2,0];T2=G[0,1]*G[2,1];C=M*(T1+ T2)
  Kg[2]=Kg[12]=(L+M)*T1+C
  Kg[23]=Kg[33]=(L+M)*T2+C
  
  T1=G[1,0]*G[2,0];T2=G[1,1]*G[2,1];C=M*(T1+ T2)
  Kg[8]=Kg[13]=(L+M)*T1+C
  Kg[29]=Kg[34]=(L+M)*T2+C  
  
  Kg[3] =Kg[18]=(L+M)*G[0,0]*G[0,1]
  Kg[10]=Kg[25]=(L+M)*G[1,0]*G[1,1]
  Kg[17]=Kg[32]=(L+M)*G[2,0]*G[2,1]
  
  T1=G[0,0]*G[1,1];T2=G[0,1]*G[1,0];
  Kg[4]=Kg[24]=L*T1+M*T2
  Kg[9]=Kg[19]=L*T2+M*T1
  
  T1=G[0,0]*G[2,1];T2=G[0,1]*G[2,0]
  Kg[5]=Kg[30]=L*T1+M*T2
  Kg[15]=Kg[20]=L*T2+M*T1
  
  T1=G[1,0]*G[2,1];T2=G[1][1]*G[2][0]
  Kg[11]=Kg[31]=L*T1+M*T2
  Kg[16]=Kg[26]=L*T2+M*T1
  
  return Kg.reshape((ndf2*nme))
