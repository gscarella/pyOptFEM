import numpy as np
  
def ElemMassMat3DP1Vec(nme,volumes):
  """  Computes all the element Mass matrices :math:`\\mathbb{M}^e(T_k)` for :math:`k\\in\{0,\hdots,\\nme-1\\}`
  
  :param volumes: volumes of all the mesh elements.
  :type volumes: :math:`\\nme` *numpy* array of floats
  :returns: a one dimensional *numpy* array of size :math:`16 \\nme`
  """
  Kg=np.zeros((volumes.shape[0],16))
  Kg[:,0]=Kg[:,5]=Kg[:,10]=Kg[:,15]=volumes/10
  Kg[:,1]=Kg[:,2]=Kg[:,3]=Kg[:,4]=Kg[:,6]=Kg[:,7]=Kg[:,8]=Kg[:,9]=Kg[:,11]=Kg[:,12]=Kg[:,13]=Kg[:,14]=volumes/20
  return Kg.reshape((16*nme))

# tableau q et me transposes
def ComputeGradientVecTr(q,me): # la plus rapide avec T4
  nme=me.shape[1]
  q1=q[:,me[0]];q2=q[:,me[1]];q3=q[:,me[2]];q4=q[:,me[3]]
  D12=q1-q2;D13=q1-q3;D14=q1-q4
  D23=q2-q3;D24=q2-q4
  #D34=q3-q4;
  G=np.ndarray(shape=(4,3,nme))
  G[0,0] = D23[2]*D24[1] - D23[1]*D24[2]
  G[0,1] = D23[0]*D24[2] - D23[2]*D24[0]
  G[0,2] = D23[1]*D24[0] - D23[0]*D24[1]
  G[1,0] = D13[1]*D14[2] - D13[2]*D14[1]
  G[1,1] = D13[2]*D14[0] - D13[0]*D14[2]
  G[1,2] = D13[0]*D14[1] - D13[1]*D14[0]
  G[2,0] = D12[2]*D14[1] - D12[1]*D14[2] 
  G[2,1] = D12[0]*D14[2] - D12[2]*D14[0]
  G[2,2] = D12[1]*D14[0] - D12[0]*D14[1]
  G[3,0] = D12[1]*D13[2] - D12[2]*D13[1]
  G[3,1] = D12[2]*D13[0] - D12[0]*D13[2]
  G[3,2] = D12[0]*D13[1] - D12[1]*D13[0]
  return G 

  # me (4,nme) array,  q (3,nq) array  
def ElemStiffMat3DP1Vec(nme,q,me,volumes):
  r"""  Computes all the element stiffness matrices :math:`\mathbb{S}^e(T_k)` for :math:`k\in\{0,\hdots,\nme-1\}`
  
  :param nme: number of mesh elements,
  :type nme: int
  :param q: mesh vertices,
  :type q: :math:`3\times \nq` *numpy* array of floats
  :param me: mesh connectivity,
  :type me: :math:`4 \times\nme` *numpy* array of integers
  :param areas: areas of all the mesh elements.
  :type areas: :math:`\nme` *numpy* array of floats
  :returns: a one dimensional *numpy* array of size :math:`9 \nme`
  """
  G=ComputeGradientVecTr(q,me)
  vol36=36*volumes;
  Kg=np.ndarray(shape=(16,nme))
  Kg[0]        =np.sum(G[0]*G[0],axis=0)/vol36
  Kg[1]=Kg[4]  =np.sum(G[0]*G[1],axis=0)/vol36
  Kg[2]=Kg[8]  =np.sum(G[0]*G[2],axis=0)/vol36
  Kg[3]=Kg[12] =np.sum(G[0]*G[3],axis=0)/vol36
  Kg[5]        =np.sum(G[1]*G[1],axis=0)/vol36
  Kg[6]=Kg[9]  =np.sum(G[1]*G[2],axis=0)/vol36
  Kg[7]=Kg[13] =np.sum(G[1]*G[3],axis=0)/vol36
  Kg[10]       =np.sum(G[2]*G[2],axis=0)/vol36
  Kg[11]=Kg[14]=np.sum(G[2]*G[3],axis=0)/vol36
  Kg[15]       =np.sum(G[3]*G[3],axis=0)/vol36
  return Kg.reshape((16*nme))
  
def ElemStiffElasMatBa3DP1Vec(nme,q,me,volumes,la,mu):
  r"""  Computes all the element elastic stiffness  matrices :math:`\mathbb{K}^e(T_k)` for :math:`k\in\{0,\hdots,\nme-1\}` 
  in local *alternate* basis.
  
  :param nme: number of mesh elements,
  :type nme: int
  :param q: mesh vertices,
  :type q: ``(3,nq)`` *numpy* array of floats
  :param me: mesh connectivity,
  :type me: ``(4,nme)`` *numpy* array of integers
  :param volumes: volumes of all the mesh elements.
  :type volumes: ``(nme,)`` *numpy* array of floats
  :param la: the  :math:`\\lambda` Lame parameter,
  :type la: float
  :param mu: the  :math:`\\mu` Lame parameter.
  :type mu: float
  :returns: a ``(144*nme,)`` *numpy* array of floats.
  """ 
  ndf2=144;
  G=ComputeGradientVecTr(q,me)
  coef=6*np.sqrt(volumes)
  for il in range(0,4):
    for i in range(0,3):
      G[il,i]=G[il,i]/coef
  #Kg=zeros((ndf2,nme))
  Kg=np.ndarray(shape=(ndf2,nme))
  T1=G[0,0]**2
  T2=G[0,1]**2
  T3=G[0,2]**2
  C=mu*(T1+ T2 + T3)
  Kg[0] =(la + mu)*T1 + C
  Kg[13]=(la + mu)*T2 + C
  Kg[26]=(la + mu)*T3 + C
  
  T1=G[1,0]**2
  T2=G[1,1]**2
  T3=G[1,2]**2
  C=mu*(T1+ T2 + T3)
  Kg[39]=(la + mu)*T1 + C
  Kg[52]=(la + mu)*T2 + C
  Kg[65]=(la + mu)*T3 + C
  
  T1=G[2,0]**2
  T2=G[2,1]**2
  T3=G[2,2]**2
  C=mu*(T1+ T2 + T3)
  Kg[78]=(la + mu)*T1 + C
  Kg[91]=(la + mu)*T2 + C
  Kg[104]=(la + mu)*T3 + C
  
  T1=G[3,0]**2
  T2=G[3,1]**2
  T3=G[3,2]**2
  C=mu*(T1+ T2 + T3)
  Kg[117]=(la + mu)*T1 + C
  Kg[130]=(la + mu)*T2 + C
  Kg[143]=(la + mu)*T3 + C
  
  Kg[1]=Kg[12]=(la+mu)*G[0,0]*G[0,1]
  Kg[2]=Kg[24]=(la+mu)*G[0,0]*G[0,2]
  
  T1=G[0,0]*G[1,0]
  T2=G[0,1]*G[1,1]
  T3=G[0,2]*G[1,2]
  C=mu*(T1+ T2 + T3)
  Kg[3] =Kg[36]=(la + mu)*T1 + C
  Kg[16]=Kg[49]=(la + mu)*T2 + C
  Kg[29]=Kg[62]=(la + mu)*T3 + C
  
  T1=G[0,0]*G[1,1]
  T2=G[0,1]*G[1,0]
  Kg[4] =Kg[48]=la*T1+mu*T2
  Kg[15]=Kg[37]=la*T2+mu*T1
  
  T1=G[0,0]*G[1,2]
  T2=G[0,2]*G[1,0]
  Kg[5] =Kg[60]=la*T1+mu*T2
  Kg[27]=Kg[38]=la*T2+mu*T1
  
  T1=G[0,0]*G[2,0]
  T2=G[0,1]*G[2,1]
  T3=G[0,2]*G[2,2]
  C=mu*(T1+ T2 + T3)
  Kg[6] =Kg[72]=(la + mu)*T1 + C
  Kg[19]=Kg[85]=(la + mu)*T2 + C
  Kg[32]=Kg[98]=(la + mu)*T3 + C
  
  T1=G[0,0]*G[2,1]
  T2=G[0,1]*G[2,0]
  Kg[7] =Kg[84]= la*T1+mu*T2
  Kg[18]=Kg[73]= la*T2+mu*T1
  
  T1=G[0,0]*G[2,2]
  T2=G[0,2]*G[2,0]
  Kg[8] =Kg[96]= la*T1+mu*T2
  Kg[30]=Kg[74]= la*T2+mu*T1
  
  T1=G[0,0]*G[3,0]
  T2=G[0,1]*G[3,1]
  T3=G[0,2]*G[3,2]
  C=mu*(T1+ T2 + T3)
  Kg[9]=Kg[108]=(la + mu)*T1 + C
  Kg[22]=Kg[121]=(la + mu)*T2 + C
  Kg[35]=Kg[134]=(la + mu)*T3 + C
  
  T1=G[0,0]*G[3,1]
  T2=G[0,1]*G[3,0]
  Kg[10]=Kg[120]= la*T1+mu*T2
  Kg[21]=Kg[109]= la*T2+mu*T1
  
  T1=G[0,0]*G[3,2]
  T2=G[0,2]*G[3,0]
  Kg[11]=Kg[132]= la*T1+mu*T2
  Kg[33]=Kg[110]= la*T2+mu*T1
  
  Kg[14]=Kg[25]=(la+mu)*G[0,1]*G[0,2]
  
  T1=G[0,1]*G[1,2]
  T2=G[0,2]*G[1,1]
  Kg[17]=Kg[61]= la*T1+mu*T2
  Kg[28]=Kg[50]= la*T2+mu*T1
  
  T1=G[0,1]*G[2,2]
  T2=G[0,2]*G[2,1]
  Kg[20]=Kg[97]= la*T1+mu*T2
  Kg[31]=Kg[86]= la*T2+mu*T1
  
  T1=G[0,1]*G[3,2]
  T2=G[0,2]*G[3,1]
  Kg[23]=Kg[133]= la*T1+mu*T2
  Kg[34]=Kg[122]= la*T2+mu*T1
  
  Kg[40]=Kg[51]= (la+mu)*G[1,0]*G[1,1]
  Kg[41]=Kg[63]= (la+mu)*G[1,0]*G[1,2]
  
  T1=G[1,0]*G[2,0]
  T2=G[1,1]*G[2,1]
  T3=G[1,2]*G[2,2]
  C=mu*(T1+ T2 + T3)
  Kg[42]=Kg[75]=(la + mu)*T1 + C
  Kg[55]=Kg[88]=(la + mu)*T2 + C
  Kg[68]=Kg[101]=(la + mu)*T3 + C
  
  T1=G[1,0]*G[2,1]
  T2=G[1,1]*G[2,0]
  Kg[43]=Kg[87]= la*T1+mu*T2
  Kg[54]=Kg[76]= la*T2+mu*T1
  
  T1=G[1,0]*G[2,2]
  T2=G[1,2]*G[2,0]
  Kg[44]=Kg[99]= la*T1+mu*T2
  Kg[66]=Kg[77]= la*T2+mu*T1
  
  T1=G[1,0]*G[3,0]
  T2=G[1,1]*G[3,1]
  T3=G[1,2]*G[3,2]
  C=mu*(T1+ T2 + T3)
  Kg[45]=Kg[111]=(la + mu)*T1 + C
  Kg[58]=Kg[124]=(la + mu)*T2 + C
  Kg[71]=Kg[137]=(la + mu)*T3 + C
  
  T1=G[1,0]*G[3,1]
  T2=G[1,1]*G[3,0]
  Kg[46]=Kg[123]= la*T1+mu*T2
  Kg[57]=Kg[112]= la*T2+mu*T1
  
  T1=G[1,0]*G[3,2]
  T2=G[1,2]*G[3,0]
  Kg[47]=Kg[135]= la*T1+mu*T2
  Kg[69]=Kg[113]= la*T2+mu*T1
  
  Kg[53]=Kg[64]= (la+mu)*G[1,1]*G[1,2]
  
  T1=G[1,1]*G[2,2]
  T2=G[1,2]*G[2,1]
  Kg[56]=Kg[100]= la*T1+mu*T2
  Kg[67]=Kg[89]= la*T2+mu*T1
  
  T1=G[1,1]*G[3,2]
  T2=G[1,2]*G[3,1]
  Kg[59]=Kg[136]= la*T1+mu*T2
  Kg[70]=Kg[125]= la*T2+mu*T1
  
  Kg[79]=Kg[90]=(la+mu)*G[2,0]*G[2,1]
  
  Kg[80]=Kg[102]=(la+mu)*G[2,0]*G[2,2]
  
  T1=G[2,0]*G[3,0]
  T2=G[2,1]*G[3,1]
  T3=G[2,2]*G[3,2]
  C=mu*(T1+ T2 + T3)
  Kg[81] =Kg[114]=(la + mu)*T1 + C
  Kg[94] =Kg[127]=(la + mu)*T2 + C
  Kg[107]=Kg[140]=(la + mu)*T3 + C
  #
  T1=G[2,0]*G[3,1]
  T2=G[2,1]*G[3,0]
  Kg[82]=Kg[126]= la*T1+mu*T2
  Kg[93]=Kg[115]= la*T2+mu*T1
  
  T1=G[2,0]*G[3,2]
  T2=G[2,2]*G[3,0]
  Kg[83] =Kg[138]= la*T1+mu*T2
  Kg[105]=Kg[116]= la*T2+mu*T1
  
  Kg[92]=Kg[103]=(la+mu)*G[2,1]*G[2,2]
  
  T1=G[2,1]*G[3,2]
  T2=G[2,2]*G[3,1]
  Kg[95] = Kg[139]=la*T1+mu*T2
  Kg[106]= Kg[128]=la*T2+mu*T1
  
  Kg[118]=Kg[129]=(la + mu)*G[3,0]*G[3,1]
  Kg[119]=Kg[141]=(la + mu)*G[3,0]*G[3,2]
  Kg[131]=Kg[142]=(la + mu)*G[3,1]*G[3,2]
  
  return Kg.reshape((ndf2*nme))  
  
def ElemStiffElasMatBb3DP1Vec(nme,q,me,volumes,L,M):
  r"""  Compute all the element elastic stiffness matrices, :math:`\mathbb{K}^e(T_k)` for :math:`k\in\{0,\hdots,\nme-1\}` 
  in local *block* basis.
  
  :param nme: number of mesh elements,
  :type nme: int
  :param q: mesh vertices,
  :type q: ``(3,nq)`` *numpy* array of floats
  :param me: mesh connectivity,
  :type me: ``(4,nme)`` *numpy* array of integers
  :param volumes: volumes of all the mesh elements.
  :type volumes: ``(nme,)`` *numpy* array of floats
  :param la: the  :math:`\\lambda` Lame parameter,
  :type la: float
  :param mu: the  :math:`\\mu` Lame parameter.
  :type mu: float
  :returns: a ``(144*nme,)`` *numpy* array of floats.
  """   
  if q.shape[0]!=3:
    q=q.T
  if me.shape[0]!=4:
    me=me.T
  
  ndf2=144;
  G=ComputeGradientVecTr(q,me)
  coef=6*np.sqrt(volumes)
  for il in range(0,4):
    for i in range(0,3):
      G[il,i]=G[il,i]/coef
  #Kg=zeros((ndf2,nme))
  Kg=np.ndarray(shape=(ndf2,nme))
  T1=G[0,0]**2;T2=G[0,1]**2;T3=G[0,2]**2
  C=M*(T1+ T2 + T3)
  Kg[0]  =(L + M)*T1 + C
  Kg[52] =(L + M)*T2 + C
  Kg[104]=(L + M)*T3 + C
  
  T1=G[1,0]**2;T2=G[1,1]**2;T3=G[1,2]**2
  C=M*(T1+ T2 + T3)
  Kg[13] =(L + M)*T1 + C
  Kg[65] =(L + M)*T2 + C
  Kg[117]=(L + M)*T3 + C
  
  T1=G[2,0]**2;T2=G[2,1]**2;T3=G[2,2]**2
  C=M*(T1+ T2 + T3)
  Kg[26] =(L + M)*T1 + C
  Kg[78] =(L + M)*T2 + C
  Kg[130]=(L + M)*T3 + C
  
  T1=G[3,0]**2;T2=G[3,1]**2;T3=G[3,2]**2
  C=M*(T1+ T2 + T3)
  Kg[39] =(L + M)*T1 + C
  Kg[91] =(L + M)*T2 + C
  Kg[143]=(L + M)*T3 + C
  
  T1=G[0,0]*G[1,0];T2=G[0,1]*G[1,1];T3=G[0,2]*G[1,2]
  C=M*(T1+ T2 + T3)
  Kg[1]  =Kg[12]=(L + M)*T1 + C
  Kg[53] =Kg[64]=(L + M)*T2 + C
  Kg[105]=Kg[116]=(L + M)*T3 + C
  
  T1=G[0,0]*G[2,0];T2=G[0,1]*G[2,1];T3=G[0,2]*G[2,2];
  C=M*(T1+ T2 + T3)
  Kg[2]  =Kg[24]=(L + M)*T1 + C
  Kg[54] =Kg[76]=(L + M)*T2 + C
  Kg[106]=Kg[128]=(L + M)*T3 + C
  
  T1=G[0,0]*G[3,0];T2=G[0,1]*G[3,1];T3=G[0,2]*G[3,2]
  C=M*(T1+ T2 + T3)
  Kg[3]  =Kg[36]=(L + M)*T1 + C
  Kg[55] =Kg[88]=(L + M)*T2 + C
  Kg[107]=Kg[140]=(L + M)*T3 + C
  
  T1=G[1,0]*G[2,0];T2=G[1,1]*G[2,1];T3=G[1,2]*G[2,2]
  C=M*(T1+ T2 + T3)
  Kg[14] =Kg[25]=(L + M)*T1 + C
  Kg[66] =Kg[77]=(L + M)*T2 + C
  Kg[118]=Kg[129]=(L + M)*T3 + C 
  
  T1=G[1,0]*G[3,0];T2=G[1,1]*G[3,1];T3=G[1,2]*G[3,2]
  C=M*(T1+ T2 + T3)
  Kg[15] =Kg[37]=(L + M)*T1 + C 
  Kg[67] =Kg[89]=(L + M)*T2 + C 
  Kg[119]=Kg[141]=(L + M)*T3 + C 
  
  T1=G[2,0]*G[3,0];T2=G[2,1]*G[3,1];T3=G[2,2]*G[3,2]
  C=M*(T1+ T2 + T3)
  Kg[27] =Kg[38]=(L + M)*T1 + C 
  Kg[79] =Kg[90]=(L + M)*T2 + C 
  Kg[131]=Kg[142]=(L + M)*T3 + C 
  
  Kg[4]=Kg[48]=(L+M)*G[0,0]*G[0,1]
  Kg[8]=Kg[96]=(L+M)*G[0,0]*G[0,2]
  Kg[17]=Kg[61]=(L+M)*G[1,0]*G[1,1]
  Kg[21]=Kg[109]=(L+M)*G[1,0]*G[1,2]
  Kg[30]=Kg[74]=(L+M)*G[2,0]*G[2,1]
  Kg[34]=Kg[122]=(L+M)*G[2,0]*G[2,2]
  Kg[43]=Kg[87]=(L+M)*G[3,0]*G[3,1]
  Kg[47]=Kg[135]=(L+M)*G[3,0]*G[3,2]
  Kg[56]=Kg[100]=(L+M)*G[0,1]*G[0,2]
  Kg[69]=Kg[113]=(L+M)*G[1,1]*G[1,2]
  Kg[82]=Kg[126]=(L+M)*G[2,1]*G[2,2]
  Kg[95]=Kg[139]=(L+M)*G[3,1]*G[3,2]
  
  T1=G[0,0]*G[1,1];T2=G[0,1]*G[1,0]
  Kg[5] =Kg[60]=L*T1+M*T2
  Kg[16]=Kg[49]=M*T1+L*T2
  
  T1=G[0,0]*G[2,1];T2=G[0,1]*G[2,0]
  Kg[6] =Kg[72]=L*T1+M*T2
  Kg[28]=Kg[50]=M*T1+L*T2
  
  T1=G[0,0]*G[3,1];T2=G[0,1]*G[3,0]
  Kg[7] =Kg[84]=L*T1+M*T2
  Kg[40]=Kg[51]=M*T1+L*T2
  
  T1=G[0,0]*G[1,2];T2=G[0,2]*G[1,0]
  Kg[9] =Kg[108]=L*T1+M*T2
  Kg[20]=Kg[97]=M*T1+L*T2
  
  T1=G[0,0]*G[2,2];T2=G[0,2]*G[2,0]
  Kg[10]=Kg[120]=L*T1+M*T2
  Kg[32]=Kg[98]=M*T1+L*T2
  
  T1=G[1,0]*G[2,1];T2=G[1,1]*G[2,0]
  Kg[18]=Kg[73]=L*T1+M*T2
  Kg[29]=Kg[62]=M*T1+L*T2
  
  T1=G[1,0]*G[3,1];T2=G[1,1]*G[3,0]
  Kg[19]=Kg[85]=L*T1+M*T2
  Kg[41]=Kg[63]=M*T1+L*T2
  
  T1=G[1,0]*G[2,2];T2=G[1,2]*G[2,0]
  Kg[22]=Kg[121]=L*T1+M*T2
  Kg[33]=Kg[110]=M*T1+L*T2
  
  T1=G[1,0]*G[3,2];T2=G[1,2]*G[3,0]
  Kg[23]=Kg[133]=L*T1+M*T2
  Kg[45]=Kg[111]=M*T1+L*T2
  
  T1=G[2,0]*G[3,1];T2=G[2,1]*G[3,0]
  Kg[31]=Kg[86]=L*T1+M*T2
  Kg[42]=Kg[75]=M*T1+L*T2
  
  T1=G[2,0]*G[3,2];T2=G[2,2]*G[3,0]
  Kg[35]=Kg[134]=L*T1+M*T2
  Kg[46]=Kg[123]=M*T1+L*T2
  
  T1=G[0,0]*G[3,2];T2=G[0,2]*G[3,0]
  Kg[11]=Kg[132]=L*T1+M*T2
  Kg[44]=Kg[99]=M*T1+L*T2
  
  T1=G[0,1]*G[1,2];T2=G[0,2]*G[1,1]
  Kg[57]=Kg[112]=L*T1+M*T2
  Kg[68]=Kg[101]=M*T1+L*T2
  
  T1=G[0,1]*G[2,2];T2=G[0,2]*G[2,1]
  Kg[58]=Kg[124]=L*T1+M*T2
  Kg[80]=Kg[102]=M*T1+L*T2
  
  T1=G[0,1]*G[3,2];T2=G[0,2]*G[3,1]
  Kg[59]=Kg[136]=L*T1+M*T2
  Kg[92]=Kg[103]=M*T1+L*T2
  
  T1=G[1,1]*G[2,2];T2=G[1,2]*G[2,1]
  Kg[70]=Kg[125]=L*T1+M*T2
  Kg[81]=Kg[114]=M*T1+L*T2
  
  T1=G[1,1]*G[3,2];T2=G[1,2]*G[3,1]
  Kg[71]=Kg[137]=L*T1+M*T2
  Kg[93]=Kg[115]=M*T1+L*T2  

  T1=G[2,1]*G[3,2];T2=G[2,2]*G[3,1]
  Kg[83]=Kg[138]=L*T1+M*T2  
  Kg[94]=Kg[127]=M*T1+L*T2  
   
  return Kg.reshape((ndf2*nme))
