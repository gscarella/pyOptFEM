import numpy as np
from scipy import sparse
from scipy.spatial import Delaunay
from .elemMatrix import *
from .elemMatrixVec import *
from .elasticityTools import *

        
def MassAssembling3DP1base(nq,nme,me,volumes):
  """  Assembly of the Mass Matrix by :math:`P_1`-Lagrange finite elements using ``base`` version (see report).
  """
  M=sparse.lil_matrix((nq,nq))
  for k in range(0,nme):
    E=ElemMassMat3DP1(volumes[k])
    for il in range(0,4):
      i=me[k,il]
      for jl in range(0,4):
        j=me[k,jl]
        M[i,j]=M[i,j]+E[il,jl]
  return M.tocsc()
  
def MassAssembling3DP1OptV1(nq,nme,me,volumes):
  """  Assembly of the Mass Matrix by :math:`P_1`-Lagrange finite elements using ``OptV1`` version (see report).
  """
  Kg=np.zeros((16*nme))
  Ig=np.zeros((16*nme),dtype=np.int32)
  Jg=np.zeros((16*nme),dtype=np.int32)
  ii=np.array([0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]);jj=np.array([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3]);
  kk=np.array(range(0,16));
  for k in range(0,nme):
    E=ElemMassMat3DP1(volumes[k])
    Ig[kk]=me[k][ii]; Jg[kk]=me[k][jj]
    Kg[kk]=E.reshape((16))
    kk+=16
  return sparse.csc_matrix((Kg,(Ig,Jg)),shape=(nq,nq))
  
def MassAssembling3DP1OptV2(nq,nme,me,volumes):
  """  Assembly of the Mass Matrix by :math:`P_1`-Lagrange finite elements using ``OptV2`` version (see report).
  """
  Jg = me[:,[0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3]]
  Ig = me[:,[0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]]
  Kg = ElemMassMat3DP1Vec(nme,volumes)
  return sparse.csc_matrix((Kg,(np.reshape(Ig,16*nme),np.reshape(Jg,16*nme))),shape=(nq,nq))

# 2) Stiff Matrix Assembly       
def StiffAssembling3DP1base(nq,nme,q,me,volumes):
  """  Assembly of the Stiffness Matrix by :math:`P_1`-Lagrange finite elements using ``base`` version (see report).
  """
  S=sparse.lil_matrix((nq,nq))
  for k in range(0,nme):
    E=ElemStiffMat3DP1(q[me[k]],volumes[k])
    for il in range(0,4):
      i=me[k,il]
      for jl in range(0,4):
        j=me[k,jl]
        S[i,j]=S[i,j]+E[il,jl]
  return S.tocsc()
  
def StiffAssembling3DP1OptV1(nq,nme,q,me,volumes):
  """  Assembly of the Stiffness Matrix by :math:`P_1`-Lagrange finite elements using ``OptV1`` version (see report).
  """
  Kg=np.zeros((16*nme))
  Ig=np.zeros((16*nme),dtype=np.int32)
  Jg=np.zeros((16*nme),dtype=np.int32)
  ii=np.array([0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]);jj=np.array([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3]);
  kk=np.array(range(0,16));
  for k in range(0,nme):
    E=ElemStiffMat3DP1(q[me[k]],volumes[k])
    Ig[kk]=me[k][ii]; Jg[kk]=me[k][jj]
    Kg[kk]=E.reshape((16))
    kk+=16
  return sparse.csc_matrix((Kg,(Ig,Jg)),shape=(nq,nq))

#   me (4,nme) array,  q (3,nq) array
def StiffAssembling3DP1OptV2(nq,nme,q,me,volumes):
  """  Assembly of the Stiffness Matrix by :math:`P_1`-Lagrange finite elements using ``OptV2`` version (see report).
  """
  if q.shape[0]!=3:
    q=q.T
  if me.shape[0]!=4:
    me=me.T
  Ig=np.ndarray(shape=(16,nme))
  Jg=np.ndarray(shape=(16,nme))
  Ig[0]=Ig[4] =Ig[8]=Ig[12]= Jg[0]= Jg[1]= Jg[2]= Jg[3]=me[0]
  Ig[1]=Ig[5] =Ig[9]=Ig[13]= Jg[4]= Jg[5]= Jg[6]= Jg[7]=me[1]
  Ig[2]=Ig[6]=Ig[10]=Ig[14]= Jg[8]= Jg[9]=Jg[10]=Jg[11]=me[2]
  Ig[3]=Ig[7]=Ig[11]=Ig[15]=Jg[12]=Jg[13]=Jg[14]=Jg[15]=me[3]
  
  Kg = ElemStiffMat3DP1Vec(nme,q,me,volumes)
  return sparse.csc_matrix((Kg,(np.reshape(Ig,16*nme),np.reshape(Jg,16*nme))),shape=(nq,nq))

# 3) Stiff Elas Matrix Assembly

def StiffElasAssembling3DP1base(nq,nme,q,me,volumes,la,mu,Num):
  """  Assembly of the Elasticity Stiffness Matrix by :math:`P_1`-Lagrange finite elements using ``base`` version (see report).
  """
  R=sparse.lil_matrix((3*nq,3*nq))
  GetI=BuildIkFunc(Num,nq)
  if Num<=1:
    ElemStiffElasMat=lambda ql,volume,C: ElemStiffElasMatBa3DP1(ql,volume,C)
  else:
    ElemStiffElasMat=lambda ql,volume,C: ElemStiffElasMatBb3DP1(ql,volume,C)
  C=Hooke(la,mu)
  for k in range(0,nme):
    E=ElemStiffElasMat(q[me[k]],volumes[k],C)
    I=GetI(me,k)
    for il in range(0,12):
      i=I[il]
      for jl in range(0,12):
        j=I[jl]
        R[i,j]=R[i,j]+E[il,jl]
  return R.tocsc()

def StiffElasAssembling3DP1OptV1(nq,nme,q,me,volumes,la,mu,Num):
  """  Assembly of the Elasticity Stiffness Matrix by :math:`P_1`-Lagrange finite elements using ``OptV1`` version (see report).
  """
  GetI=BuildIkFunc(Num,nq)
  if Num<=1:
    ElemStiffElasMat=lambda ql,volume,C: ElemStiffElasMatBa3DP1(ql,volume,C)
  else:
    ElemStiffElasMat=lambda ql,volume,C: ElemStiffElasMatBb3DP1(ql,volume,C)
  C=Hooke(la,mu)
  Kg=np.zeros((144*nme))
  Ig=np.zeros((144*nme),dtype=np.int32)
  Jg=np.zeros((144*nme),dtype=np.int32)
  kk=np.array(range(0,144));
  for k in range(0,nme):
    E=ElemStiffElasMat(q[me[k]],volumes[k],C)
    I=GetI(me,k)
    jA=np.array([I,]*12,dtype=np.int32)
    Ig[kk]=jA.T.reshape((144)); Jg[kk]=jA.reshape((144))
    Kg[kk]=E.reshape((144))
    kk+=144
  return sparse.csc_matrix((Kg,(Ig,Jg)),shape=(3*nq,3*nq))
  

def StiffElasAssembling3DP1OptV2(nq,nme,q,me,volumes,la,mu,Num):
  """  Assembly of the Elasticity Stiffness Matrix by :math:`P_1`-Lagrange finite elements using ``OptV2`` version (see report).
  """
  if q.shape[0]!=3:
    q=q.T
  if me.shape[0]!=4:
    me=me.T
  Ig,Jg=BuildIgJg3DP1VF(Num,me,nq)
  if (Num>1):
    Kg=ElemStiffElasMatBb3DP1Vec(nme,q,me,volumes,la,mu)
  else:
    Kg=ElemStiffElasMatBa3DP1Vec(nme,q,me,volumes,la,mu)
  return sparse.csc_matrix((Kg,(Ig,Jg)),shape=(3*nq,3*nq))
