import numpy as np
from scipy import sparse
from scipy.spatial import Delaunay
from .toolsVec import ComputeAreaOpt
from .elemMatrix import *
from .elemMatrixVec import *
from .elasticityTools import *

#
# 1) Mass Matrix Assembly
#    --------------------

# q (nq,2), me(nme,3),
def MassAssembling2DP1base(nq,nme,me,areas):
  """  Assembly of the Mass Matrix by :math:`P_1`-Lagrange finite elements using ``base`` version (see report).
  """
  M=sparse.lil_matrix((nq,nq))
  for k in range(0,nme):
    E=ElemMassMat2DP1(areas[k])
    for il in range(0,3):
      i=me[k,il]
      for jl in range(0,3):
        j=me[k,jl]
        M[i,j]=M[i,j]+E[il,jl]
  return M.tocsc()

# q (nq,2), me(nme,3),  
def MassAssembling2DP1OptV1(nq,nme,me,areas):
  """  Assembly of the Mass Matrix by :math:`P_1`-Lagrange finite elements using ``OptV1`` version (see report).
  """
  Kg=np.zeros((9*nme))
  Ig=np.zeros((9*nme),dtype=np.int32)
  Jg=np.zeros((9*nme),dtype=np.int32)
  ii=np.array([0,1,2,0,1,2,0,1,2]);jj=np.array([0,0,0,1,1,1,2,2,2]);
  kk=np.array(range(0,9));
  for k in range(0,nme):
    E=ElemMassMat2DP1(areas[k])
    Ig[kk]=me[k][ii]; Jg[kk]=me[k][jj]
    Kg[kk]=E.reshape((9))
    kk+=9
  return sparse.csc_matrix((Kg,(Ig,Jg)),shape=(nq,nq))

# q (nq,2), me(nme,3),
def MassAssembling2DP1OptV2(nq,nme,me,areas):
  """  Assembly of the Mass Matrix by :math:`P_1`-Lagrange finite elements using OptV2 version (see report).
  """
  Kg=ElemMassMat2DP1Vec(areas)
  Ig=me[:,[0,0,0,1,1,1,2,2,2]]
  Jg=me[:,[0,1,2,0,1,2,0,1,2]]
  return sparse.csc_matrix((Kg,(np.reshape(Ig,9*nme),np.reshape(Jg,9*nme))),shape=(nq,nq))

#
# 2) Stiff Matrix Assembly
#    ---------------------

# q (nq,2), me(nme,3),
def StiffAssembling2DP1base(nq,nme,q,me,areas):
  """  Assembly of the Stiffness Matrix by :math:`P_1`-Lagrange finite elements using ``base`` version (see report).
  """
  S=sparse.lil_matrix((nq,nq))
  for k in range(0,nme):
    E=ElemStiffMat2DP1(q[me[k,0]],q[me[k,1]],q[me[k,2]],areas[k])
    for il in range(0,3):
      i=me[k,il]
      for jl in range(0,3):
        j=me[k,jl]
        S[i,j]=S[i,j]+E[il,jl]
  return S.tocsc()

# q (nq,2), me(nme,3),
def StiffAssembling2DP1OptV1(nq,nme,q,me,areas):
  """  Assembly of the Stiffness Matrix by :math:`P_1`-Lagrange finite elements using ``OptV1`` version (see report).
  """
  Kg=np.zeros((9*nme))
  Ig=np.zeros((9*nme),dtype=np.int32)
  Jg=np.zeros((9*nme),dtype=np.int32)
  ii=np.array([0,1,2,0,1,2,0,1,2]);jj=np.array([0,0,0,1,1,1,2,2,2]);
  kk=np.array(range(0,9));
  for k in range(0,nme):
    E=ElemStiffMat2DP1(q[me[k,0]],q[me[k,1]],q[me[k,2]],areas[k])
    Ig[kk]=me[k][ii]; Jg[kk]=me[k][jj]
    Kg[kk]=E.reshape((9))
    kk+=9
  return sparse.csc_matrix((Kg,(Ig,Jg)),shape=(nq,nq))

# q (2,nq), me(3,nme),
def StiffAssembling2DP1OptV2(nq,nme,q,me,areas):
  """  Assembly of the Stiffness Matrix by :math:`P_1`-Lagrange finite elements using ``OptV2`` version (see report).
  """
  Kg=ElemStiffMat2DP1Vec(nme,q,me,areas)
  Ig=me[:,[0,0,0,1,1,1,2,2,2]]
  Jg=me[:,[0,1,2,0,1,2,0,1,2]]
  return sparse.csc_matrix((Kg,(np.reshape(Ig,9*nme),np.reshape(Jg,9*nme))),shape=(nq,nq))

#
# 3) Stiff Elas Matrix Assembly
#    --------------------------

# q (nq,2), me(nme,3),
def StiffElasAssembling2DP1base(nq,nme,q,me,areas,la,mu,Num):
  """  Assembly of the Elasticity Stiffness Matrix by :math:`P_1`-Lagrange finite elements using OptV2 version (see report).
  """
  R=sparse.lil_matrix((2*nq,2*nq))
  GetI=BuildIkFunc(Num,nq)
  if Num<=1:
    ElemStiffElasMat=lambda ql,area,C: ElemStiffElasMat2DP1Ba(ql,area,C)
  else:
    ElemStiffElasMat=lambda ql,area,C: ElemStiffElasMat2DP1Bb(ql,area,C)
  C=Hooke(la,mu)
  for k in range(0,nme):
    E=ElemStiffElasMat(q[me[k]],areas[k],C)
    I=GetI(me,k)
    for il in range(0,6):
      i=I[il]
      for jl in range(0,6):
        j=I[jl]
        R[i,j]=R[i,j]+E[il,jl]
  return R.tocsr()

# q (nq,2), me(nme,3),  
def StiffElasAssembling2DP1OptV1(nq,nme,q,me,areas,la,mu,Num):
  """  Assembly of the Elasticity Stiffness Matrix by :math:`P_1`-Lagrange finite elements using OptV1 version (see report).
  """
  GetI=BuildIkFunc(Num,nq)
  if Num<=1:
    ElemStiffElasMat=lambda ql,area,C: ElemStiffElasMat2DP1Ba(ql,area,C)
  else:
    ElemStiffElasMat=lambda ql,area,C: ElemStiffElasMat2DP1Bb(ql,area,C)
  C=Hooke(la,mu)
  Kg=np.zeros((36*nme))
  Ig=np.zeros((36*nme),dtype=np.int32)
  Jg=np.zeros((36*nme),dtype=np.int32)
  kk=np.array(range(0,36));
  for k in range(0,nme):
    E=ElemStiffElasMat(q[me[k]],areas[k],C)
    I=GetI(me,k)
    jA=np.array([I,]*6,dtype=np.int32)
    Ig[kk]=jA.T.reshape((36)); Jg[kk]=jA.reshape((36))
    Kg[kk]=E.reshape((36))
    kk+=36
  return sparse.csc_matrix((Kg,(Ig,Jg)),shape=(2*nq,2*nq))

# q (2,nq), me(3,nme),
def StiffElasAssembling2DP1OptV2(nq,nme,q,me,areas,la,mu,Num):
  """  Assembly of the Elasticity Stiffness Matrix by :math:`P_1`-Lagrange finite elements using OptV2 version (see report).
  """
  if q.shape[0]!=2:
    q=q.T
  if me.shape[0]!=3:
    me=me.T
  Ig,Jg=BuildIgJg2DP1VF(Num,me,nq)
  if Num<=1:
    Kg=ElemStiffElasMatBaVec2DP1(nme,q,me,areas,la,mu)
  else:
    Kg=ElemStiffElasMatBbVec2DP1(nme,q,me,areas,la,mu)
  return sparse.csc_matrix((Kg,(Ig,Jg)),shape=(2*nq,2*nq))
