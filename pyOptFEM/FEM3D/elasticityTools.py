import numpy as np


def Hooke(la,mu):
  H=np.zeros((6,6));
  H[0:3,0:3]=la*np.ones((3,3))+2*mu*np.eye(3)
  H[3,3]=H[4,4]=H[5,5]=mu
  return H
  
def BuildIkFunc(Num,nq):
  options = {0 : BuildIkFunc0,
             1 : BuildIkFunc1,
             2 : BuildIkFunc2,
             3 : BuildIkFunc3}
  return options[Num](nq)

def BuildIkFunc0(nq):
  return lambda me,k: np.array([3*me[k,0],3*me[k,0]+1,3*me[k,0]+2,
                                3*me[k,1],3*me[k,1]+1,3*me[k,1]+2,
                                3*me[k,2],3*me[k,2]+1,3*me[k,2]+2,
                                3*me[k,3],3*me[k,3]+1,3*me[k,3]+2,])

def BuildIkFunc1(nq):
  return lambda me,k: np.array([me[k,0],me[k,0]+nq,me[k,0]+2*nq,
                                me[k,1],me[k,1]+nq,me[k,1]+2*nq,
                                me[k,2],me[k,2]+nq,me[k,2]+2*nq,
                                me[k,3],me[k,3]+nq,me[k,3]+2*nq])

def BuildIkFunc2(nq):
  return lambda me,k: np.concatenate((3*me[k],3*me[k]+1,3*me[k]+2))
  
def BuildIkFunc3(nq):
  return lambda me,k: np.concatenate((me[k],me[k]+nq,me[k]+2*nq))

  
def BuildElemStiffElasMatFunc(Num):
  options = {0 : BuildElemStiffElasMatFunc0,
             1 : BuildElemStiffElasMatFunc0,
             2 : BuildElemStiffElasMatFunc1,
             3 : BuildElemStiffElasMatFunc1}
  return options[Num](ql,area,C)
  

def BuildIkVecFunc(Num,nq):
  options = {0 : BuildIkVecFunc0,
             1 : BuildIkVecFunc1,
             2 : BuildIkVecFunc2,
             3 : BuildIkVecFunc3}
  return options[Num](nq)

def BuildIkVecFunc0(nq):
  return lambda meT: np.array([3*meT[0],3*meT[0]+1,3*meT[0]+2,
                               3*meT[1],3*meT[1]+1,3*meT[1]+2,
                               3*meT[2],3*meT[2]+1,3*meT[2]+2,
                               3*meT[3],3*meT[3]+1,3*meT[3]+2])
def BuildIkVecFunc0a(me,nq):
  nme=me.shape[1]
  I=np.ndarray(shape=(12,nme),dtype=np.int32)
  for i in range(0,4):
    I[3*i]=3*me[i]
    I[3*i+1]=3*me[i]+1
    I[3*i+2]=3*me[i]+2
  return I
                            
def BuildIkVecFunc1(nq):
  return lambda meT: np.array([meT[0],meT[0]+nq,meT[0]+2*nq,
                               meT[1],meT[1]+nq,meT[1]+2*nq,
                               meT[2],meT[2]+nq,meT[2]+2*nq,
                               meT[3],meT[3]+nq,meT[3]+2*nq])
                            
def BuildIkVecFunc1a(me,nq):
  nme=me.shape[1]
  I=np.ndarray(shape=(12,nme),dtype=np.int32)
  for i in range(0,4):
    I[3*i]=me[i]
    I[3*i+1]=me[i]+nq
    I[3*i+2]=me[i]+2*nq
  return I

def BuildIkVecFunc2(nq):
  return lambda meT: np.concatenate((3*meT,3*meT+1,3*meT+2))

def BuildIkVecFunc2a(me,nq):
  nme=me.shape[1]
  I=np.ndarray(shape=(12,nme),dtype=np.int32)
  for i in np.arange(0,3):
    for j in np.arange(0,4):
      I[4*i+j]=3*me[j]+i
  return I
  
def BuildIkVecFunc3(nq):
  return lambda meT: np.concatenate((meT,meT+nq,meT+2*nq))
  
def BuildIkVecFunc3a(me,nq):
  nme=me.shape[1]
  I=np.ndarray(shape=(12,nme),dtype=np.int32)
  for i in np.arange(0,3):
    for j in np.arange(0,4):
      I[4*i+j]=me[j]+i*nq
  return I
  
  
def BuildIkVec(Num,me,nq):
  if Num==0:
    return BuildIkVecFunc0a(me,nq)
  if Num==1:
    return BuildIkVecFunc1a(me,nq)
  if Num==2:
    return BuildIkVecFunc2a(me,nq)
  if Num==3:
    return BuildIkVecFunc3a(me,nq)
  print('Error in BuildIkVec')
  
def BuildIgJgP1VF(Num,meT,nq):
  GetI=BuildIkVecFunc(Num,nq)
  nme=meT.shape[1]
  ii=(np.array(range(0,12),dtype=np.int32,ndmin=2).T)*np.ones((1,12),dtype=np.int32)
  jj=ii.T.reshape(144)
  ii.shape = (144)
  I=GetI(meT)
  Ig=I[ii].reshape((144*nme))
  Jg=I[jj].reshape((144*nme))
  return Ig,Jg

# Version + rapide que BuildIgJgP1VF (5 a 10 fois)
def BuildIgJg3DP1VF(Num,me,nq):
  nme=me.shape[1]
  I=BuildIkVec(Num,me,nq)
  ii=(np.array(np.arange(0,12),dtype=np.int32,ndmin=2).T)*np.ones((1,12),dtype=np.int32)
  jj=ii.T.reshape(144)
  ii.shape = (144)
  
  Ig=np.ndarray(shape=(144,nme),dtype=np.int32)
  Jg=np.ndarray(shape=(144,nme),dtype=np.int32)
  for i in np.arange(0,12):
    for j in np.arange(0,12):
      Ig[12*i+j]=I[i]
      Jg[12*i+j]=I[j]
  return Ig.reshape((144*nme)),Jg.reshape((144*nme))  
