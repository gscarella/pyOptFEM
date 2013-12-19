import numpy as np

def Hooke(la,mu):
  return np.array([[la+2*mu,la,0],[la,la+2*mu,0],[0,0,mu]]);
  
def BuildIkFunc(Num,nq):
  options = {0 : BuildIkFunc0,
             1 : BuildIkFunc1,
             2 : BuildIkFunc2,
             3 : BuildIkFunc3}
  return options[Num](nq)

def BuildIkFunc0(nq):
  return lambda me,k: np.array([2*me[k,0],2*me[k,0]+1,2*me[k,1],2*me[k,1]+1,2*me[k,2],2*me[k,2]+1])

def BuildIkFunc1(nq):
  return lambda me,k: np.array([me[k,0],me[k,0]+nq,me[k,1],me[k,1]+nq,me[k,2],me[k,2]+nq])

def BuildIkFunc2(nq):
  return lambda me,k: np.concatenate([2*me[k],2*me[k]+1])
  
def BuildIkFunc3(nq):
  return lambda me,k: np.concatenate([me[k],me[k]+nq])

  
def BuildIkVecFunc(Num,nq):
  options = {0 : BuildIkVecFunc0,
             1 : BuildIkVecFunc1,
             2 : BuildIkVecFunc2,
             3 : BuildIkVecFunc3}
  return options[Num](nq)
  
def BuildIkVecFunc0(nq):
  return lambda meT: np.array([2*meT[0],2*meT[0]+1,2*meT[1],2*meT[1]+1,2*meT[2],2*meT[2]+1])
  
def BuildIkVecFunc0a(me,nq):
  nme=me.shape[1]
  I=np.ndarray(shape=(6,nme),dtype=np.int32)
  for i in range(0,3):
    I[2*i]=2*me[i]
    I[2*i+1]=2*me[i]+1
  return I

def BuildIkVecFunc1(nq):
  return lambda meT: np.array([meT[0],meT[0]+nq,meT[1],meT[1]+nq,meT[2],meT[2]+nq])
  
def BuildIkVecFunc1a(me,nq):
  nme=me.shape[1]
  I=np.ndarray(shape=(6,nme),dtype=np.int32)
  for i in range(0,3):
    I[2*i]=me[i]
    I[2*i+1]=me[i]+nq
  return I
  
  
def BuildIkVecFunc2(nq):
  return lambda meT: np.concatenate([2*meT,2*meT+1])
  
def BuildIkVecFunc2a(me,nq):
  nme=me.shape[1]
  I=np.ndarray(shape=(6,nme),dtype=np.int32)
  for i in np.arange(0,2):
    for j in np.arange(0,3):
      I[3*i+j]=2*me[j]+i
  return I
  
def BuildIkVecFunc3(nq):
  return lambda meT: np.concatenate([meT,meT+nq])
  
  
def BuildIkVecFunc3a(me,nq):
  nme=me.shape[1]
  I=np.ndarray(shape=(6,nme),dtype=np.int32)
  for i in np.arange(0,2):
    for j in np.arange(0,3):
      I[3*i+j]=me[j]+i*nq
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
  ii=(np.array(range(0,6),dtype=np.int32,ndmin=2).T)*np.ones((1,6),dtype=np.int32)
  jj=ii.T.reshape(36)
  ii.shape = (36)
  I=GetI(meT)
  Ig=I[ii].reshape((36*nme))
  Jg=I[jj].reshape((36*nme))
  return Ig,Jg
  
def BuildIgJg2DP1VF(Num,me,nq):
  nme=me.shape[1]
  I=BuildIkVec(Num,me,nq)
  ii=(np.array(np.arange(0,6),dtype=np.int32,ndmin=2).T)*np.ones((1,6),dtype=np.int32)
  jj=ii.T.reshape(36)
  ii.shape = (36)
  
  Ig=np.ndarray(shape=(36,nme),dtype=np.int32)
  Jg=np.ndarray(shape=(36,nme),dtype=np.int32)
  for i in np.arange(0,6):
    for j in np.arange(0,6):
      Ig[6*i+j]=I[i]
      Jg[6*i+j]=I[j]
  return Ig.reshape((36*nme)),Jg.reshape((36*nme))  
  
def BuildIgJg2DP1VFsym(Num,me,nq):
  nme=me.shape[1]
  I=BuildIkVec(Num,me,nq)
  ni=(6*6-6)/2+6
  ii=np.empty(ni,dtype=np.int32);
  jj=np.empty(ni,dtype=np.int32);
  kd=0;ld=6
  for i in range(0,6):
    ii[kd:kd+ld]=i
    jj[kd:kd+ld]=np.arange(i,6,dtype=np.int32)
    kd+=ld
    ld+=-1
  Ig=I[ii]#.reshape((ni*nme))
  Jg=I[jj]#.reshape((ni*nme))
  return Ig,Jg  
  