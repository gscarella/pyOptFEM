import numpy as np

def ComputeAreaOpt(q,me):
  c1=q[me[:,0]]
  d21=q[me[:,1]]-c1
  d31=q[me[:,2]]-c1
  return 0.5*np.abs(d21[:,0]*d31[:,1]-d21[:,1]*d31[:,0])

def ComputeAreaOpt2(q,me):
  c1=q[:,me[0,:]]
  d21=q[:,me[1,:]]-c1
  d31=q[:,me[2,:]]-c1
  return 0.5*np.abs(d21[0,:]*d31[1,:]-d21[1,:]*d31[0,:])


def GetMaxLengthEdges(q,me):
  U=q[me[:,0]]-q[me[:,1]]
  V=q[me[:,1]]-q[me[:,2]]
  W=q[me[:,2]]-q[me[:,0]]
  return np.sqrt(max(max(np.sum(U**2,1)),max(np.sum(V**2,1)),max(np.sum(W**2,1))))