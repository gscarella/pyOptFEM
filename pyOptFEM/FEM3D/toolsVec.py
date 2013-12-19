import numpy as np

def ComputeVolumesOpt(q,me):
  D21=q[me[:,1]]- q[me[:,0]]
  D31=q[me[:,2]]- q[me[:,0]]
  D41=q[me[:,3]]- q[me[:,0]]
  return np.abs((D21[:,0]*(D31[:,1]*D41[:,2]-D31[:,2]*D41[:,1])+D31[:,0]*(D41[:,1]*D21[:,2]-D41[:,2]*D21[:,1])+D41[:,0]*(D21[:,1]*D31[:,2]-D21[:,2]*D31[:,1]))/6.)
  
def ComputeVolumesOptT(q,me):
  D21=q[:,me[1]]- q[:,me[0]]
  D31=q[:,me[2]]- q[:,me[0]]
  D41=q[:,me[3]]- q[:,me[0]]
  return np.abs((D21[0]*(D31[1]*D41[2]-D31[2]*D41[1])+D31[0]*(D41[1]*D21[2]-D41[2]*D21[1])+D41[0]*(D21[1]*D31[2]-D21[2]*D31[1]))/6.)
  
def GetMaxLengthEdges(q,me):
  LE=[q[me[:,1]]- q[me[:,0]],q[me[:,2]]- q[me[:,0]],q[me[:,3]]- q[me[:,0]],
      q[me[:,2]]- q[me[:,1]],q[me[:,3]]- q[me[:,1]],q[me[:,3]]- q[me[:,2]]]
  h=max(np.sum(LE[0]**2,axis=1))
  for i in range(1,6):
    h=max(h,max(np.sum(LE[i]**2,axis=1)))
  return np.sqrt(h)
