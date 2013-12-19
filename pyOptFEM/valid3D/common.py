import numpy as np
import matplotlib.pyplot as plt
from ..common  import *

def EvalFunc3D(u,X,Y,Z):
  n=len(X)
  val=np.zeros(n)
  for i in range(n):
    val[i]=u(X[i],Y[i],Z[i])
  return val
  
def checkTest1(E):
  print("----------------------------")
  if (max(E)<1e-14):
    print('  Test 1 (results): OK')
    return 0
  else:
    print('  Test 1 (results): FAILED')
    return 1
  
def checkTest2(deg,E):
  cntFalse=0
  for k in range(len(E)):
    if (deg[k]==0 or deg[k]==1) and E[k]>1e-14:
      cntFalse+=1
   
  print("----------------------------")
  if cntFalse==0:
    print('  Test 2 (results): OK')
    return 0
  else:
    print('  Test 2 (results): FAILED')
    return 1

def checkTest3(h,E):
  P=np.polyfit(log(h),log(E),1)
  if abs(P[0]-2)< 0.2:
    print("----------------------------")
    print('  Test 3 (results): OK')
    print('    -> found numerical order %f. Must be 2' % P[0])
    print('----------------------------')
    return 0
  else:
    print('----------------------------')
    print('  Test 3 (results): FAILED')
    print('    -> found numerical order %f. Must be 2' % P[0])
    print('----------------------------')
    return 1
   
def PlotTest3(h,Error,cTitle):
  plt.loglog(h,Error,'r',label='Error');
  plt.loglog(h,h**2,'b',label='$O(h^2)$');
  legend = plt.legend(loc='lower right', shadow=True);
  plt.title(cTitle);
  plt.xlabel('h');
  plt.grid(True);
  plt.show()