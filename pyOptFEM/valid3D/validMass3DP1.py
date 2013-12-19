import numpy,sympy,time,sys
from ..FEM3D import *
from .common import *


class TestMass3D:
  def __init__(self, cu,cv):
    x, y, z = sympy.symbols('x y z')
    self.cu=cu
    self.cv=cv
    self.u=sympy.Lambda((x,y,z),cu)
    self.v=sympy.Lambda((x,y,z),cv)
    self.fu=eval("lambda x,y,z: "+cu)
    self.fv=eval("lambda x,y,z: "+cv)
    if self.u.is_polynomial(x,y,z):
      D=sympy.polys.Poly(cu,x,y,z).as_dict()
      self.du=max(numpy.sum(list(D.keys()),axis=1))
    else:
      self.du=-1
    if self.v.is_polynomial(x,y,z):
      D=sympy.polys.Poly(cv,x,y,z).as_dict()
      self.dv=max(numpy.sum(list(D.keys()),axis=1))
    else:
      self.dv=-1
    self.I=sympy.integrate(sympy.integrate(sympy.integrate(self.u(x,y,z)*self.v(x,y,z),(x,0,1)),(y,0,1)),(z,0,1))
    
    
def validMass3DP1(**kwargs):
  """ Validation of Mass Matrix for P1-Lagrange finite elements in 3D
  """
  Plot=kwargs.get('plot',True)
  Verbose=kwargs.get('verbose',True)
  print('*********************************************')
  print('*     3D Mass Assembling P1 validations     *')
  print('*********************************************')    

  Th=CubeMesh(5)

  LF=[['x + y + z','x - y - z'],
      ['3*x + 2*y - z - 1','2*x - 2*y + 2*z + 1'],
      ['3*x**2 - x*y + 2*y**2 + y*z - z**2 - 3','2*x**2 + x*y - 3*y**2 - x*z - y']]
  if Verbose:     
    print('-----------------------------------------')
    print('  Test 1: Matrices errors and CPU times  ')
    print('-----------------------------------------')

  T=numpy.zeros(3)
  E=numpy.zeros(2)
  tstart=time.time()
  Mbase=MassAssembling3DP1base(Th.nq,Th.nme,Th.me,Th.volumes)
  T[0]=time.time()-tstart
  if Verbose: print("    Matrix size           : (%d,%d)" % Mbase.shape)
  tstart=time.time()
  MOptV1=MassAssembling3DP1OptV1(Th.nq,Th.nme,Th.me,Th.volumes)
  T[1]=time.time()-tstart
  E[0]=NormInf(Mbase-MOptV1)
  if Verbose: print("    Error P1base vs OptV1 : %e" % E[0])
  tstart=time.time()
  MOptV2=MassAssembling3DP1OptV2(Th.nq,Th.nme,Th.me,Th.volumes)
  T[2]=time.time()-tstart
  E[1]=NormInf(Mbase-MOptV2)
  if Verbose: 
    print("    Error P1base vs OptV2 : %e" % E[1])
    print("    CPU times base (ref)  : %3.4f (s)" % T[0])
    print("    CPU times OptV1       : %3.4f (s) - Speed Up X%3.3f" % (T[1],T[0]/T[1]))
    print("    CPU times OptV2       : %3.4f (s) - Speed Up X%3.3f" % (T[2],T[0]/T[2]))
  if checkTest1(E)==1:
    sys.exit()
  if Verbose:
    print('-----------------------------------------------------')
    print('  Test 2: Validations by integration on [0,1]x[0,1]  ')
    print('-----------------------------------------------------')
  deg=numpy.zeros(len(LF))
  E=numpy.zeros(len(LF))
  for i in range(len(LF)):
    test=TestMass3D(LF[i][0],LF[i][1])
    deg[i]=test.du+test.dv
    U=test.fu(Th.q[:,0],Th.q[:,1],Th.q[:,2])
    V=test.fv(Th.q[:,0],Th.q[:,1],Th.q[:,2])
    Ifem=numpy.dot(Mbase*U,V)
    E[i]=abs(Ifem-test.I)
    if Verbose: print("    function %d :\n      u(x,y,z)=%s," % (i,test.cu) )      
    if Verbose: print("      v(x,y,z)=%s,\n           -> Stiff error=%e" % (test.cv,E[i]) )
  if checkTest2(deg,E)==1:
    sys.exit()
  if Verbose:
    print('--------------------------------')
    print('  Test 3: Validations by order  ')
    print('--------------------------------')
  tstart=time.time()
  #LN=range(5,55,5)
  LN=4*np.arange(1,7)
  n=len(LN)
  h=numpy.zeros(n)
  Error=numpy.zeros(n)
  k=len(LF)-1
  test=TestMass3D(LF[k][0],LF[k][1])
  if Verbose: print("    function %d :\n      u(x,y,z)=%s,\n      v(x,y,z)=%s" %(k,test.cu,test.cv))
  for i in range(n):
    Th=CubeMesh(LN[i])
    tstart=time.time()
    M=MassAssembling3DP1OptV2(Th.nq,Th.nme,Th.me,Th.volumes)
    TT=time.time()-tstart
    if Verbose: 
      print("        Matrix size                       : (%d,%d)" % M.shape)
      print("        MassAssembling3DP1OptV2 CPU times : %3.3f(s)" % TT)
    U=test.fu(Th.q[:,0],Th.q[:,1],Th.q[:,2])
    V=test.fv(Th.q[:,0],Th.q[:,1],Th.q[:,2])
    Ifem=numpy.dot(M*U,V)
    h[i]=GetMaxLengthEdges(Th.q,Th.me)
    Error[i]=abs(Ifem-test.I)
    if Verbose: print("        -> Error                         : %e" % Error[i]);
  if checkTest3(h,Error)==1:
    sys.exit() 
  if Plot:    
    PlotTest3(h,Error,'Test 3 : Mass Matrix (3D/P1)')
