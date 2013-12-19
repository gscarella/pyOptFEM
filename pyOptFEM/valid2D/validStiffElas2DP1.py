import numpy,sympy,time,sys
from ..FEM2D import *
from .common import *
  
def Gamma(u):
  x, y = sympy.symbols('x y')
  return sympy.Matrix([sympy.diff(u[0],x),sympy.diff(u[1],y),sympy.diff(u[1],x)+sympy.diff(u[0],y)])
  

def Hooke(L,M):
  return sympy.Matrix([[L+2*M,L,0],[L,L+2*M,0],[0,0,M]])
  
class TestStiffElas:
  def __init__(self, cu,cv,la,mu):
    x, y = sympy.symbols('x y')
    self.cu=cu
    self.cv=cv
    self.L=la
    self.M=mu
    self.u=sympy.sympify(cu)
    self.v=sympy.sympify(cv)
    self.fu=eval("lambda x,y: numpy.array(["+cu[0]+","+cu[1]+"])")
    self.fv=eval("lambda x,y: numpy.array(["+cv[0]+","+cv[1]+"])")
    if self.u[0].is_polynomial(x,y) and self.u[1].is_polynomial(x,y):
      D0=sympy.polys.Poly(self.u[0],x,y).as_dict()
      D1=sympy.polys.Poly(self.u[1],x,y).as_dict()
      self.du=max(max(numpy.sum(list(D0.keys()),axis=1)),max(numpy.sum(list(D1.keys()),axis=1)))
    else:
      self.du=-1
    if self.v[0].is_polynomial(x,y) and self.v[1].is_polynomial(x,y):
      D0=sympy.polys.Poly(self.v[0],x,y).as_dict()
      D1=sympy.polys.Poly(self.v[1],x,y).as_dict()
      self.dv=max(max(numpy.sum(list(D0.keys()),axis=1)),max(numpy.sum(list(D1.keys()),axis=1)))
    else:
      self.dv=-1
    H=Hooke(self.L,self.M)
    gU=Gamma(self.u)
    gV=Gamma(self.v)
    self.I=sympy.integrate(sympy.integrate(gV.T*H*gU,(x,0,1)),(y,0,1))[0,0]
    
def validStiffElas2DP1(**kwargs):
  """ validation of elasticity stiffness matrix assembly functions 
  
  """
  Num=kwargs.get('Num',0)
  la=kwargs.get('la',1.5)
  mu=kwargs.get('mu',0.5)
  Plot=kwargs.get('plot',True)
  Verbose=kwargs.get('verbose',True)
  if Num not in [0,1,2,3]:
    print("Num parameter must be in [0,1,2,3]!")
    return
      
  print('**************************************************')
  print('*     2D StiffElas Assembling P1 validations     *')
  print('**************************************************')    
  (ext,cbase)=BasesChoice(Num)
  print("  Numbering Choice : %s" % cbase)
  print("  lambda, mu       : %g, %g" % (la,mu))
  
  Th=SquareMesh(30)
  qT=Th.q.T;meT=Th.me.T
  LP=[[['x - 2*y','x + y'],['x + 2*y','2*x - y'],la,mu],
      [['x**2+2*y*x+y','-2*y**2+x**2+x-y'],['3*x*y+y**2+1','3*x**2-x*y+1'],la,mu],
      [['x**3+2*y**2*x+y**2+x','y**3-2*x**2*y'],['2*x*y+y**3+x*y','3*x**3-2*x*y+x-1'],la,mu]]
  if Verbose:     
    print('-----------------------------------------')
    print('  Test 1: Matrices errors and CPU times  ')
    print('-----------------------------------------')

  T=numpy.zeros(3)
  E=numpy.zeros(2)
  tstart=time.time()
  Mbase=StiffElasAssembling2DP1base(Th.nq,Th.nme,Th.q,Th.me,Th.areas,la,mu,Num)
  T[0]=time.time()-tstart
  if Verbose: print("    Matrix size           : (%d,%d)" % Mbase.shape)
  tstart=time.time()
  MOptV1=StiffElasAssembling2DP1OptV1(Th.nq,Th.nme,Th.q,Th.me,Th.areas,la,mu,Num)
  T[1]=time.time()-tstart
  E[0]=NormInf(Mbase-MOptV1)
  if Verbose: print("    Error P1base vs OptV1 : %e" % E[0])
  tstart=time.time()
  MOptV2=StiffElasAssembling2DP1OptV2(Th.nq,Th.nme,qT,meT,Th.areas,la,mu,Num)
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
  deg=numpy.zeros(len(LP))
  E=numpy.zeros(len(LP))
  for i in range(len(LP)):
    test=TestStiffElas(LP[i][0],LP[i][1],LP[i][2],LP[i][3])
    M=StiffElasAssembling2DP1OptV2(Th.nq,Th.nme,qT,meT,Th.areas,test.L,test.M,Num)
    deg[i]=test.du+test.dv
    U=test.fu(Th.q[:,0],Th.q[:,1])
    V=test.fv(Th.q[:,0],Th.q[:,1])
    if Num==0 or Num==2:
      U=U.T;V=V.T
    U=U.reshape((2*Th.nq))
    V=V.reshape((2*Th.nq))
    Ifem=numpy.dot(M*U,V)
    E[i]=abs(Ifem-test.I)
    if Verbose: print("    function %d :\n      u(x,y)=[%s,%s],\n      v(x,y)=[%s,%s],\n           -> StiffElas error=%e" % (i,test.cu[0],test.cu[1],test.cv[0],test.cv[1],E[i]) );
  if checkTest2(deg,E)==1:
    sys.exit()
  if Verbose:
    print('--------------------------------')
    print('  Test 3: Validations by order  ')
    print('--------------------------------')
  LN=range(10,220,20)
  n=len(LN)
  h=numpy.zeros(n)
  Error=numpy.zeros(n)
  k=len(LP)-1
  test=TestStiffElas(LP[k][0],LP[k][1],LP[k][2],LP[k][3])
  if Verbose: print("    functions %d:\n      u(x,y)=%s,\n      v(x,y)=%s,\n      lambda=%f, mu=%f" %(k,test.cu,test.cv,test.L,test.M))
  for i in range(n):
    Th=SquareMesh(LN[i],version=1)
    tstart=time.time()
    M=StiffElasAssembling2DP1OptV2(Th.nq,Th.nme,Th.q,Th.me,Th.areas,test.L,test.M,Num)
    TT=time.time()-tstart
    U=test.fu(Th.q[0],Th.q[1])
    V=test.fv(Th.q[0],Th.q[1])
    if Num==0 or Num==2:
      U=U.T;V=V.T
    U=U.reshape((2*Th.nq))
    V=V.reshape((2*Th.nq))
    Ifem=numpy.dot(M*U,V)
    h[i]=GetMaxLengthEdges(Th.q.T,Th.me.T)
    Error[i]=abs(Ifem-test.I)
    if Verbose: 
      print("        Matrix size                          : (%d,%d)" % M.shape)
      print("        StiffElasAssemblingP1OptV2 CPU times : %3.3f(s)" % TT)
      print("        Error                                : %e" % Error[i]);
  if checkTest3(h,Error)==1:
    sys.exit() 
  if Plot:   
    PlotTest3(h,Error,'Test 3 : Elasticity Stiffness Matrix (2D/P1)')   
