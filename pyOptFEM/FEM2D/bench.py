from numpy import *
from . import *
#from FEM2Dtests import *
import time, socket, sys, os, errno, platform
            
AssemblyList = { 
                 'MassAssembling2DP1'      : 1,
                 'StiffAssembling2DP1'     : 2,
                 'StiffElasAssembling2DP1' : 4
               }

VersionList = ['OptV2a', 'OptV2' , 'OptV1', 'base' ]

bench_defaults = {  'LN':     range(50, 90, 10),
              'meshdir':     '.',
              'meshbase':    '',
              'outdir':      './results',
              'output':      'bench2D',
              'save'  :      False,
              'plot'   :     True,
              'assembly':    'StiffAssembling2DP1',
              'versions':     VersionList,
              'tag'     :     '',
              'nbruns':      1,
              'la':          2,   # lambda
              'mu':          0.3, # mu
              'Num':         0
            }          

def bench(**kwargs):
  """ Benchmark code for :math:`P_1`-Lagrange finite elements Matrices defined in :mod:`FEM2D`.
  
  
  """
  LN=kwargs.get('LN',    bench_defaults['LN'] )
  meshdir=kwargs.get('meshdir', bench_defaults['meshdir'])
  meshbase=kwargs.get('meshbase', bench_defaults['meshbase'])
  Release=sys.version.split(' ')[0]
  Hostname=socket.gethostname()
  outdir=kwargs.get('outdir',bench_defaults['outdir'])+'/'+Hostname+'/Python_'+Release
  output=kwargs.get('output',bench_defaults['output'])
  save=kwargs.get('save',bench_defaults['save'])
  plot=kwargs.get('plot',bench_defaults['plot'])
  assembly=kwargs.get('assembly',bench_defaults['assembly'])
  versions=kwargs.get('versions',bench_defaults['versions'])
  nbruns=kwargs.get('nbruns',bench_defaults['nbruns'])
  la=kwargs.get('la',bench_defaults['la'])
  mu=kwargs.get('mu',bench_defaults['mu'])
  Num=kwargs.get('Num',bench_defaults['Num'])
  tag=kwargs.get('tag',bench_defaults['tag'])
  Date=time.strftime("%d_%m_%Y_%H_%M_%S")
  ext,cbase=BasesChoice(Num)
  if not(assembly in AssemblyList):
    print("function %s not implemented!" % assembly)
    print("  <assembly> value must be in %s." % str(list(AssemblyList.keys())))
    return
  
  if (tag==''):
    tag=Date
    
  if (meshbase==''):
    Meshbase='SquareMesh'
  else:
    Meshbase=meshbase
    
  if versions.__class__.__name__!='list':
    print("versions paramater must be a <list> class!")
    return
    
  if not checkVersions(versions,VersionList):
    print("versions contents not allowed!")
    print("  <versions> values must be in %s" % str(VersionList))
    return
    
  nV=len(versions)
  versionsname=versions[0]
  for v in range(1,nV):
    versionsname+='.'+versions[v]
  Hostname=socket.gethostname()
  if AssemblyList[assembly]==4:
    ext,cbase=BasesChoice(Num)
    filename=outdir+'/'+output+'_'+assembly+versionsname+ext+'_Python_'+Release+'_'+Meshbase+'_'+Hostname+'_'+tag+'.txt'
  else:
    filename=outdir+'/'+output+'_'+assembly+versionsname+'_Python_'+Release+'_'+Meshbase+'_'+Hostname+'_'+tag+'.txt'
  if save:
    mkdir_p(outdir)
    f = open(filename, 'w')
    f.write('% Bench           : bench2D\n')
    f.write('% Software        : python\n')
    f.write('% Version         : '+Release+'\n')
    f.write('% Computer name   : '+Hostname+'\n')
    f.write('% System          : '+platform.system()+' '+platform.machine()+'\n')
    f.write('% Assembly matrix : '+assembly+'\n')
    f.write('% code versions   : '+str(versions)+'\n')
    if AssemblyList[assembly]==4:
      f.write('% lambda          : '+str(la)+'\n')
      f.write('% mu              : '+str(mu)+'\n')
      f.write('% Num             : '+str(Num)+'\n')
      f.write('%      -> '+cbase+'\n')
    if meshbase=='':
      f.write('% Mesh files      : '+Meshbase+'(N)\n')
    else:
      f.write('% Mesh files      : '+meshdir+'/'+meshbase+'-<N>.msh\n')
      f.write('% Mesh format     : FreeFEM\n')
    f.write('% Number of runs  : '+str(nbruns)+'\n')
    f.write('% List of N       : '+str(LN)+'\n');
    f.write('%-------------------------------\n')
    f.write('% N       nq     ndof   ')  
    for i in range(0,nV):
      f.write('   '+versions[i]+'(s)')
    f.write('\n')
  nN=len(LN)
  T=zeros((nN,nV))
  Lnq=zeros(nN)
  Lndof=zeros(nN)
  n=0;
  for N in LN:
    if (meshbase==''):
      Th=SquareMesh(N)
      meshfile='SquareMesh('+str(N)+')'
    else:
      meshfile=meshdir+'/'+meshbase+'-'+str(N)+'.msh'
      Th=Mesh(meshfile)
   
    Lnq[n]=Th.nq
    for v in range(0,nV):
      T[n,v],dim=RunVersion(meshfile,assembly,versions[v],Th,la,mu,Num,nbruns)
      if nbruns>1:
        print(" -> mean cputime=%.6f(s)" %(T[n,v]))
    Lndof[n]=dim[0]
    if save:
      f.write('%4d %8d %8d' % (N,Lnq[n],Lndof[n]))
      for v in range(0,nV):
        f.write('   %.6f' % T[n,v])
      f.write('\n')
    n+=1
  PrintResults(versions,LN,Lnq,Lndof,T)
  if save:
    f.close()
    print("Write results in file :\n  %s" % filename)
  if plot:
    plotBench(versions,Lndof,T)
  #return (versions,Lndof,T)
  
def RunVersion(meshfile,assembly,version,Th,la,mu,Num,nbruns):
  print("%s:\n -> %s(nq=%d, nme=%d)" %(meshfile,assembly+version,Th.nq,Th.nme))
  Tmean=0
  for j in range(0,nbruns):
    if AssemblyList[assembly]==1: #Mass
      tstart=time.time()
      R=globals()[assembly+version](Th.nq,Th.nme,Th.me,Th.areas)
      T=time.time()-tstart
    elif AssemblyList[assembly]==2: # Stiff
      tstart=time.time()
      R=globals()[assembly+version](Th.nq,Th.nme,Th.q,Th.me,Th.areas)
      T=time.time()-tstart
    elif AssemblyList[assembly]==4: #StiffElas
      tstart=time.time()
      R=globals()[assembly+version](Th.nq,Th.nme,Th.q,Th.me,Th.areas,la,mu,Num)
      T=time.time()-tstart
    print("  run (%2d/%2d) : cputime=%.6f(s) - matrix %d-by-%d" %(j+1,nbruns,T,R.shape[0],R.shape[1]));
    Tmean+=T
  return Tmean/nbruns,R.shape
  
def PrintResults(versions,LN,Lnq,Lndof,T):
  nV=len(versions)
  nN=len(LN)
  S='  N       nq     ndof   ';
  for i in range(0,nV):
    S=S+'   '+versions[i]+'(s)'
  Title=" Bench results "
  nT=int(round(len(S)-len(Title))/2+1)
  print('='*nT+Title+'='*nT)
  S='  N       nq     ndof   ';
  for i in range(0,nV):
    S=S+'   '+versions[i]+'(s)'
  print(S)
  
  for n in range(0,nN):
    S='%4d %8d %8d' % (LN[n],Lnq[n],Lndof[n])
    X='                       '
    for v in range(0,nV):
      S+='   %.6f' % T[n,v]
      X+='     x%4.2f' % (T[n,v]/T[n,0])
    print(S)
    print(X)
  print("="*(2*nT+len(Title)))
  
def checkVersions(versions,VersionList):
  for i in range(0,len(versions)):
    if versions[i] not in VersionList:
      return False
  return True
  
def plotBench(versions,Lndof,T):
  import matplotlib.pyplot as plt
  nV=len(versions)
  for i in range(0,nV):
    plt.loglog(Lndof,T[:,i],label=versions[i])
  plt.loglog(Lndof,1.2*max(T[0,:])*Lndof/Lndof[0],'k--',label="$O(n_{dof})$")
  plt.loglog(Lndof,mean(T[0,:])*Lndof**2/(Lndof[0]**2),'k.-',label="$O(n_{dof}^2)$")
  #plt.legend(loc='lower right')
  #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
  plt.grid()
  plt.xlabel('$n_{dof}$')
  plt.ylabel('cputime(s)')
  plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
       ncol=nV+2, mode="expand", borderaxespad=0.)
  plt.show()
  
  