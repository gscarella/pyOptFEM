import numpy as np
import time, socket, sys, os, errno, platform
from .assembly import *
from .mesh import *
from ..common import *

VersionList = ['OptV2' , 'OptV1', 'base' ]

AssemblyList = { 'StiffElasAssembling3DP1' : 4,
                 'MassAssembling3DP1' : 1,
                 'StiffAssembling3DP1' : 2
               }
bench_defaults = {  'LN':     range(5, 14, 2),
              'meshdir':     '.',
              'meshbase':    '',
              'outdir':      './results',
              'output':      'bench3D',
              'save'  :      False,
              'plot'   :     True,
              'verbose':     True,
              'assembly':    'StiffAssembling3DP1',
              'versions':     ['OptV2' , 'OptV1', 'base' ],
              'tag'     :     '',
              'nbruns':      1,
              'la':          2,   # lambda
              'mu':          0.3, # mu
              'Num':         0
            }
    
def assemblyBench(**kwargs):
  LN=kwargs.get('LN',    bench_defaults['LN'] )
  meshdir=kwargs.get('meshdir', bench_defaults['meshdir'])
  meshbase=kwargs.get('meshbase', bench_defaults['meshbase'])
  Release=sys.version.split(' ')[0]
  Hostname=socket.gethostname()
  outdir=kwargs.get('outdir',bench_defaults['outdir'])+'/'+Hostname+'/Python_'+Release
  output=kwargs.get('output',bench_defaults['output'])
  save=kwargs.get('save',bench_defaults['save'])
  Plot=kwargs.get('plot',bench_defaults['plot'])
  assembly=kwargs.get('assembly',bench_defaults['assembly'])
  versions=kwargs.get('versions',bench_defaults['versions'])
  nbruns=kwargs.get('nbruns',bench_defaults['nbruns'])
  la=kwargs.get('la',bench_defaults['la'])
  mu=kwargs.get('mu',bench_defaults['mu'])
  Num=kwargs.get('Num',bench_defaults['Num'])
  tag=kwargs.get('tag',bench_defaults['tag'])
  Date=time.strftime("%d_%m_%Y_%H_%M_%S")
  Verbose=kwargs.get('verbose',bench_defaults['verbose'])
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
  memScalar,memVector=memoryNeeded(max(LN),meshbase,meshdir)
  memRAM=memoryCheck().value
  if Verbose or (1.3*memScalar/10**6>=memRAM) or (1.3*memVector/10**6>=memRAM):
    print("Minimal memory needed (Scalar): %.2f Mo (RAM : %.2f Mo)" % (memScalar/10**6,memRAM))
    print("Minimal memory needed (Vector): %.2f Mo (RAM : %.2f Mo)" % (memVector/10**6,memRAM))
    
  
  nV=len(versions)
  versionsname=versions[0]
  for v in range(1,nV):
    versionsname+='.'+versions[v]
  Hostname=socket.gethostname()
  if AssemblyList[assembly]==4:
    ext,cbase=BasesChoice(Num)
    filename=outdir+'/'+output+'_'+assembly+versionsname+ext+'_Python_'+Release+'_'+meshbase+'_'+Hostname+'_'+tag+'.txt'
  else:
    filename=outdir+'/'+output+'_'+assembly+versionsname+'_Python_'+Release+'_'+meshbase+'_'+Hostname+'_'+tag+'.txt'
  if save:
    mkdir_p(outdir)  
    f = open(filename, 'w')
    f.write('% Bench           : bench3D\n')
    f.write('% Software        : python\n')
    f.write('% Version         : '+Release+'\n')
    f.write('% Date            : '+Date+'\n')
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
      f.write('% Mesh format     : medit\n')
    f.write('% Mesh files      : '+meshdir+'/'+meshbase+'-<N>.msh\n')
    f.write('% Number of runs  : '+str(nbruns)+'\n')
    f.write('% List of N       : '+str(LN)+'\n');
    f.write('%-------------------------------\n')
    f.write('% N       nq     ndof   ')  
    for i in range(0,nV):
      f.write('   '+versions[i]+'(s)')
    f.write('\n')
  nN=len(LN)
  T=np.zeros((nN,nV))
  Lnq=np.zeros(nN)
  Lndof=np.zeros(nN)
  n=0;
  for N in LN:
    if (meshbase==''):
      Th=CubeMesh(N)
      meshfile='CubeMesh('+str(N)+')'
    else:
      meshfile=meshdir+'/'+meshbase+'-'+str(N)+'.mesh'
      Th=getMesh(meshfile)
    Lnq[n]=Th.nq
    for v in range(0,nV):
      T[n,v],dim=RunVersion(meshfile,assembly,versions[v],Th,la,mu,Num,nbruns,Verbose)
      if nbruns>1:
        print(" -> mean cputime=%.6f(s)" %(T[n,v]))
    Lndof[n]=dim[0]
    if save:
      f.write('%4d %8d %8d' % (N,Lnq[n],Lndof[n]))
      for v in range(0,nV):
        f.write('   %.6f' % T[n,v])
      f.write('\n')
    n+=1
  print('\nBENCH RESUME :')
  print('    Assembly matrix : '+assembly)
  print('    code versions   : '+str(versions)+'\n') 
  PrintResultsSphinx(versions,LN,Lnq,Lndof,T)
  if save:
    f.close()
    print("Write results in file :\n  %s" % filename)
  if Plot:
    plt=plotBench(versions,Lndof,T)
    if isinstance(plt,int):
      print('Unable to plot bench results')
    else:
      plt.show()

 
def RunVersion(meshfile,assembly,version,Th,la,mu,Num,nbruns,Verbose):
  if Verbose: print("%s:\n -> %s(nq=%d, nme=%d)" %(meshfile,assembly+version,Th.nq,Th.nme))
  Tmean=0
  for j in range(0,nbruns):
    if AssemblyList[assembly]==1: #Mass
      tstart=time.time()
      R=globals()[assembly+version](Th.nq,Th.nme,Th.me,Th.volumes)
      T=time.time()-tstart
    elif AssemblyList[assembly]==2: # Stiff
      if version=='OptV2':
        q=Th.q.T;me=Th.me.T
      else:
       q=Th.q;me=Th.me
      tstart=time.time()
      R=globals()[assembly+version](Th.nq,Th.nme,q,me,Th.volumes)
      T=time.time()-tstart
    elif AssemblyList[assembly]==4: #StiffElas
      if version=='OptV2':
        q=Th.q.T;me=Th.me.T
      else:
        q=Th.q;me=Th.me
      tstart=time.time()
      R=globals()[assembly+version](Th.nq,Th.nme,q,me,Th.volumes,la,mu,Num)
      T=time.time()-tstart
    if Verbose: print("  run (%2d/%2d) : cputime=%.6f(s) - matrix %d-by-%d" %(j+1,nbruns,T,R.shape[0],R.shape[1]));
    Tmean+=T
  return Tmean/nbruns,R.shape

  
def memoryNeeded(N,meshbase,meshdir):
  if (meshbase==''):
    Th=CubeMesh(N)
    meshfile='CubeMesh('+str(N)+')'
  else:
    meshfile=meshdir+'/'+meshbase+'-'+str(N)+'.mesh'
    Th=getMesh(meshfile)
  memMesh=Th.q.nbytes+Th.me.nbytes+Th.volumes.nbytes
  d=3;
  # Kg,Ig,Jg arrays + sparse nnz=17*nq
  memSparse = 17*Th.nq*(8+4) + Th.nq*4
  memScalar = (d+1)*(d+1)*Th.nme*(8+4+4) 
  memVector = d*d*memScalar
  return (memScalar+memMesh+memSparse,memVector+memMesh+d*memSparse)  