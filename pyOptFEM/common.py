from scipy import sparse
import matplotlib.pyplot as plt
import  os, errno, ctypes
from numpy import log

def NormInf(A):
  """This function returns the norm Inf of a *Scipy* sparse Matrix

  :param A: A *Scipy* sparse matrix 
  :returns: norm Inf of A given by :math:`\| A\|_\infty=\max_{i,j}(|A_{i,j}|)`.
  """
  if (A.data.shape[0]==0):
    return 0
  else:
    return max(abs(A.data))
    
    
def showSparsity(M):
#  from matplotlib.pyplot as plt
  plt.spy(M, precision=1e-8, marker='.', markersize=3)
  plt.show()
  
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise
        
def BasesChoice(Num):
    if   Num==0:
      ext='BaBa'
      cbase='global alternate numbering with local alternate numbering'
    elif Num==1:
      ext='BaBb'
      cbase='global block numbering with local alternate numbering'
    elif Num==2:
      ext='BbBa'
      cbase='global alternate numbering with local block numbering'
    elif Num==3:
      ext='BbBb'
      cbase='global block numbering with local block numbering'
    return ext,cbase   
    
def PrintResultsSphinx(versions,LN,Lnq,Lndof,T):
  nV=len(versions)
  nN=len(LN)
  Sep1='+{:-^8}'.format("")*3 + '+{:-^14}'.format("")*nV+'+'
  Sep2='+{:=^8}'.format("")*3 + '+{:=^14}'.format("")*nV+'+'
  Sep3='|{:^8}'.format("")*3 +  '+{:-^14}'.format("")*nV+'+'
  Tit='|{:^8}'.format('N')+'|{:^8}'.format('nq')+'|{:^8}'.format('ndof')
  for i in range(0,nV):
    Tit+='|{:^14}'.format(versions[i])
  Tit+='|'
  print(Sep1)
  print(Tit)
  print(Sep2)
  
  for n in range(0,nN):
    S1='|{:^8}'.format('%d' % LN[n])+'|{:^8}'.format('%d' % Lnq[n])+'|{:^8}'.format('%d' % Lndof[n])
    S2='|{:^8}'.format("")*3
    for v in range(0,nV):
      S1+='|{:^14}'.format('%.4f(s)' % T[n,v])
      if (T[n,0]<1e-6):
        S2+='|{:^14}'.format('x%s' % ('NaN'))
      else:
        S2+='|{:^14}'.format('x%4.2f' % (T[n,v]/T[n,0]))
    S1+='|'
    S2+='|'
    print(S1)
    print(Sep1)
    print(S2)
    print(Sep1)
  
def PrintResultsLatexTabular(FileName,versions,LN,Lnq,Lndof,T):
  nV=len(versions)
  nN=len(LN)
  fp = open(FileName, 'wt')
  fp.write(format('\\begin{tabular}{@{}|r|r||*{%d}{@{}c@{}|}@{}}\n' % nV))
  fp.write('  \\hline\n')
  fp.write('  $n_q$ & $n_{dof}$')
  for v in range(0,nV):
    fp.write(' & '+versions[v])
  fp.write('  \\\\ \\hline \\hline\n')
  
  for n in range(0,nN):
    fp.write(format('  $%d$ & $%d$ ' % (Lnq[n],Lndof[n])))
    for v in range(0,nV):
      if T[n,0]<1e-8:
        fp.write(format('& \\begin{tabular}{c} %.3f (s) \\\\ \\texttt{x %s} \\end{tabular} ' %(T[n,v],'NaN')))
      else:
        fp.write(format('& \\begin{tabular}{c} %.3f (s) \\\\ \\texttt{x %.3f} \\end{tabular} ' %(T[n,v],T[n,v]/T[n,0])))
    fp.write('\\\\ \\hline\n')
  fp.write('\\end{tabular}')
  
def checkVersions(versions,VersionList):
  for i in range(0,len(versions)):
    if versions[i] not in VersionList:
      return False
  return True
  
def plotBench(versions,Lndof,T):
  import matplotlib.pyplot as plt
  nV=len(versions)
  
  if T.min()<1e-8:
    return 0
  plt.loglog(Lndof,T[0,0]*Lndof/Lndof[0],'k--',label="$O(n)$")
  plt.loglog(Lndof,T[0,0]*Lndof*log(Lndof)/(Lndof[0]*log(Lndof[0])),'k.-',label="$O(nlog(n))$")
  for i in range(1,nV):
    plt.loglog(Lndof,T[0,i]*Lndof/Lndof[0],'k--')
    plt.loglog(Lndof,T[0,i]*Lndof*log(Lndof)/(Lndof[0]*log(Lndof[0])),'k.-')

  for i in range(0,nV):
    plt.loglog(Lndof,T[:,i],label=versions[i])
    
  #plt.legend(loc='lower right')
  #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
  plt.grid()
  plt.xlabel('$n=n_{dof}$')
  plt.ylabel('cputime(s)')
  if nV<=3:
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=nV+2, mode="expand", borderaxespad=0.)
  else:
    plt.legend(loc='upper left')
  return plt
  
def printReport(FileName,assembly,Release):
  fp = open(FileName+'_report.tex', 'wt')
  basename=os.path.basename(FileName)
  PWD=os.path.realpath('.')
  fp.write('\\documentclass{article}\n');
  fp.write(format('\\input{%s/report.sty}\n' % PWD));
  fp.write(format('\\title{Automatic bench report  : \\texttt{%s} functions under Python (%s)  }\n' % (assembly,Release)))
  fp.write('\\begin{document}\n');
  fp.write('\\maketitle\n');
  fp.write(format('\\inputtabular{%s}\n{cputimes and speedup}\n\n' % basename+'.tex'))
  fp.write(format('\\imageps{%s}{0.5}\n' % basename+'.eps'))
  fp.write('\\end{document}\n')
  
class memoryCheck():
    """Checks memory of a given system"""
 
    def __init__(self):
 
        if os.name == "posix":
            self.value = self.linuxRam()
        elif os.name == "nt":
            self.value = self.windowsRam()
        else:
            print("I only work with Win or Linux :P")
 
    def windowsRam(self):
        """Uses Windows API to check RAM in this OS"""
        kernel32 = ctypes.windll.kernel32
        c_ulong = ctypes.c_ulong
        class MEMORYSTATUS(ctypes.Structure):
            _fields_ = [
                ("dwLength", c_ulong),
                ("dwMemoryLoad", c_ulong),
                ("dwTotalPhys", c_ulong),
                ("dwAvailPhys", c_ulong),
                ("dwTotalPageFile", c_ulong),
                ("dwAvailPageFile", c_ulong),
                ("dwTotalVirtual", c_ulong),
                ("dwAvailVirtual", c_ulong)
            ]
        memoryStatus = MEMORYSTATUS()
        memoryStatus.dwLength = ctypes.sizeof(MEMORYSTATUS)
        kernel32.GlobalMemoryStatus(ctypes.byref(memoryStatus))
 
        return int(memoryStatus.dwTotalPhys/1024**2)
 
    def linuxRam(self):
        """Returns the RAM of a linux system"""
        totalMemory = os.popen("free -m").readlines()[1].split()[1]
        return int(totalMemory)
