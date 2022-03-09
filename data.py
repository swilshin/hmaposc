'''
Data sets.

@author: Simon Wilshin
@contact: swilshin@rvc.ac.uk
@date: Feb 2013
'''
from __future__ import print_function

from formphase.hmaposc import (FloquetCoordinateOsc,FloquetCoordinateSim,
  FloquetCoordinateOscFactory,HMapFactory,HMap,HMapChainFactory,HMapChain,
  HFloquetSystem,HFloquetSystemFactory,HFloquetSim,SimSpec,Noise,Times,
  saveSimulation,loadSimulation,FMap,FMapFactory)
from formphase.formphaseutil import (kalmanWrapper,serialiseArray,
  unserialiseArray,paddedHilbert,EllipseReflectorFactory,EllipseReflector,
  kalman,Kalman,PCA,FourierSeries,TangentVectors)
from formphase.hmaposc import (HFloquetSystem,HMap,HMapChain)
from .datatransform import (PCASpec,ZScoreSpec,DataTransformStack,
  PlanarZScoreSpec,QuickScale)

from numpy import (array,hstack,median,std,flipud,zeros,
  eye,sum,pi,abs,diff,arange,einsum,ones)
from scipy.interpolate import UnivariateSpline

from os.path import join as osjoin
from os import listdir

from os.path import split as ossplit
from os.path import join as osjoin
from os.path import isfile

class DataSet(object):
  def __init__(self,H,kal,trials,source,psfidx=0):
    self.H = H
    self.kal = kal
    self.trials = trials
    self.source = source
    
    self.psf = zeros(self.H.sim.D)
    self.psf[psfidx] = 1.0
    
  def __repr__(self):
    return("DataSet("+
      "H="+repr(self.H)+
      ",kal="+repr(self.kal)+
      ",trials="+repr(self.trials)+
      ",source="+repr(self.source)+
    ")")

class DataSetFactory(object):
  def __init__(self,source,pca,fd,psfidx=0):
    # Load the prototype
    self.source = source
    self.noExtSource = source[:-7]
    self.sourceDir,self.noExtSourceFile = ossplit(self.noExtSource)
    
    # Make a list of files under consideration
    self.fles = listdir(self.sourceDir)
    self.pca = pca
    self.fd = fd
    self.psfidx=psfidx
  
  def __call__(self,Ntrain):
    d = loadSimulation(self.source)
    H = eval(d[0])
    
    # Check if a kalman filter exists, if not create one
    kalFile = self.noExtSource+"_kal.py"
    if isfile(kalFile):
      with open(kalFile,'r') as f:
        kal = eval(f.read())
    else:
      kal=kalman(d[-2].T)
      with open(kalFile,'w') as f:
        f.write(repr(kal))
    
    # Load data generated from same system
    self.sims = list()
    for f in self.fles:
      if f[-7:]==".pkl.gz" and f[-13:]!='_cache.pkl.gz':
        try:
          s = loadSimulation(osjoin(self.sourceDir,f))
          if eval(s[0]).spec==H.spec:
            self.sims.append(s)
        except EOFError:
          print("Failed to open", f, "skipping.")
    
    # Grab the trials from the loaded data
    x,y = zip(*[(s[2],s[-2]) for s in self.sims])
    trials = Trials(x,y,Ntrain)
    trials.initialise(kal,self.pca,self.fd)
    return(DataSet(H,kal,trials,self.source,self.psfidx))
    
  def __repr__(self):
    '''
    Note this is not a genuine serialisation like other methods in the 
    formphase package since instantiation of this object depends on the 
    existence of files in the file system, and behaviour depends on if 
    things like the kalman filter have already been computed.
    '''
    return("DataSetFactory("+
      "source="+repr(self.source)+
      ",pca="+repr(self.pca)+
      ",fd="+repr(self.fd)+
      ",psfidx="+repr(self.psfidx)+      
    ")")


class Trials(object):
  def __init__(self,x,y,Ntrain,maxN=None,ytf=None,ydf=None,ytp=None,ydp=None,
    yt=None,yd=None,dtran=None
  ):
    if type(x)==str:
      self.x = unserialiseArray(x)
      self.y = unserialiseArray(y)
    else:
      self.x = array(x)
      self.y = array(y)
    self.Ntrain = Ntrain
    self.maxN = maxN
    self.yt0 = self.y[:Ntrain]
    self.yd0 = self.y[Ntrain:self.maxN]
    self.xt = self.x[:Ntrain]
    self.xd = self.x[Ntrain:self.maxN]
    # Filtered
    self.ytf = ytf
    self.ydf = ydf
    # Z-scored (normalized)
    self.yt = yt
    self.yd = yd
    # PCAed
    self.ytp = ytp
    self.ydp = ydp
    if dtran is None:
      self.dtran = DataTransformStack()
    else:
      self.dtran = dtran
  
  @staticmethod
  def _fd(y0):
    '''
    Calculate the derivative of y0 using finite differences and a univariate 
    spline.
    '''
    y = list()
    for iy in y0:
      diy = diff(iy,axis=0)
      xint = arange(iy.shape[0])
      x0 = xint[:-1]+0.5
      diy = array([UnivariateSpline(x0,idiy,s=0)(xint) for idiy in diy.T]).T
      y.append(hstack([iy,diy]))
    return(array(y))
  
  def _finitediff(self):
    '''
    Calculate derivatives using finite differences. Use this if the kalman 
    filter is under performing due to non-linearities.
    '''
    self.ytf = self._fd(self.yt0)
    self.ytf = TangentVectors(
      self.ytf[:,:,:self.ytf.shape[-1]/2].transpose(0,2,1),
      self.ytf[:,:,self.ytf.shape[-1]/2:].transpose(0,2,1)
    )
    
    self.ydf = self._fd(self.yd0)
    self.ydf = TangentVectors(
      self.ydf[:,:,:self.ydf.shape[-1]/2].transpose(0,2,1),
      self.ydf[:,:,self.ydf.shape[-1]/2:].transpose(0,2,1)
    )    
  
  @staticmethod
  def _kf(y0,kal):
    '''
    Apply a kalman kal filter to a specific set of trajectories in y0
    '''
    y = list()
    for iy in y0:
      K = kal.filter(flipud(iy).T,zeros(2*iy.shape[1]),eye(2*iy.shape[1]))
      ix = K[0][:,-1]
      iV = K[1][:,:,-1]
      y.append(kal.smooth(iy.T,ix,iV)[0].T)
    return(array(y))
  
  def _kalfilt(self,kal):
    '''
    Apply a Kalman filter
    '''
    self.ytf = self._kf(self.yt0,kal)
    self.ytf = TangentVectors(
      self.ytf[:,:,:self.ytf.shape[-1]/2].transpose(0,2,1),
      self.ytf[:,:,self.ytf.shape[-1]/2:].transpose(0,2,1)
    )
    
    self.ydf = self._kf(self.yd0,kal)
    self.ydf = TangentVectors(
      self.ydf[:,:,:self.ydf.shape[-1]/2].transpose(0,2,1),
      self.ydf[:,:,self.ydf.shape[-1]/2:].transpose(0,2,1)
    )
          
  def _zscore(self):
    self.muy = median(self.ytp.getFlatX(),1)
    self.sigmay = std(self.ytp.getFlatX(),1)
    
    self.yt = TangentVectors(
      ((self.ytp.getX().transpose(0,2,1)-self.muy)/self.sigmay).transpose(0,2,1),
      (self.ytp.getdX().transpose(0,2,1)/self.sigmay).transpose(0,2,1)
    )
    self.yd = TangentVectors(
      ((self.ydp.getX().transpose(0,2,1)-self.muy)/self.sigmay).transpose(0,2,1),
      (self.ydp.getdX().transpose(0,2,1)/self.sigmay).transpose(0,2,1)
    )

  def _pca(self):
    y,self.Lda,self.U=PCA(self.ytf.getFlatX().T,True)
    self.ytp = TangentVectors(
      einsum('...ij,...jl->...il',self.U,self.ytf.getX()),
      einsum('...ij,...jl->...il',self.U,self.ytf.getdX())
    )
    self.ydp = TangentVectors(
      einsum('...ij,...jl->...il',self.U,self.ydf.getX()),
      einsum('...ij,...jl->...il',self.U,self.ydf.getdX())  
    )
    
  def initialise(self,kal,pca=True,fd=False):
    '''
    Perform initialisation, applying the kalman filter kal, z-scoring and 
    doing a PCA
    '''
    if fd:
      self._finitediff()
    else:
      self._kalfilt(kal)
    if pca:
      #self._pca()
      self.dtran.addTransform(PCASpec.pcaspec,self.ytf.getFlatX())
      self.ytp = TangentVectors(
        self.dtran(self.ytf.getX()),
        self.dtran.dapply(self.ytf.getdX())
      )
      self.ydp = TangentVectors(
        self.dtran(self.ydf.getX()),
        self.dtran.dapply(self.ydf.getdX())
      )      
    else:
      #self.Lda = ones((self.y.shape[-1]))
      #self.U = eye(self.y.shape[-1])
      self.ytp = self.ytf
      self.ydp = self.ydf
    #self.dtran.addTransform(ZScoreSpec.zscorespec,self.ytp.getFlatX())
    self.dtran.addTransform(PlanarZScoreSpec.planarzscorespec,self.ytp.getFlatX())
    self.dtran.addTransform(QuickScale,0.1)
    self.yt = TangentVectors(
      self.dtran(self.ytf.getX()),
      self.dtran.dapply(self.ytf.getdX())
    )
    self.yd = TangentVectors(
      self.dtran(self.ydf.getX()),
      self.dtran.dapply(self.ydf.getdX())
    )
    #self._zscore()
  
  def __repr__(self):
    return("Trials("+
      "x="+repr(serialiseArray(self.x))+
      ",y="+repr(serialiseArray(self.y))+
      ",Ntrain="+repr(self.Ntrain)+
      ",maxN="+repr(self.maxN)+
      ",ytf="+repr(self.ytf)+
      ",ydf="+repr(self.ydf)+
      ",ytp="+repr(self.ytp)+
      ",ydp="+repr(self.ydp)+
      ",yt="+repr(self.yt)+
      ",yd="+repr(self.yd)+
      ",dtran="+repr(self.dtran)+
    ")")

