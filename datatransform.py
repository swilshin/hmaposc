'''
Abstractions for applying transformations and keeping track of derivatives.

@author: Simon Wilshin
@contact: swilshin@rvc.ac.uk
@date: Feb 2013
'''
from formphase.formphaseutil import PCA

from numpy import array,median,std,sum,abs,einsum,diag,ones

class PCASpec(object):
  def __init__(self,Lda=None,U=None):
    self.Lda = Lda
    self.U = U
  
  def train(self,y):
    '''
    Train the PCA on the data in y.
    '''
    dump,self.Lda,self.U=PCA(y.swapaxes(-1,-2),True)
  
  @staticmethod
  def pcaspec(y):
    '''
    Make a new pca specification and train it.
    '''
    p = PCASpec()
    p.train(y)
    return(p)
    
  def __call__(self,y):
    return(einsum('...ij,...jl->...il',self.U,y))
  
  def inv(self,y):
    '''
    Since U.U^T=I, we can just apply the transpose.
    '''
    return(einsum('...ji,...jl->...il',self.U,y))
  
  def dapply(self,y):
    '''
    Has the same effect on derivatives.
    '''
    return(self(y))
  
  def dinv(self,y):
    '''
    Has the same effect on derivatives.
    '''    
    return(self.inv(y))
  
  def jacobian(self,y):
    return(self.U)
  
  def __repr__(self):
    return("PCASpec("+
      "Lda="+repr(self.Lda)+
      ",U="+repr(self.U)+ 
    ")")

class QuickScale(object):
  def __init__(self,sigma=None):
    self.sigma=sigma
    
  def train(self):
    pass

  def getV(self,y):
    v = ones(y.shape[-2])
    v[2:] *= self.sigma
    return(v)
  
  def __call__(self,y):
    v = self.getV(y)
    return((v*y.swapaxes(-1,-2)).swapaxes(-1,-2))
  
  def inv(self,y):
    v = self.getV(y)
    return((y.swapaxes(-1,-2)/v).swapaxes(-1,-2))

  def dapply(self,y):
    return(self(y))
  
  def dinv(self,y):
    return(self.inv(y))
  
  def jacobian(self,y):
    v = self.getV(y)
    return(diag(v))
  
  def __repr__(self):
    return("QuickScale("+
      "sigma="+repr(self.sigma)+
    ")")
    

class ZScoreSpec(object):
  '''
  Uses median and standard deviation to zscore data.
  '''
  def __init__(self,mu=None,sigma=None):
    self.mu = mu
    self.sigma = sigma

  def train(self,y):
    self.mu = median(y,1)
    self.sigma = std(y,1)    
    
  @staticmethod
  def zscorespec(y):
    z = ZScoreSpec()
    z.train(y)
    return(z)
  
  def __call__(self,y):
    return(((y.swapaxes(-1,-2)-self.mu)/self.sigma).swapaxes(-1,-2))
  
  def inv(self,y):
    return(((y.swapaxes(-1,-2)*self.sigma)+self.mu).swapaxes(-1,-2))
    
  def dapply(self,y):
    '''
    Derivatives are rescaled but not shifted by this transform.
    '''
    return((y.swapaxes(-1,-2)/self.sigma).swapaxes(-1,-2))
  
  def dinv(self,y):
    '''
    Derivatives are rescaled but not shifted by this transform.
    '''
    return((y.swapaxes(-1,-2)*self.sigma).swapaxes(-1,-2))
  
  def jacobian(self,y):
    return(diag(1.0/self.sigma))
  
  def __repr__(self):
    return("ZScoreSpec("+
      "mu="+repr(self.mu)+
      ",sigma="+repr(self.sigma)+ 
    ")")      

class PlanarZScoreSpec(ZScoreSpec):
  '''
  Uses median and standard deviation to zscore data.
  '''
  def __init__(self,mu=None,sigma=None):
    super(PlanarZScoreSpec,self).__init__(mu,sigma)

  def train(self,y):
    self.mu = median(y,1)
    self.sigma = std(y,1)
    self.sigma[2:] = self.sigma[1]
    
  @staticmethod
  def planarzscorespec(y):
    z = PlanarZScoreSpec()
    z.train(y)
    return(z)
    
  def __repr__(self):
    return("PlanarZScoreSpec("+
      "mu="+repr(self.mu)+
      ",sigma="+repr(self.sigma)+ 
    ")")      


class DataTransformStack(object):
  def __init__(self,T=None):
    if T is not None:
      self.T = list(T)
    else:
      self.T = None
  
  def addTransform(self,t,y):
    if self.T is None:
      self.T=list()
    self.T.append(t(y))
  
  def __call__(self,x):
    y = array(x).copy()
    for t in self.T:
      y = t(y)
    return(y)
  
  def inv(self,x):
    y = array(x).copy()
    for t in reversed(self.T):
      y = t.inv(y)
    return(y)
  
  def dapply(self,x):
    y = array(x).copy()
    for t in self.T:
      y = t.dapply(y)
    return(y)
  
  def dinv(self,x):
    y = array(x).copy()
    for t in reversed(self.T):
      y = t.dinv(y)
    return(y)
  
  def jacobian(self,y):
    J = self.T[0].jacobian(y)
    for t in self.T[1:]:
      J = dot(t.jacobian(y),J)
    return(J)
  
  def __repr__(self):
    return("DataTransformStack("+
      "T="+repr(self.T)+
    ")")
