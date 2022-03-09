'''
For generating and applying Henon type maps. Originally
used to distort test signals for form phase.

@author: Simon Wilshin
@contact: swilshin@rvc.ac.uk
@date: Feb 2013
'''
from __future__ import print_function

from formphase.formphaseutil import (getRandomOrthogonal,getRandomAffine,
  EllipseReflector,invAffine,applyAffine,getRandomPositiveDef,
  serialiseArray,unserialiseArray,FourierSeries)

from numpy import (min,max,hstack,dot,array,asfarray,linspace,sqrt,vstack,
  meshgrid,cos,sin,pi,sum,zeros,eye,einsum,tile,where,unwrap,arctan2,nan,
  ones_like)
from numpy.random import randint,randn
from numpy.linalg import inv 

'''
from hmaposc import FMapFactory,HMapFactory
from formphaseutil import EllipseReflectorFactory

F = FMapFactory()
f = F(8)
tau = linspace(0,100,1000)
x = zeros((8,1000))
x[1] = sin(tau)
x[0] = cos(tau)
y = f(x.T)

H = HMapFactory(EllipseReflectorFactory())
h = H(8)
tau = linspace(0,100,1000)
y = f(x.T)

Hfsys=HMapChainFactory(hf=HMapFactory(erFac=EllipseReflectorFactory(invMax=1.01,invMin=0.99,beta=-1.0,m=0.5,scaleFactor=0.0,yielded=0),affMag=0.15),N=20,orthMag=0.15,FF=FMapFactory(c=0.0))
Hfsys=HMapChainFactory(hf=HMapFactory(erFac=EllipseReflectorFactory(invMax=1.01,invMin=0.99,beta=-1.0,m=0.1,scaleFactor=0.0),affMag=0.15),N=100,orthMag=0.15)
'''

class FMap(object):
  def __init__(self,F):
    self.F = F
  
  def __call__(self,tx):
    x = tx[:,0]
    y = tx[:,1]
    phi = unwrap(arctan2(y,x))
    nx = nan*ones_like(tx)
    nx[:,0] = x
    nx[:,1] = y
    nx[:,2:] = tx[:,2:] + self.F.val(phi).real
    return(nx)
  
  def derv(self,tx):
    D = tx.shape[1]
    N = tx.shape[0]
    J = eye(D).repeat(N,-1).reshape(D,D,N).transpose(2,0,1)
    x = tx[:,0]
    y = tx[:,1]
    phi = unwrap(arctan2(y,x))    
    D = x**2+y**2
    G = self.F.diff()
    J[:,0,2:] = ((-y/D)*G.val(phi).real.T).T
    J[:,1,2:] = ((x/D)*G.val(phi).real.T).T
    return(J)
    
  
  def inv(self,tx):
    x = tx[:,0]
    y = tx[:,1]
    phi = unwrap(arctan2(y,x))
    nx = nan*ones_like(tx)
    nx[:,0] = x
    nx[:,1] = y
    nx[:,2:] = tx[:,2:] - self.F.val(phi).real
    return(nx)
    
  def __repr__(self):
    return(
      "FMap("+(
        "F="+repr(self.F)
      )+
      ")"
    )

class FMapFactory(object):
  def __init__(self,c=0.8,order=5,N=1000,ordH=3,cutoff=0.3,ordMult=10):
    self.c=c
    self.order = order
    self.N = N
    self.ordH = ordH
    self.cutoff = cutoff
    self.ordMult=ordMult
  
  def __call__(self,D):
    td = self.c*randn(self.N,D-2)
    phi = linspace(0,self.order*self.ordMult*pi,self.N)
    F = FourierSeries().fit(self.order,phi,2*pi*td.T)
    flt = linspace(1,0.0,self.order)
    idxH = where(flt>self.cutoff)[0]
    flt[idxH] = self.cutoff
    F.filter(flt)
    return(FMap(F))
  
  def __repr__(self):
    return(
      "FMapFactory("+(
        "c="+repr(self.c)+
        ",order="+repr(self.order)+
        ",N="+repr(self.N)+
        ",ordH="+repr(self.ordH)+
        ",cutoff="+repr(self.cutoff)+
        ",ordMult="+repr(self.ordMult)        
      )+
      ")"
    )    

class HMap(object):
  '''
  Constructs a Henon type map which splits a space,
  performs seperate affine transformations on
  each space, reflects one portion of this space
  through a ellipse into the other space and
  then recombines.
  '''
  def __init__(self,ga,gb,f):
    if type(ga)==str:
      self.ga = unserialiseArray(ga)
    else:
      self.ga = ga
    if type(gb)==str:
      self.gb = unserialiseArray(gb)    
    else:
      self.gb = gb
    self.f = f
    self.split = self.ga.shape[0]-1
    self.D = self.ga.shape[0]+self.gb.shape[0]-2
    
  def __call__(self,tx):
    '''
    Apply the map. If this is the first time
    this has been called it will automatically
    generate the ellipse reflect transformation.
    '''
    a = tx[:,:self.split]
    btilde = self.computebtilde(tx)
    ashift = self.f(btilde)
    atilde = applyAffine(self.ga,a) + ashift
    nx = hstack([atilde,btilde])
    return(nx)
  
  def derv(self,tx):
    '''
    Compute the Jacobian of the transformation
    '''
    btilde = self.computebtilde(tx)
    d = zeros((tx.shape[0],self.D,self.D))
    d[:,:self.split,:self.split] += self.ga[:-1,:-1]
    d[:,self.split:,self.split:] += self.gb[:-1,:-1]
    d[:,:self.split,self.split:] = einsum('ijk,jl->ikl',self.f.derv(btilde),self.gb[:-1,:-1])
    return(d)
    
  
  def computebtilde(self,tx):
    '''
    Computes the btilde portion after
    the transformation.
    '''
    b = tx[:,self.split:]
    btilde = applyAffine(self.gb,b)
    return(btilde)

  def invertbtilde(self,nx):
    '''
    Get inverse of the transformation for btilde to get back b in the 
    original space.
    '''
    btilde = nx[:,self.split:]
    b = applyAffine(invAffine(self.gb),btilde)
    return(btilde,b)

  def inv(self,nx):
    '''
    Inverse map, if the map has not been defined
    via the ellipse reflection then it will be.
    '''
    btilde,b = self.invertbtilde(nx)
    atilde = nx[:,:self.split]
    ashift = self.f(btilde)
    a = applyAffine(invAffine(self.ga),atilde-ashift)
    tx = hstack([a,b])
    return(tx)    
  
  def __repr__(self):
    return(
      "HMap("+(
        "ga="+repr(serialiseArray(self.ga))+
        ",gb="+repr(serialiseArray(self.gb))+
        ",f="+repr(self.f)
      )+
      ")"
    )

class HMapFactory(object):
  '''
  Build random H-mpas
  '''
  def __init__(self,erFac,affMag = 0.15):
    self.erFac = erFac
    self.affMag = affMag
    
  
  def __call__(self,D,split=None,C=None):
    if split is None:
      split = 1+randint(D-1)
    else:
      split = split
    ga = getRandomAffine(split,self.affMag)
    gb = getRandomAffine(D-split,self.affMag)
    f = self.erFac(D-split,split)
    return(HMap(ga,gb,f))
  
  def __repr__(self):
    return(
      "HMapFactory("+(
        "erFac="+repr(self.erFac)+
        ",affMag="+repr(self.affMag)
      )+
      ")"
    )

class HMapChain(object):
  '''
  The HMap transformation is generally
  not used in isolation but as part of a
  chain of transformations applied to a
  data set. This is that chain
  '''
  def __init__(self, H, A, F=None):
    '''
    Instantiate with the HMaps and transformations that make up the chain.
    '''
    if type(H)==list:
      self.H = H
    else:
      self.H = eval(H)
    if type(A)==list:
      self.A = A
    else:
      self.A = eval(A)
    self.A = [unserialiseArray(a) if type(a)==str else a for a in self.A]
    self.F = F

  def __call__(self,y):
    '''
    Alias for the transform method with some default behaviour
    '''
    return(self.transform(y))
  
  def transform(self,y,full=False,abortRadius=None):
    '''
    Apply our transformation, if full output
    is selected will also return the states
    after each HMap and orthogonal
    transformation.
    '''
    if full:
      inty = list()
    ry = asfarray(y)
    if self.F is not None:
      ry = self.F(ry)
      if full:
        inty.append([array(ry)])
    for n,(h,a) in enumerate(zip(self.H,self.A)):
      ry = h(ry)
      if full:
        inty.append([array(ry)])
      if abortRadius is not None and abs(ry).max()>abortRadius:
        print("Aborting, radius limit reached, possible singularity in transform. Iterations: ",n)
        ry = h.inv(ry)
        break
      ry = dot(a,ry.T).T
      if full:
        inty[-1].append(array(ry))
    if full:
      return(inty)
    return(ry)
  
  def derv(self,y):
    '''
    Calculate the Jacobian of our transformation.
    '''
    J = tile(eye(y.shape[1]),(y.shape[0],1,1))
    ry = asfarray(y)
    if self.F is not None:
      J = einsum('kij,kjl->kil',self.F.derv(ry),J)
      ry = self.F(ry)
    for h,a in zip(self.H,self.A):
      J = einsum('kij,kjl->kil',h.derv(ry),J)
      ry = h(ry)
      J = einsum('ij,kjl->kil',a,J)
      ry = dot(a,ry.T).T
    return(J)
    
    
  def dtheta(self,eta):
    '''
    Compute the derivative of the inverse transformation
    with respect to the angular co-ordinate for the original 
    system, this is the phase response curve (PRC).
    '''
    a = self.invert(eta)
    J = array([inv(iJ) for iJ in self.derv(a)])
    return(-einsum('ijk,ik->ij',J,dot(a,[[0,-1],[1,0]])))
  
  def invert(self,ry):
    '''
    This routine applies the inverse
    series of inverted HMaps to some.
    since this map is invertable this
    should fully invert the signal to
    it's original form to within machine
    precision.
    '''
    y = asfarray(ry)
    for h,a in reversed(zip(self.H,self.A)):
      y = dot(inv(a),y.T).T
      y = h.inv(y)
    if self.F is not None:
      y = self.F.inv(y)
    return(y)

  def __iter__(self):
    for h in self.H:
      yield h

  def __getitem__(self,k):
    return(self.H[k])
  
  def __repr__(self):
    '''
    Produces a string representation of the hmap chain. This is a 
    serialisation as it is just the expression needed to instantiate 
    the represented hmap chain.
    @returns: expression which instantiates this hmap chain
    @rtype: str
    '''
    return(
      "HMapChain("+(
        "H="+repr(self.H)+
        ",A=["+",".join([repr(serialiseArray(a)) for a in self.A])+"]"+
        ",F="+repr(self.F)
      )+
      ")"
    )

class HMapChainFactory(object):
  def __init__(self,hf,N=25,orthMag=0.15,C=None,nC=100,FF=None):
    self.hf = hf
    self.orthMag = orthMag
    self.N = N
    self.C = C
    if self.C=='unit circle':
      tau = linspace(0,2*pi,nC)
      self.C=array([cos(tau),sin(tau)])
    self.FF=FF
    
    
  def __call__(self,D):
    if self.C is None:
      # Random orthognal transformations between each map
      H = [self.hf(D) for i in xrange(self.N)]
      A = [getRandomOrthogonal(D,self.orthMag) for i in xrange(self.N)]
    else:
      # Random orthognal transformations between each map
      H = list()
      A = list()
      Cp = vstack([self.C,zeros((D-self.C.shape[0],self.C.shape[1]))])
      for i in xrange(self.N):
        H.append(self.hf(D,C=Cp))
        A.append(getRandomOrthogonal(D,self.orthMag))
        Cp = HMapChain(H,A)(vstack([self.C,zeros((D-self.C.shape[0],self.C.shape[1]))]))
    if self.FF is None:
      return(HMapChain(H,A))
    else:
      return(HMapChain(H,A,self.FF(D)))
  
  def __repr__(self):
    return(
      "HMapChainFactory("+(
        "hf="+repr(self.hf)+
        ",N="+repr(self.N)+
        ",orthMag="+repr(self.orthMag)+
        ",FF="+repr(self.FF)
      )+
      ")"
    )
    
def hmap2DExample(N=5,save=False):
  '''
  Apply the HMap transform to some simple
  data and plot the results.
  '''
  import pylab as pl
  from formphase.formphaseutil import simpleTestData
  # Some extra tests and examples.
  # Grab ourselves the simple 2D test data.
  y = simpleTestData()
  # Pick the number of times we  will apply
  # the transform.
  D = 2
  H = HMapChain(N,D,axeFac=0.4, refScale=0.9)
  # Apply this transformation and get
  # intermediate states
  ty = H.transform(y,True)
  # Plot the transforms of the data
  pl.figure()
  pl.plot(*y.T,label="untransformed")
  for ity,h in zip(ty,H):
    pl.plot(*ity[0].T,label="post HMap")
    # Grab the 'ellipse' of the ellipse
    # reflection and plot the extrema
    ex = h.f.getEllipse()
    eb,et = h.invertbtilde(vstack([[0,0],ex[0]]).T)
    pl.hlines(et[:,0],ity[0][:,0].min(),ity[0][:,0].max(),colors='k',label="ellipse")
    pl.legend()
    pl.figure()
    pl.plot(*ity[1].T,label="post orth")
  # We would like to know what our
  # Transformation does to the space.
  # Plot the transformation of the annulus
  # containing our data
  r = linspace(sqrt(min(sum(y**2,1))),sqrt(max(sum(y**2,1))),10)
  theta = linspace(0,2*pi,100)
  r, theta = meshgrid(r,theta)
  r = r.reshape(r.size)
  theta = theta.reshape(theta.size)
  X = r*cos(theta)
  Y = r*sin(theta)
  P = array(zip(X,Y))
  # Transform annulus
  tP = H.transform(P,True)
  # Plot the transforms
  pl.figure()
  pl.subplot(1,2,1)
  plcols = [[(ir-r.min())/(r.max()-r.min()),0,abs(it-pi)/pi] for ir,it in zip(r,theta)]
  pl.scatter(*P.T,label="untransformed",c=plcols,marker='x')
  for ity,h in zip(tP,H):
    ex = h.f.getEllipse()
    eb,et = h.invertbtilde(vstack([[0,0],ex[0]]).T)
    pl.hlines(et[:,0],ity[0][:,0].min(),ity[0][:,0].max(),colors='k',label="ellipse")
    pl.subplot(1,2,2)
    pl.scatter(*ity[0].T,label="post HMap",c=plcols,marker='x')
    # Grab the 'ellipse' of the ellipse
    # reflection and plot the extrema
    pl.figure()
    pl.subplot(1,2,1)
    pl.scatter(*ity[1].T,label="post orth",c=plcols,marker='x')

  pad = 0.1 # Vertical padding in plot
  pl.figure()
  ax0 = pl.subplot(2,3,1)
  pl.xticks([])
  pl.yticks([])
  xmin = array(tP)[:,0,:,0].min()
  xmax = array(tP)[:,0,:,0].max()
  ymin = 0.0
  ymax = 0.0
  pl.scatter(*P.T,label="untransformed",c=plcols,marker='x')
  for i,(ity,h) in enumerate(zip(tP,H)):
    ex = h.f.getEllipse()
    eb,et = h.invertbtilde(vstack([[0,0],ex[0]]).T)
    ymax = max(ymax,max(et[:,0]))
    ymin = min(ymin,min(et[:,0]))
    pl.hlines(et[:,0],xmin,xmax,colors='k',label="ellipse")
    pl.subplot(2,3,i+2,sharex=ax0, sharey=ax0)
    pl.scatter(*ity[0].T,label="post HMap",c=plcols,marker='x')
    pl.xticks([])
    pl.yticks([])
  pl.xlim([xmin-pad,xmax+pad])
  pl.ylim([ymin-pad,ymax+pad])
  pl.tight_layout(pad=1.1)

  if save:
    pass
  return(tP,H)

if __name__=="__main__":
  import doctest
  doctest.testmod()
  
  from numpy import random
  delta = 1e-11
  
  D = 8
  h = HMap(D)
  tx = random.randn(1,D)
  print(sum((array([(h(array(tx+delta*d))-h(array(tx)))/delta for d in eye(D)])[:,0,:]-h.derv(tx))**2))

  D = 4
  h = HMapChain(10,D)
  tx = random.randn(1,D)
  print(sum((array([(h(array(tx+delta*d))-h(array(tx)))/delta for d in eye(D)])[:,0,:]-h.derv(tx))**2))
