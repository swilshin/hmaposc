'''
An oscillator which orbits about the origin
with a variable number of dimensions, variable
noise, and the ability to control how rapidly
we return to the orbit.

CO-ORDINATE SYSTEM
==================
This set of objects uses the following affine cylindrical polar co-ordinate
system.

The first co-ordinate is the angle in the plane. The second co-ordinate is the
radius minus unity. The remaining co-ordinates are all perpendicular to the
plane, with the expection of the last co-ordinate, which is always unity as it
is the affine co-ordinate.

The transition matrix, T is always D+1 by D+1 dimensional.

As such the transition matrix, T, always has zeros in the first row except for
the last entry which is always minus unity. The last row of T is also always
zero since the affine co-ordinate is never adjusted. The first column is always
zero because no phase dependence for the evolution of the co-ordinates exists
for this system. The last column is always zero as there is no constant term
for the shifted radius or the perpendicular terms. This leaves the D-1 by D-1
submatrix in the middle of T. This is a symmetric matrix, when diagonalised
it gives the floquet modes and their time constants.

@author: Simon Wilshin
@contact: swilshin@rvc.ac.uk
@date: Feb 2013
'''

from formphase.formphaseutil import (getRandomPositiveDef,serialiseArray,
  unserialiseArray)
from numpy import (diag,linspace,hstack,vstack,zeros,sqrt,cos,sin,dot,
  arctan2,ones_like)
from numpy.random import randn,rand
from numpy.linalg import qr

def fromFixedRPolar(x):
  '''
  Convert x from polar co-ordinates to cartesian co-ordinates, shifting the
  radial co-ordinate by unity (the systems simulated in the
  L{FloquetCoordinateOsc} have a co-ordinate system with radius 1 at the
  origin and the radius running from -1.
  @param x: co-ordinate to convert, the first index runs over the co-ordinates
  with the first being the angle, the second the shifted radius, the
  remainder are perpendicular.
  @type x: ndarray, n, n x d or n x d1 x ... x dq
  @return: array in shifted cartesian co-ordinates
  @rtype: ndarray
  '''
  if x.ndim==1:
    if x.shape[0]>3:
      y = x[:-1]
      y = hstack([
                  (y[1]+1)*cos(y[0]),
                  (y[1]+1)*sin(y[0]),
                  y[2:]
                ])
    else:
      y = x[:-1]
      y = hstack([
                  (y[1]+1)*cos(y[0]),
                  (y[1]+1)*sin(y[0]),
                ])
  else:
    if x.shape[0]>3:
      y = x[:-1]
      y = vstack([
                  (y[1]+1)*cos(y[0]),
                  (y[1]+1)*sin(y[0]),
                  y[2:]
                ])
    else:
      y = x[:-1]
      y = vstack([
                  (y[1]+1)*cos(y[0]),
                  (y[1]+1)*sin(y[0]),
                ])
  return(y)

def toFixedRPolar(y):
  '''
  Inverse transform of L{fromFixedRPolar}.
  @param x: co-ordinate to convert, the first index runs over the co-ordinates
  with the first being the angle, the second the shifted radius, the
  remainder are perpendicular.
  @type x: ndarray, n, n x d or n x d1 x ... x dq
  @return: array in shifted polar co-ordinates
  @rtype: ndarray
  '''
  if y.ndim==1:
    if y.shape[0]>2:
      x = hstack([
            arctan2(y[1],y[0]),
            sqrt(y[0]**2+y[1]**2)-1.0,
            y[2:],
            ones_like(y[0])
          ])
    else:
      x = hstack([
            arctan2(y[1],y[0]),
            sqrt(y[0]**2+y[1]**2)-1.0,
            ones_like(y[0])
          ])
  else:
    if y.shape[0]>2:
      x = vstack([
            arctan2(y[1],y[0]),
            sqrt(y[0]**2+y[1]**2)-1.0,
            y[2:],
            ones_like(y[0])
          ])
    else:
      x = vstack([
            arctan2(y[1],y[0]),
            sqrt(y[0]**2+y[1]**2)-1.0,
            ones_like(y[0])
          ])
  return(x)


class FloquetCoordinateOsc(object):
  '''
  An oscillator which orbits about the origin
  with a variable number of dimensions, variable
  noise, and the ability to control how rapidly
  we return to the orbit.
  One can also initialise the transition
  and noise covariance matricies seperately
  by passing T and S directly, this will override
  all other settings.
  '''
  def __init__(self,T,S):
    '''
    Constructor, can be invoked directly if specific transiton and noise
    covariance matricies are desired, otherwise the
    L{FloquetCoordinateOscFactory} can be used to construct a system with
    consistent transition and noise covariance matricies.
    @param T: The transition matrix for the SDE in time.The first co-ordinate
    is the phase, the second is a radial component minus unity and the
    remaining terms are orthogonal directions.
    @type T: ndarray or L{unserialiseArray}, (n+1 x n+1)
    @param S: The covariance matrix for the noise. It is zero in the last row
    and column as there is no noise in the affine co-ordinate, otherwise it can
    contain any entries.
    @type S: ndarray or L{unserialiseArray}, (n+1 x n+1)
    '''
    if type(T)==str:
      self.T = unserialiseArray(T)
    else:
      self.T = T
    if type(S)==str:
      self.S = unserialiseArray(S)
    else:
      self.S = S

  def drift(self,x):
    '''
    This is the drift term for the SDE, given positions in the state space, x
    (which should be a D or a D x N array, this generates the time derivative
    from the deterministic part of the SDE.
    @param x: point(s) to evaluate the drift term at
    @type x: ndarray, n, n x d or n x d1 x ... x dq
    @param dW: sample from a Wiener process
    @type dW: ndarray, n, n x d or n x d1 x ... x dq
    @return: drift term
    @rtype: ndarray
    '''
    return(-dot(self.T,x))

  def diffusion(self,x,dW):
    '''
    This is the diffusion term for the SDE. For compatibility with the SDE
    integrator this takes are arguments both positions, x, and samples from a
    Wiener process, dW. However, the sample positions are unused.

    Any future modifications will assume the positions, x, take the form of a
    D or a D x N array. The Weiner term should be in the form of a D or a D x N
    array.
    @param x: point(s) to evaluate the diffusion term at
    @type x: ndarray, n, n x d or n x d1 x ... x dq
    @param dW: sample from a Wiener process
    @type dW: ndarray, n, n x d or n x d1 x ... x dq
    @return: diffusion term
    @rtype: ndarray
    '''
    raise RuntimeError('Never used')
    return(dot(self.S.T,dW))

  def __repr__(self):
    '''
    Produces a string representation of the floquet co-ordinate oscillator.
    This is a serialisation as it is just the expression needed to
    instantiate the represented oscillator.
    @returns: expression which instantiates this hmap chain
    @rtype: string
    '''
    return(
      "FloquetCoordinateOsc("+(
        "T="+repr(serialiseArray(self.T))+
        ",S="+repr(serialiseArray(self.S))
      )+
      ")"
    )

class FloquetCoordinateOscFactory(object):
  def __init__(self,kMax=0.3,sMax=0.01):
    '''
    Standardises the method of creating the co-ordinate oscillators.
    @param kMax: Maximum eigenvalues of the evolution
    equation, larger eigenvalues means the system
    recovers to the orbit faster.
    @type kMax: float
    @param sMax: Scales the noise, the covariance matrix
    is multiplied by the square root of this value.
    @type sMax: float
    '''
    self.kMax = kMax
    self.sMax = sMax

  def __call__(self,D):
    '''
    Create a random evolution matrix T, and a
    noise covariance matrix S
    D: The dimension of the system to generate
    '''
    # Define a random transition matrix T.
    M = getRandomPositiveDef(D-1,self.kMax)
    cT = zeros((D-1)) # randn(D-1)

    T = vstack([cT,M,zeros((D-1))])
    T = hstack([zeros((D+1,1)),T,zeros((D+1,1))])
    T[0,-1]=-1

    # Generate a random noise covariance, S.
    # Covariance matrix of noise is dot(S.T,S)
    # That is cov(dot(S.T,randn(D+1,1000000)))-dot(S.T,S) is roughly zero
    Q = qr(randn(D,D))[0]
    S = dot(sqrt(self.sMax*diag(rand(D))),Q.T)
    S = hstack([S,zeros((D,1))])
    S = vstack([S,zeros((1,D+1))])

    return(FloquetCoordinateOsc(T,S))

  def __repr__(self):
    return(
      "FloquetCoordinateOscFactory("+(
        'kMax='+repr(self.kMax)+
        ',sMax='+repr(self.sMax)
      )+
      ")"
    )

class FloquetCoordinateSim(object):
  def __init__(self,fco):
    self.fco = fco
    self.D = self.fco.S.shape[0]-1

  def solve(self,x0=None,tauMax=1000,N=100000,dt=0.01):
    '''
    Get a solution to this floquet system
    starting at x0. Currently uses the
    SDE version of the Euler method.
    tauMax: The total time in seconds to
    run the simulation for.
    N: The total number of time points to
    simulate
    dt: the time step

    '''
    dt=float(tauMax)/N
    t = linspace(0,tauMax,N)

    # If we don't get a start location pick a reasonable random one.
    if x0 == None:
      x0 = hstack([[0],dot(self.fco.S.T[1:-1,1:-1],randn(self.D-1)),[1]])

    # Initialise
    x = zeros((N,self.D+1))
    x[0] = x0

    # Euler equivalent SDE method
    for i in range(N-1):
      x[i+1] = (
        x[i] +
        self.fco.drift(x[i])*dt +
        self.fco.diffusion(x[i],sqrt(dt)*randn(self.D+1))
      )
    x[:,1] += 1.0
    return(t,x)

  def __repr__(self):
    return(
      "FloquetCoordinateSim("+(
        "fco="+repr(self.fco)
      )+
      ")"
    )

if __name__=="__main__":
  F = FloquetCoordinateOscFactory()
  osc = F()
  S = FloquetCoordinateSim(osc)
  t,x = S.solve()
