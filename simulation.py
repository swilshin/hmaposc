'''
A simulation for the formphase test cases. This objected is a wrapper around
an SDE integrator and the floquet co-ordinate oscillator and hmap. The way
this system solves the SDE is very much not safe for parrelisation,
naive parrelisation should be used instead.

@author: Simon Wilshin
@contact: swilshin@rvc.ac.uk
@date: 18 Feb 2014
'''

from formphase.formphaseutil import SDE
from formphase.hmaposc import fromFixedRPolar

from numpy import array,pi,zeros_like,dot,hstack,random,ones

from gzip import open as gzopen
from pickle import load,dump

class SimSpec(object):
  def __init__(self,noise,tau):
    self.noise=noise
    self.tau=tau

  def __repr__(self):
    return(
      "SimSpec("+(
        "noise="+repr(self.noise)+
        ",tau="+repr(self.tau)
      )+
      ")"
    )

  def pprint(self):
    return(
      self.noise.pprint()+"_"+
      self.tau.pprint()
    )

  def __eq__(self,x):
    return(
      self.noise==x.noise
        and
      self.tau==x.tau
    )

class Noise(object):
  '''
  A specification for the noise of our system
  '''
  def __init__(self,sys,init,phase):
    self.sys = sys
    self.init = init
    self.phase = phase

  def __repr__(self):
    return(
      "Noise("+(
        "sys="+repr(self.sys)+
        ",init="+repr(self.init)+
        ",phase="+repr(self.phase)
      )+
      ")"
    )

  def pprint(self):
    return(
      "Noise_"+(
        "Sys"+repr(self.sys)+
        "_Init"+repr(self.init)+
        "_Phase"+repr(self.phase)
      )
    )

  def __eq__(self,x):
    return(
      self.sys==x.sys
        and
      self.init==x.init
        and
      self.phase==x.phase
    )

class Times(object):
  def __init__(self,tMax,dt):
    self.dt = dt
    self.tMax = tMax

  def __repr__(self):
    return(
      "Times("+(
        "dt="+repr(self.dt)+
        ",tMax="+repr(self.tMax)
      )+
      ")"
    )

  def pprint(self):
    return(
      "Times_"+(
        "dt"+repr(self.dt)+
        "_tMax"+repr(self.tMax)
      )
    )

  def __eq__(self,x):
    return(
      self.dt==x.dt
        and
      self.tMax==x.tMax
    )

class HFloquetSim(object):
  '''
  Given a system and a collection of simulation parameters this
  can perform a stochastic simulation of that system
  '''
  def __init__(self,sim,spec):
    self.sim = sim
    self.spec = spec

  def _makeInitialConditions(self):
    x0 = ones(self.sim.D+1)
    x0[0] = 2*pi*random.rand()
    x0[1:self.sim.D] = self.spec.noise.init*(1.0-2*random.rand(self.sim.D-1))
    return(x0)

  def simulate(self,x0=None):
    '''
    Using the oscillator desfined by osc and Hnoise and the transformation
    defined in Hsys simulate a new trajectory.
    @ivar x0: initial conditions for simulation
    @type x0: array
    @ivar tMax: length of simulation
    @type tMax: float
    @ivar dt: time step of the simulation
    @type dt: float
    @ivar noise: system noise of the simulation
    @type noise: float
    @ivar simpleDrift: if true will use straightforward stochastic term rather
      than hmap mediated noise
    @type simpleDrift: bool
    '''
    if x0 is None:
      x0 = self._makeInitialConditions()
    sde = SDE(
      lambda x: self.sim.drift(x),
      lambda x,dW: self.sim.diffusion(x,dW,self.spec.noise),
      self.sim.D+1
    )
    tf,nxf,wf = sde(x0,0,self.spec.tau.tMax,self.spec.tau.dt)
    yf = fromFixedRPolar(nxf.T).T
    yt = self.sim.Hsys.transform(yf)
    return(tf,nxf,wf,yf,yt)

  def __repr__(self):
    return(
      "HFloquetSim("+(
        "sim="+repr(self.sim)+
        ",spec="+repr(self.spec)
      )+
      ")"
    )


class HFloquetSystem(object):
  '''
  Combines an SDE integrator and a HMap floquet system into a single class
  capable of generating new stochastic trajectories of said HMap floquet
  oscillator.
  When simulating this class is absolutely not thread safe as the class
  variables and class methods are used to run the  actual simultion. If
  you want to parellise this process do it niavely using seperate processes.
  @cvar osc: Oscillator used in the running simulation
  @type osc FloquetCoordinateOsc
  @cvar Hnoise: HMap used to non-linearly distort the noise in the
    running simulation
  @type Hnoise: HMapChain
  @cvar noise: multiplier for the noise term in the running simulation
  @type noise: float
  @ivar D: Dimension of the oscillator
  @type D: int
  @ivar Hnoise: HMap chain for non-linearly transforming the noise
  @type Hnoise: HMapChain
  @ivar Hsys: HMap transformation to apply to the floquet oscillator
    post simulation
  @type Hsys: HMapChain
  @ivar osc: floquet oscillator used in simulation
  @type osc: FloquetCoordinateOsc
  '''

  def __init__(self,Hnoise,Hsys,osc):
    '''
    Constructor, sets the dimension of the oscillator, which must be
    specified in advance, and will set the HMaps, oscillator and
    noise level if these are supplied. Otherwise the L{generateSystem}
    method should be used to create the instance variables and they will
    be set to None.
    Instance variables are described in class documentation L{HFloquetSystem}
    and this method parameters correspond and have the same name.

    '''
    # Check dimensions match
    assert all(osc.S.shape[0]-1==array([a.shape[0] for a in Hnoise.A])), \
      repr(self)+"\nDimensions of oscillator and HMaps don't match"
    assert all(osc.S.shape[0]-1==array([a.shape[0] for a in Hsys.A])), \
      repr(self)+"\nDimensions of oscillator and HMaps don't match"
    self.D = osc.S.shape[0]-1
    self.Hnoise = Hnoise
    self.Hsys = Hsys
    self.osc = osc

  def drift(self,x):
    '''
    Drift term for the L{SDE} integrator.
    @param x: position in state space
    @type x: ndarray
    @return: drift term at x
    @rtype: ndarray
    '''
    return(self.osc.drift(x))
  
  def diffusion(self,x,dW,noise):
    '''
    Diffusion term for the L{SDE} integrator with non-linear distortion of the 
    noise term through a HMap chain. Note that strictly speaking using this 
    non-linear drift term violates the assumptions of the sde integrator.
    @param x: position in state space
    @type x: ndarray
    @param dW: weiner term
    @type dW: ndarray
    @param noise: A noise specification
    @type noise: Noise
    @return: non-linear drift term at x
    @rtype: ndarray
    '''
    # Transform to noise co-ordinates and get Jacobian
    y = fromFixedRPolar(x)
    J = self.Hnoise.derv(array([y]))[0]
    # Add phase noise
    pn = zeros_like(dW[1:])
    pn[0] = noise.phase*dW[0]
    # Sum phase noise and non-linear noise from derivative of a HMap
    dw = hstack([dot(J,noise.sys*dW[1:])+pn,0])
    return(dw)

  def __repr__(self):
    '''
    Produces a string representation of the simulation. This is a
    serlisation as it is just the expression needed to instantiate the
    represented simulation
    @returns: expression which instantiates this hmap chain
    @rtype: str
    '''
    return(
      "HFloquetSystem("+(
        "Hnoise="+repr(self.Hnoise)+
        ",Hsys="+repr(self.Hsys)+
        ",osc="+repr(self.osc)
      )+
      ")"
    )

class HFloquetSystemFactory(object):
  '''
  Generate a new H-mapped floquet co-ordinate oscillator. Needs a factory
  for the noise, one to map the system through and a factory for making
  floquet oscillators. Can be called like a function to produce a new
  simulator.
  '''
  def __init__(self,Hfnoise,Hfsys,oscf):
    '''

    '''
    self.Hfnoise = Hfnoise
    self.Hfsys = Hfsys
    self.oscf = oscf

  def __call__(self,D):
    '''

    '''
    Hnoise = self.Hfnoise(D)
    Hsys = self.Hfsys(D)
    osc = self.oscf(D)
    return(HFloquetSystem(Hnoise,Hsys,osc))

  def __repr__(self):
    '''
    Produces a string representation of the simulation factory. This is a
    serlisation as it is just the expression needed to instantiate the
    represented simulation factory
    @returns: expression which instantiates this hmap chain
    @rtype: str
    '''
    return(
      "HFloquetSystemFactory("+(
        "Hfnoise="+repr(self.Hfnoise)+
        ",Hfsys="+repr(self.Hfsys)+
        ",oscf="+repr(self.oscf)
      )+
      ")"
    )

def saveSimulation(f,sim,tf,nxf,wf,yf,yt,desc=None):
  '''
  Saves the result of a simulation to the file (or file name) fn, for
  the simulation sim, with the results in tf,nxf,wf,yf and yt, and an
  optional description in desc which should include relevant metadata.
  Pickled to a tuple in the same order as the arguments of this
  function in a list.
  @param f: file or filename to save to
  @type f: str or file
  @param sim: HFloquetSimulation use to generate the trajectory
  @type sim: HFloquetSimulation
  @param tf: time course of simulation
  @type tf: ndarray
  @param nxf: co-ordinates of oscillator in radius-angle form with
    radius shifted by unity
  @type nxf: ndarray
  @param wf: weiner terms used in simulation
  @type wf: ndarray
  @param yf: polar form of nxf
  @type yf: ndarray
  @param yt: yf after being put through the Hsys transformation. This
    is the end result of the simulation
  @type yt: ndarray
  @param desc: description of this trial, put metadata here.
  @type desc: str
  @return: None
  @rtype: None
  '''
  if type(f)==str:
    f = gzopen(f,'wb')
  dump(['''A simulation for the form-phase project. The first entry here is
this header, the second is a string serialisation of the simulation,
the third is the time course of the simulation, the forth is the
simulation in floquet co-ordinates, the fifth is the stochastic term
added to the simulation, the sixth is the floquet system transformed to
polar co-ordinates, the seventh is the final simulation after
transformation through the H-map, the final entry is a string description
with misc metadata for this simulation including the noise level.''',
    repr(sim),tf,nxf,wf,yf,yt,desc], f,-1)
  f.close()

def loadSimulation(f):
  '''
  Inverse operation of L{saveSimulation}. Given a file or filename
  this returns the parameters passed to L{saveSimulation} saved to
  that file in the same order.
  @param f: file or filename to load
  @type f: file or str
  @return: list or simulation results, see L{saveSimulation} for format.
  @rtype: list
  '''
  if type(f)==str:
    f = gzopen(f,'rb')
  header,sim,tf,nxf,wf,yf,yt,desc = load(f)
  f.close()
  return(sim,tf,nxf,wf,yf,yt,desc)

if __name__=="__main__":
  '''
  Example simulation code. Generates a 2D osccillator, simulates a run with
  fixed initial conditions and plots the result.
  plots the
  '''
