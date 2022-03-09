'''
Libraries for creating a non-trivial oscillator using a henon map like 
transformation.

@author: Simon Wilshin
@contact: swilshin@rvc.ac.uk
@date: March 2013
'''

from .hmap import HMap,HMapChain,HMapFactory,HMapChainFactory,FMap,FMapFactory
from .sdeosc import (FloquetCoordinateOsc,FloquetCoordinateSim,
  FloquetCoordinateOscFactory,fromFixedRPolar,toFixedRPolar)
from .simulation import (HFloquetSystem,HFloquetSystemFactory,
  saveSimulation,loadSimulation,SimSpec,Noise,Times,HFloquetSim)
from . import plots
from .data import DataSet,DataSetFactory,Trials
from .datatransform import (PCASpec,ZScoreSpec,DataTransformStack,
  PlanarZScoreSpec,QuickScale)