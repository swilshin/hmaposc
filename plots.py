
from formphase.formphaseutil import sampleSCrossSD,applyAffine

from numpy import (linspace,meshgrid,vstack,ones,pi,ones_like,zeros_like,
  ceil,zeros,array,max,min,sum,where,logical_and,sqrt,dot,hstack,abs,diff,
  cumsum,cos,sin,mean,angle,exp)
from pylab import (figure,plot,gca,scatter,xlabel,ylabel,subplot,gcf,Circle, 
  hlines)
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial.distance import cdist

from itertools import count

#=============================================================================
# PLOT FUNCTIONS
#=============================================================================
cols = {
  'phaser':'k',
  'form':'#106AA4', #Teal
  'event':'#43BF3C', # Green
  'true':'#FF7F00' # Orange
}
markers = {
  'phaser':'s',
  'form':'D',
  'event':'h',
  'true':'+'
}

cdict = {'red': ((0.0, 0.0, 0.0),
                (0.5, 1.0, 1.0),
                (1.0, 0.0, 0.0)),
        'green': ((0.0, 0.0, 0.0),
                  (0.25, 0.5, 0.5),
                  (0.5, 0.0, 0.0),
                  (0.75, 0.0, 0.0),
                  (1.0, 0.0, 0.0)),
        'blue': ((0.0, 1.0, 1.0),
                (0.5, 0.0, 0.0),
                (1.0, 1.0, 1.0))}    
cbfcm = LinearSegmentedColormap('color blind friendly cyclic map',cdict,256)


def isoplot(th,ax,Niso,yd,th0,delta,alpha,sscale,Ra,f,shiftedls,idxMaxRec=1200,cmap=cbfcm):     
  '''
  Make a scatter plot of the isochrones with th phases, ax the axis to 
  plot on, Niso the number of isochrones, yd the positions of the 
  system at phases in th, th0 the ground truth phases, delta the width 
  of the phases in radians, alpha the transparency of the scatter, 
  sscale the area of the scattered points, Ra transformation to apply 
  to yd (callable), f is a function which calculates the color of points 
  based on distance from the limit cycle, shiftedls is the position of the 
  limit cycle, idxMaxRec the maximum number of points to use for an 
  isochrone and cmap the colormap to use for the points.
  '''
  thi = mean(angle(exp(1j*(th0-th))))
  c,p = array([]),array([[]])
  p.shape = (2,0)
  for thMid in linspace(0,2*pi,Niso):
    for theta,iyd in zip(th,yd):
      theta = theta[:idxMaxRec]
      idx = where(abs((theta-thi-thMid+pi)%(2*pi)-pi)<delta)[0]
      p=hstack([p,iyd[idx].T])
      c=hstack([c,f(cdist(shiftedls,iyd[idx]).min(0))])
  ax.scatter(*Ra(p),c=cmap(c),alpha=alpha,s=sscale**2,lw=0,label='phaser',zorder=2)      
  

def plotPhases(phR,cols=cols,ax=None,**args):
  '''
  Plot the phases in the dictionary phR against time using the colours 
  specified in cols. Just a wrapper for the loop and axis labels and so on. 
  Allows plot parameters to be specified via additional arguments, anything 
  which works in plot will work here, defaults to an alpha of 0.1
  @param phR: dictionary of array of phases, key is the method used to 
    estimate the phase and is used as a label
  @type phR: dict
  @param cols: dictionary of colours for the different methods in phR
  @type cols: dict
  @param ax: axis to make the plot on
  @type ax: Axis
  @return: None
  @rtype: None
  '''
  # Plot parameters
  if ax is None:
    ax=gca()
  if not args.has_key('alpha'):
    args['alpha']=0.1
  
  # Plot
  for k in phR:
    for i,p in enumerate(phR[k]):
      if i==0:
        ax.plot(p,c=cols[k],label=k,**args)
      else:
        ax.plot(p,c=cols[k],**args)
  
  # Labels and legend
  ax.legend()
  ax.set_title("phaser estimation performance")
  ax.set_xlabel("time (arb)")
  ax.set_ylabel("phase error (rad)")

def errorByPhase(ph,phR,ax=None,cols=cols,markers=markers,**args):
  '''
  Scatterplot of errors in phase (in dictionary phR) against ground truth 
  phase given by the 'true' entry in the ph dictionary. Plots on axis ax 
  if provided, otherwise plots on current axis. Accepts matplotlib args 
  compatible with the scatter routine, default alpha is 0.05. Uses the 
  colours and markers in cols and markers, has defaults for the standard 
  formphase methods.
  @param ph: dictionary of phases, must have an entry 'true' which is 
    used as ground truth phase
  @type ph: dict
  @param phR: dictionary of phase errors, keys are phase extraction method 
    and are used as labels
  @type phR: dict
  @param ax: axis to make plot on
  @type ax: Axis
  @param cols: colors for the plot
  @type cols: dict
  @param markers: markers for the plot
  @type markers: dict
  @return: None
  @rtype: None
  '''
  # Plot parameters
  if ax is None:
    ax=gca()
  if not args.has_key('alpha'):
    args['alpha']=0.05
  
  # Get ground truth phase and plot
  theta = hstack(ph['true'])
  for k in phR:
    p = hstack(phR[k])
    ax.scatter(theta%(2*pi),p,marker=markers[k],c=cols[k],label=k,**args)
  
  # Labels and legend
  ax.legend()
  ax.set_title('phase error by ground truth phase')
  ax.set_xlabel('ground truth phase (rad)')
  ax.set_ylabel('phase error (rad)')

def errorHistograms(phR,errN=0.02,Nstep=1000,ax=None,cols=cols,**args):
  '''
  Produce a histogram of phase errors for each method in the dictionary 
  phR indexed by method name. errN specifies the error range to plot, while 
  Nstep specifies the number of bins in the histogram, will be plotted on ax 
  or the current axis is none if specified using the colours in cols, uses 
  the standard formphase defaults if this is not specified.
  Will not plot a histogram for any entry lablled 'true', since this label 
  is reserved for the ground truth phase whose histogram is always a 
  delta function.
  Accepts additional plot parameters for his via args, defaults to an alpha 
  of 0.5 if not specified. Returns a dictionary where entries correspond to 
  the phase methods labelled by phR keys in the same format as returned by 
  the hist method of matplotlib.
  @param phR: phase errors
  @type phR: dictionary
  @param errN: plots in the range -errN to errN
  @type errN: float
  @param Nstep: number of bins in the histogram
  @type Nstep: int
  @param ax: axis to plot on
  @type ax: Axis
  @param cols: colors for the plot
  @type cols: dict
  @return: the histogram specification in the same format as returned by 
    matplotlib
  @rtype: list
  '''
  # Parameters
  if ax is None:
    ax=gca()
  if not args.has_key('alpha'):
    args['alpha']=0.5
  
  # Plot
  errLims = linspace(-errN,errN,Nstep)
  h = dict()
  h0 = None
  for k in phR:
    if k!='true':
      p = hstack(phR[k])
      if h0 is None:
        h0 = ax.hist(p,errLims,color=cols[k],label=k,**args)
        h[k] = h0
      else:
        h[k] = ax.hist(p,h0[1],color=cols[k],label=k,**args)
  
  # Labels and legend
  ax.legend()
  ax.set_title("phase error histogram")
  ax.set_xlabel("phase error (rad)")
  ax.set_ylabel("frequency")
  
  # Return dictionary of historgram bins and frequencies
  return(h)

def errorPartialProbPlot(i,j,k,h,phR,yd=None,fig=None,semilog=False,alphaH=0.05,
                         padding=0.1,Nticks=5,cols=cols,markers=markers,xl=None,
                         yl=None, noLabel=False,retAx=False,ms=3, **args):
  '''
  Creates a partial probability plot for the phase errors with the methods in 
  h. Plot is created as a subplot at matplotlib index i,j,k on figure fig, 
  the current figure is used by default if no figure is specified. 
  If yd is specified then trajectories are plotted in the background on a 
  hidden axis.
  The y axis can be made logarithmic by setting semilog to True (false by 
  default), the trajectory alpha value is set by alphaH, the amount of 
  padding around these trajectories is specified by padding, the number 
  of xticks is specified via Nticks, colours and markers specified by 
  cols and markers (keys correspond to h) and uses the standard formphase 
  defaults if these are not specified.
  @param i: matplotlib subplot specification for number of rows
  @type i: int
  @param j: matplotlib subplot specification for number of columns
  @type j: int
  @param k: matplotlib subplot index
  @type k: int
  @param yd: trajectories from which phase estimates are produced. Will 
    plot these in the background in the x-y plane.
  @param fig: figure to plot on
  @type fig: Figure
  @param semilog: if true the y-axis will be log scaled
  @type semilog: bool
  @param alphaH: alpha for the background trajectory plot
  @type alphaH: float
  @param padding: amount of whitespace around the background trajectory 
    plots
  @type padding: float
  @param Nticks: number of xticks
  @type Nticks: int
  @param cols: colors for the plot
  @type cols: dict
  @param markers: markers for the plot
  @type markers: dict
  @return: tuple containing the partial probabilities and the index of the 
    bins these correspond to.
  @rtype tuple
  '''
  # Figure parameters
  if fig is None:
    fig=gcf()
  
  # Create figure
  ax = fig.add_subplot(i,j,k,sharex=errorPartialProbPlot.shareax,sharey=errorPartialProbPlot.shareax)
  if semilog:
    ax.set_yscale('log')
  hidax = fig.add_axes(ax.get_position())
    
  # Compute and plot partial probabilities
  z=3 # z-order of this plot entry
  dP = dict()
  idxPlt = dict()
  for k in h:
    Nstep=len(h[k][0])
    dP[k] = (h[k][0]/diff(h[k][1]))/hstack(phR[k]).size
    idxInt = (((cumsum(dP[k])/sum(dP[k]))*Nstep)%1)
    idxPlt[k] = where(logical_and(idxInt[:-1]<0.5,idxInt[1:]>0.5))[0]
    ax.plot(
      h[k][1][1:][idxPlt[k]],dP[k][idxPlt[k]],
      color=cols[k],marker=markers[k],label=k,zorder=z,
      ms=ms
    )
    z+=1
    ax.set_xlim([h[k][1][0],h[k][1][-1]])
    ax.set_xticks(linspace(h[k][1][0],h[k][1][-1],Nticks))
    ax.set_zorder(2)

  # Fix visibility so ax is on top of hidax but the background does not 
  # cover it
  ax.patch.set_facecolor('none')
  
  # Labels
  if not noLabel:
    ax.set_xlabel("phase error $\\Delta \\phi$ (rad)")
    ax.set_ylabel("$dPr\\left(\\Delta \\phi\\right)/d\\phi$ (arb)")
  if xl is not None:
    ax.set_xlim(xl)
    ax.set_xticks(linspace(xl[0],xl[1],Nticks))
  if yl is not None:
    ax.set_ylim(yl)
  
  
  # If we have trajectories of the oscillator then plot them in the 
  # background
  if yd is not None:
    for iyd in yd:
      hidax.plot(iyd[:,0],iyd[:,1],c='b',alpha=alphaH,zorder=1)
    hidax.set_xlim(
      min([iyd[:,0].min()-padding for iyd in yd]),
      max([iyd[:,0].max()+padding for iyd in yd])
    )
    hidax.set_ylim(
      min([iyd[:,1].min()-padding for iyd in yd]),
      max([iyd[:,1].max()+padding for iyd in yd])
    )  
  
  hidax.set_axis_off()
  if retAx:
    return(dP,idxPlt,ax)
  return(dP,idxPlt)
errorPartialProbPlot.shareax=None


def subContourPlot(ax,Y0,Z,yd=None,**args):
  ax.contour(Y0[0],Y0[1],(Z+pi)%(2*pi)-pi,args['V'],extend='neither',**args)
  if yd is not None:
    ax.scatter(*yd,color='b',marker='.',lw=0,alpha=args["alphascat"])

def formatAx(ax,zmax,zoom,ar=1.0):
  ax.patch.set_facecolor('none')
  #ax.axis('equal')
  ax.set_xlim([-zmax,zmax])
  ax.set_ylim([-zmax*ar,zmax*ar])
  if zoom is not None:
    ax.set_xlim([zoom[0],zoom[1]])
    ax.set_ylim([zoom[2],zoom[3]])


class Ident(object):
  def __call__(x):
    return(x)
  def inv(x):
    return(x)

def termContributionsPlot(
  form,rmax,Nsamp=100,yd=None,ccent=(0,0),zmax=None,zoom=None,ar=1.0,
  Ra=Ident(),fig=None,cumulativeOnly=False,singleAxis=False,makeColor=False,
  gtTrans=None,
  **args
):
  '''
  Makes a contour plot of the potential for each term, and a the sum of these 
  contributions which constitute the estimated isochrones. Produces a zero 
  centered grid of half width rmax with Nsamp samples. If a number of 
  contours is not specified (for example via 
  termContributionsPlot(form,rmax,N=10) for 10 contours) the 50 are used. 
  Accepts any matplotlib parameters that can be used with contour
  @param form: form phase form chain to recurse on and plot term by term 
    contributions of
  @type form: FormChain
  @param rmax: contour plot is produced on a grid whose extrema have this 
    value
  @type rmax: float    
  @param rmin: If not None, then the radius of the white circle to plot in 
    the middle to overlay the contours
  @type rmin: float
  @param Nsamp: the grid will have Nsamp*Nsamp evenly spaced entries
  @type Nsamp: int
  @param ccent: center of the circle to cut out in the middle of radius rmin
  @type ccent: tuple
  @param zmax: axis scales (equal, so this is the min and max x and y limits)
  @type zmax: float
  @param zoom: a region to zoom in on
  @type zoom: list
  @param ar: aspect ratio of plot
  @type ar: float
  @return: None
  @rtype: None
  '''
  ccol = ['r','#008010','#001060']
  
  if zmax is None:
    zmax = rmax
  # Parameters
  if not args.has_key("alpha"):
    args["alpha"]=0.5
  if not args.has_key("alphascat"):
    args["alphascat"]=0.3
  
  # Generate co-ordiantes on which to evaluate the potential
  X = linspace(-rmax,rmax,Nsamp)
  Y = linspace(-rmax*ar,rmax*ar,Nsamp)
  Y0 = array(meshgrid(X,Y))
  Y = Y0.reshape(2,Y0.size/2)
  U = zeros(Y.shape[1])
  
  # Get forms by following chain up
  frm = form
  forms = [form]
  while frm.par is not None:
    frm = frm.par
    forms.append(frm)
  
  # Create figure
  if fig is None:
    fig = figure(figsize=(12,6))
  #labs = ["$\\theta$ term","affine term","fourier term","rbf term"]
  labs = ["$\\theta$ term","fourier term","rbf term"]
  N = len(labs)
  
  # Now make plots of each of the terms and their effect when added
  ax0 = None
  for j,form in enumerate(reversed(forms)):
    Z = form.getV(Ra.inv(Y))
    U = Z-U
    
    if not cumulativeOnly:
      if not singleAxis:
        if ax0 is None:
          ax = fig.add_subplot(2,N,j+1)
          ax0 = ax
        else:
          ax = fig.add_subplot(2,N,j+1,sharex=ax0,sharey=ax0)
      else:
        if ax0 is None:
          ax = fig.add_subplot(2,1,1)
          ax0 = ax
      
      subContourPlot(ax,Y0,U.reshape(Y0[0].shape),Ra(yd[:2]),colors=ccol[j] if makeColor else 'k',linestyles='solid',linewidths=3,**args)
      formatAx(ax,zmax,zoom,ar)
      ax.set_xticklabels([])
      ax.set_yticklabels([])
      if j==0:
        ax.set_ylabel("single")
    
    if not singleAxis:    
      if ax0 is None:
        ax = fig.add_subplot(2-cumulativeOnly,N,j+1)
        ax0 = ax
      ax = fig.add_subplot(2-cumulativeOnly,N,N-(N*cumulativeOnly)+j+1,sharex=ax0,sharey=ax0)
    else:
      if ax0 is None:
        ax = fig.add_subplot(2-cumulativeOnly,1,2-cumulativeOnly)
        ax0 = ax
      ax = fig.add_subplot(2-cumulativeOnly,1,2-cumulativeOnly,sharex=ax0,sharey=ax0)
      
    
    xlabel(labs[j])
    subContourPlot(ax,Y0,Z.reshape(Y0[0].shape),Ra(yd[:2]),colors=ccol[j] if makeColor else 'k',linestyles='solid',linewidths=3,**args)
    formatAx(ax,zmax,zoom,ar)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    if j==0 and not cumulativeOnly:
      ax.set_ylabel("cumulative")
    U = Z
  
  if singleAxis and cumulativeOnly and gtTrans is not None:
    U = gtTrans.H.sim.osc.fromPolar(gtTrans.H.sim.Hsys.invert(gtTrans.trials.dtran.inv(Ra.inv(Y)).T).T)[0]
    subContourPlot(ax,Y0,U.reshape(Y0[0].shape),Ra(yd[:2]),colors='#FF10EE',linestyles='solid',linewidths=3,**args)
  

def scatterIsochrones(y,psi,theta,eth=0.01,ax=None,**args):
  '''
  Plot the implied 2D or 3D isochrones for the phases psi using the locations 
  in y at the angles in theta. Makes a scatter plot of all points in y whose 
  phase in psi is within eth of the angles in theta. Will plot on the current 
  axis if ax is not specified. Accepts additional plot arguments, anything 
  compatible with the matplotlib scatter routines. Will default to an alpha 
  of 0.3 if no alpha value is specified, and a line width of zero if none 
  is specified.
  @param y: xy locations of trajectories
  @type y: ndarray
  @param psi: estimated phases to scatter
  @type psi: ndarray
  @param theta: phases of isochrones to plot
  @type theta: list
  @param eth: tolerance for isochrones, points within eth of entries in 
    theta are plotted
  @type eth: float
  @param ax: axis to plot on
  @type ax: Axis
  @return: None
  @rtype: None
  '''
  # Plot parameters
  if ax is None:
    ax=gca()
  if not args.has_key("alpha"):
    args["alpha"]=0.3
  if not args.has_key("lw"):
    args["lw"]=0   
  
  # Find indicies of points close to requested phases
  idxP = None
  for itheta in theta:
    if idxP is None:
      idxP = where(logical_and(psi<itheta+eth,psi>itheta-eth))[0]
    else:
      idxP = hstack([idxP,where(logical_and(psi<itheta+eth,psi>itheta-eth))[0]])
  
  # Scatter
  if y.shape[1]==2:
    ax.scatter(*y[idxP].T,**args)
  else:
    ax.scatter3D(*y[idxP].T,**args)
  
  return(idxP)

def plotProjection(yt,pair,ax=None,**args):
  '''
  Plot the two dimensional projection of co-ordinates c[0] and c1[] of the 
  trajectories in yt on axis ax, if no axis is specified the current one is 
  used. Additional arguments are passed to the axis plot function.
  '''
  if ax is None:
    ax=gca()
  ax.plot(yt[:,pair[0]],yt[:,pair[1]],**args)

def plotCyclicOrdProj(yt,**args):
  '''
  Creates a figure which plots in two dimensions cyclic pairs of co-ordinates 
  of some trajectories specified in yt.
  @param yt: trajectories to plot
  @type yt: list
  @return: None
  @rtype: None
  '''
  # Create figure and work out how many subplots are needed (N)
  fig = figure()
  N = int(ceil(sqrt(yt.shape[1]-1)))
  
  # Create cyclic ordering of pairs of co-ordiantes
  pairs = zip(range(yt.shape[1])[:-1],range(yt.shape[1])[1:])
  
  # Plot trajectories for each pair
  ax0 = None
  for i,p in enumerate(pairs):
    if ax0 is None:
      ax=fig.add_subplot(N,N,i+1)
      ax0 = ax
    else:
      ax=fig.add_subplot(N,N,i+1,sharex=ax0,sharey=ax0)
    plotProjection(yt,p,ax,**args)
    ax.set_xticks([])
    ax.set_yticks([])

def plot2DIsochrones(s,Niso=20,Nrad=100,ax=None,rmax=0.3,**args):
  '''
  
  '''
  if ax is None:
    ax=gca()
  osc = s.osc
  H = s.Htrans
  # Generate co-ordinates of isochrones
  theta = linspace(0,2*pi,Niso,endpoint=False)
  r = linspace(-rmax,rmax,Nrad)
  T,R = meshgrid(theta,r)
  for i in xrange(T.shape[1]):
    iso = vstack([T[:,i],R[:,i],ones(R.shape[0])]).T
    iso = osc.toPolar(iso.T).T
    iso = H.transform(iso)
    ax.plot(*iso.T,**args)
  return(theta)

def plot3DIsochrones(s,Niso=5,Nrad=25,Nx=25,ax=None,rmax=0.3,**args):
  if ax is None:
    ax=gca()
  osc = s.osc
  H = s.Htrans
  # Generate co-ordinates of isochrones
  if type(Niso) is list:
    theta=Niso
  else:
    theta = linspace(0,2*pi,Niso,endpoint=False)
  r = linspace(-rmax,rmax,Nrad)
  x = linspace(-rmax,rmax,Nx)
  R,X = meshgrid(r,x)
  s0 = R.shape
  R=R.reshape(R.size)
  X=X.reshape(X.size)
  for phi in theta:
    iso = vstack([phi*ones_like(R),R,X,ones_like(R)]).T
    iso = osc.toPolar(iso.T).T
    iso = H.transform(iso)
    ax.plot_surface(iso[:,0].reshape(s0),iso[:,1].reshape(s0),iso[:,2].reshape(s0),**args)
  return(theta)

def plotLimitCycle(s,theta0=0,theta1=2*pi,Ntheta=100,ax=None,**args):
  if ax is None:
    ax=gca()
  osc = s.osc
  H = s.Htrans
  # Generate co-ordinates of isochrones
  theta = linspace(theta0,theta1,Ntheta)
  lcx = vstack([theta,zeros((H.D-1,Ntheta)),ones_like(theta)]).T
  lcx = osc.toPolar(lcx.T).T
  lcx = H.transform(lcx)
  ax.plot(*lcx.T,**args)

def plot2DOscillatorEvolution(s,Niso=20,Nrad=100,Ntheta=100):
  '''
  Specifically designed for making a figure for the paper, this method 
  expects a 2D oscillator made with 10 applications of the Hmap and shows 
  how the oscillator changes are those transformations are applied.
  '''
  if s.D != 2:
    Exception("This plot expects a 2D oscillator")
  if s.Htrans.N != 10:
    Exception("This plot expects a oscillator distorted with 10 applications of the HMap")
  osc = s.osc
  Htrans= s.Htrans    
  theta = linspace(0,2*pi,Ntheta)
  lcx = osc.toPolar(vstack([theta,zeros((Htrans.D-1,Ntheta)),ones_like(theta)])).T
  theta = linspace(0,2*pi,Niso,endpoint=False)
  r = linspace(-0.9,0.9,Nrad)
  T,R = meshgrid(theta,r)
  isox = [osc.toPolar(vstack([T[:,i],R[:,i],ones(R.shape[0])])).T for i in xrange(T.shape[1])]
  fig = figure()
  ax0=fig.add_subplot(2,6,1)
  ax0.plot(*lcx.T,color='k',alpha=0.8,lw=2)
  for x in isox:
    ax0.plot(*x.T,color='b',alpha=0.8,lw=2)
  ax0.set_xticks([])
  ax0.set_yticks([])
  for i,(h,A) in enumerate(zip(Htrans.H,Htrans.A)):
    lcx = dot(A,h(lcx).T).T
    ax=fig.add_subplot(2,6,i+2,sharex=ax0,sharey=ax0)
    ax.plot(*lcx.T[:2])  
    for i,x in enumerate(isox):
      isox[i] = dot(A,h(x).T).T
      ax.plot(*isox[i].T,color='b',alpha=0.8,lw=2)
    ax.set_xticks([])
    ax.set_yticks([])

def serialHMapDistortion2D(s,alpha=0.1,Nsamp=1000):
  D = s.D
  Htrans = s.Htrans
  x = sampleSCrossSD(r0=1.0,r1=0.3,D=D-1,N=Nsamp)  
  tmpx = array(x)
  fig = figure()
  N = Htrans.N
  isqrtN = int(ceil(sqrt(N+1)))
  ax=fig.add_subplot(isqrtN,isqrtN,1)
  ax.scatter(*x.T[:2])
  for i,(h,A) in enumerate(zip(Htrans.H,Htrans.A)):
    tmpx = dot(A,h(tmpx).T).T
    ax=fig.add_subplot(isqrtN,isqrtN,i+2)
    ax.scatter(*tmpx.T[:2])

def plotHMapEffects(s,alpha=0.1,Nsamp=1000):
  D = s.D
  Htrans = s.Htrans
  cpairs = zip(
    (['r','b','k'][i%3] for i in count(0)),
    zip(xrange(D),[j%D for j in xrange(1,D+1)])
  )
  x = sampleSCrossSD(r0=1.0,r1=0.3,D=D-1,N=Nsamp)
  tmpx = array(x)
  for h,A in zip(Htrans.H,Htrans.A):
    f = h.f
    figure(figsize=(12,12))
    subplot(2,2,1)
    for c,(i,j) in cpairs:
      scatter(tmpx[:,i],tmpx[:,j],alpha=alpha,c=c)
    subplot(2,2,2)
    bt = h.computebtilde(tmpx)
    if bt.shape[1]==1:
      scatter(zeros(bt.shape[0]),bt[:,0],alpha=alpha)
      lnes = f.getEllipse()
      for l in lnes:
        hlines(l,-1,1)
      subplot(2,2,3)
      a = applyAffine(h.ga,tmpx[:,:h.split])
      if a.shape[1]>2:
        scatter(a[:,0],a[:,1],alpha=alpha)
      else:
        scatter(zeros_like(a[:,0]),a[:,0],alpha=alpha)
    else:
      scatter(*bt.T,alpha=alpha)
      plot(*f.getEllipse())
      subplot(2,2,3)
      a = applyAffine(h.ga,tmpx[:,:h.split])
      scatter(zeros(a.shape[0]),a[:,0],alpha=alpha)
    subplot(2,2,4)
    tmpx = h(tmpx)
    for c,(i,j) in cpairs:
      scatter(tmpx[:,i],tmpx[:,j],alpha=alpha,c=c)
    tmpx = dot(A,tmpx.T).T

if __name__=="__main__":
  pass
