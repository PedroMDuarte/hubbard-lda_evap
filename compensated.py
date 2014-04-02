import numpy as np
import matplotlib.pyplot as plt
import matplotlib


from vec3 import vec3, cross
import scipy.constants as C 

## Gauss beam and Lattice beam potentials

def beam(xb,yb,zb,wx,wy,wavelen):
    zRx = np.pi * wx**2 / wavelen
    zRy = np.pi * wy**2 / wavelen 
    
    sqrtX = np.sqrt( 1 + np.power(zb/zRx,2) ) 
    sqrtY = np.sqrt( 1 + np.power(zb/zRy,2) ) 
    return np.exp( -2.*( np.power(xb/(wx*sqrtX ),2) + np.power(yb/(wy*sqrtY),2) )) / sqrtX / sqrtY

def U2uK( wavelen ):
    Cc = C.c * 1e6 # speed of light um s^-1
    Gamma = 2*np.pi *5.9e6 # linewidth s^-1
    
    omega0  = 2*np.pi*Cc / .671
    omegaL  = 2*np.pi*Cc / wavelen
    intensity = 1.0 
    depthJ  = (intensity)* -3*np.pi* Cc**2*Gamma / ( 2*omega0**3) * ( 1/(omega0 - omegaL )  + 1/(omega0 + omegaL ) ) # Joule
    depthuK = depthJ / C.k  *1e6 # C.k is Boltzmann's constant
    return depthuK

def Erecoil( wavelen, mass):
    inJ  = C.h**2 / ( 2* mass*C.physical_constants['unified atomic mass unit'][0] * (wavelen*1e-6)**2 ) 
    inuK = inJ / C.k *1e6
    return inuK
    
                  

class GaussBeam:
    """Properties of a gaussian beam""" 
    def __init__( self,
                 **kwargs ):
        self.mW = kwargs.get('mW',1000.0 )
        self.w  = kwargs.get('waists', (30.,30.) )
        self.l  = kwargs.get('wavelength', 1.064 )
        #
        self.axis   = kwargs.get('axis', (np.pi/2,0.) )
        self.origin = kwargs.get('origin', vec3(0,0,0) )
        
        
        # Make sure vectors are of type(vec3)
        self.axisvec = vec3()
        th = self.axis[0]
        ph = self.axis[1]
        self.axisvec.set_spherical( 1, th, ph) 
        self.origin = vec3(self.origin)
        
        # Calculate two orthogonal directions 
        # which will be used to specify the beam waists
        self.orth0 = vec3( np.cos(th)*np.cos(ph) , np.cos(th)*np.sin(ph), -1.*np.sin(th) )
        self.orth1 = vec3( -1.*np.sin(ph), np.cos(ph), 0. )
        
    def transform(self, X, Y, Z):
        # coordinates into beam coordinates 
        zb = X*self.axisvec[0] + Y*self.axisvec[1] + Z*self.axisvec[2]
        xb = X*self.orth0[0]   + Y*self.orth0[1]   + Z*self.orth0[2]
        yb = X*self.orth1[0]   + Y*self.orth1[1]   + Z*self.orth1[2]
        return xb,yb,zb
        
    def __call__( self, X, Y, Z):
        xb,yb,zb = self.transform( X,Y,Z)
        
        gauss = beam( xb,yb,zb, self.w[0], self.w[1], self.l)
        intensity = (2/np.pi)* self.mW/1000. /self.w[0]/self.w[1] *gauss  # W um^-2
        
        return U2uK(self.l)*intensity
        
class LatticeBeam(GaussBeam):
    def __init__(self, **kwargs):
        """Lattice beam, with retro factor and polarization """
        GaussBeam.__init__(self, **kwargs)
        self.scale = kwargs.get('scale',10.)
        self.mass  = kwargs.get('mass', 6.0)
        self.Er    = kwargs.get('Er', 7.0)
        self.retro = kwargs.get('retro', 1.0)
        self.alpha = kwargs.get('alpha', 1.0)
        self.Er0  = Erecoil( self.l , self.mass)  
        self.mW = 1000*(self.Er/4.)*self.Er0 / np.abs( U2uK(self.l)*2/np.pi )  * self.w[0]*self.w[1] 
        
        
    def __call__( self, X, Y, Z):
        xb,yb,zb = self.transform( X,Y,Z)
        
        gauss = beam( xb,yb,zb, self.w[0], self.w[1], self.l)
        intensity = (2/np.pi)* self.mW/1000. /self.w[0]/self.w[1] *gauss  # W um^-2
        
        lattice =  np.power(np.cos(2*np.pi/self.l * zb/self.scale ),2)*4*np.sqrt(self.retro*self.alpha)\
                 + ( 1 + self.retro - 2*np.sqrt(self.retro*self.alpha) )
        
        return U2uK(self.l)*intensity*lattice
    
    def getBottom( self, X, Y, Z):
        xb,yb,zb = self.transform( X,Y,Z)
        
        gauss = beam( xb,yb,zb, self.w[0], self.w[1], self.l)
        intensity = (2/np.pi)* self.mW/1000. /self.w[0]/self.w[1] *gauss  # W um^-2
        
        latticeBot = 4*np.sqrt(self.retro*self.alpha)  + 1 + self.retro - 2*np.sqrt(self.retro*self.alpha)
        return U2uK(self.l)*intensity * latticeBot
    
    def getV0( self, X, Y, Z):
        xb,yb,zb = self.transform( X,Y,Z)
        
        gauss = beam( xb,yb,zb, self.w[0], self.w[1], self.l)
        intensity = (2/np.pi)* self.mW/1000. /self.w[0]/self.w[1] *gauss  # W um^-2
    
        latticeV0  = 4*np.sqrt(self.retro*self.alpha) 
        return np.abs(U2uK(self.l)*intensity * latticeV0)

## Define the potential class, useful for making plots

from mpl_toolkits.mplot3d import axes3d

class potential:
    def __init__(self, beams, **kwargs ):
        self.units = kwargs.get('units', ('$\mu\mathrm{K}$', 1.))
        self.unitlabel = self.units[0]
        self.unitfactor = self.units[1] 
        self.beams = beams 
        
    def evalpotential( self, X, Y, Z):
        EVAL = np.zeros_like(X) 
        for b in self.beams:
            EVAL += b(X,Y,Z)
        return EVAL* self.unitfactor

    def line_points( self, **kwargs ):
        npoints = kwargs.get('npoints', 320)
        extents = kwargs.get('extents',None)
        lims    = kwargs.get('lims', (-80.,80.))
        direc   = kwargs.get('direc', (np.pi/2, 0.))
        origin  = kwargs.get('origin', vec3(0.,0.,0.))

        if extents is not None:
            lims = (-extents, extents)

        # Prepare set of points for plot 
        t = np.linspace( lims[0], lims[1], npoints )
        unit = vec3()
        th = direc[0]
        ph = direc[1] 
        unit.set_spherical(1, th, ph) 
        # Convert vec3s to ndarray
        unit = np.array(unit)
        origin = np.array(origin) 
        #
        XYZ = origin + np.outer(t, unit)
        X = XYZ[:,0]
        Y = XYZ[:,1]
        Z = XYZ[:,2] 
        return t, X, Y, Z, lims
        
        
    def plotLine(self, **kwargs):
        gs      = kwargs.get('gs',None)
        figGS   = kwargs.get('figGS',None)
        func    = kwargs.get('func',None)
        
        if func is None:
            func = self.evalpotential
        
        t, X,Y,Z, lims = self.line_points( **kwargs ) 
        
        
        if gs is None:
            # Prepare figure
            fig = plt.figure(figsize=(9,4.5))
            gs = matplotlib.gridspec.GridSpec( 1,2, wspace=0.2)
        
            ax0 = fig.add_subplot( gs[0,0], projection='3d')
            ax1 = fig.add_subplot( gs[0,1])
        else:
            gsSub0 = matplotlib.gridspec.GridSpecFromSubplotSpec( 2,2, subplot_spec=gs,\
                         width_ratios=[1,1.6], height_ratios=[1,1],\
                         wspace=0.25)
            ax0 = figGS.add_subplot( gsSub0[0,0], projection='3d')
            ax1 = figGS.add_subplot( gsSub0[0:2,1] )
            ax2 = figGS.add_subplot( gsSub0[1,0] )
            
            ax1.set_xlim( lims[0],lims[1])
            ax2.set_xlim( lims[0]/2.,lims[1]/2.)
            ax2.grid()
            ax2.set_xlabel('$\mu\mathrm{m}$', fontsize=14)

        y2lims = kwargs.pop('y2lims', None)
        if y2lims is not None:
            ax2.set_ylim( *y2lims )
            
            
        ax0.plot(X, Y, Z, c='blue', lw=3)
        ax0.set_xlabel('X')
        ax0.set_ylabel('Y')
        ax0.set_zlabel('Z')
        lmin = min([ ax0.get_xlim()[0], ax0.get_ylim()[0], ax0.get_zlim()[0] ] )
        lmax = max([ ax0.get_xlim()[1], ax0.get_ylim()[1], ax0.get_zlim()[1] ] )
        ax0.set_xlim( lmin, lmax )
        ax0.set_ylim( lmin, lmax )
        ax0.set_zlim( lmin, lmax )
        LMIN = np.ones_like(X)*lmin
        LMAX = np.ones_like(X)*lmax
        ax0.plot(X, Y, LMIN, c='gray', lw=2,alpha=0.3)
        ax0.plot(LMIN, Y, Z, c='gray', lw=2,alpha=0.3)
        ax0.plot(X, LMAX, Z, c='gray', lw=2,alpha=0.3)
        ax0.set_yticklabels([])
        ax0.set_xticklabels([])
        ax0.set_zticklabels([])
        ax0.text2D(0.05, 0.87, kwargs.get('ax0label',None),transform=ax0.transAxes)
        
        # Evaluate function at points and make plot
        EVAL = func(X,Y,Z, **kwargs)
        # EVAL can be of various types, handled below
        Emin =[]; Emax=[]
        if isinstance(EVAL,list):
            for p in EVAL:
                if isinstance(p,dict):
                    if 'y' in p.keys():
                        whichax = p.get('axis',1)
                        axp = ax2 if whichax ==2 else ax1
                        porder = p.get('zorder',2)
                        labelstr = p.get('label',None)
                        
                        fill = p.get('fill', False)
                        if fill:
                            ydat = p.get('y',None)
                            if ydat is not None:
                                axp.plot(t,ydat[0],
                                         lw=p.get('lw',2.),\
                                         color=p.get('color','black'),\
                                         alpha=p.get('fillalpha',0.5),\
                                         zorder=porder,\
                                         label=labelstr
                                         )
                                axp.fill_between( t, ydat[0], ydat[1],\
                                                  lw=p.get('lw',2.),\
                                                  color=p.get('color','black'),\
                                                  facecolor=p.get('fillcolor','gray'),\
                                                  alpha=p.get('fillalpha',0.5),\
                                                  zorder=porder
                                                ) 
                                Emin.append( min( ydat[0].min(), ydat[1].min() ))
                                Emax.append( max( ydat[0].max(), ydat[1].max() )) 
                        else:
                            ydat = p.get('y',None)
                            if ydat is not None:
                                axp.plot( t, ydat,\
                                          lw=p.get('lw',2.),\
                                          color=p.get('color','black'),\
                                          alpha=p.get('alpha',1.0),\
                                          zorder=porder,\
                                          label=labelstr
                                        )
                                Emin.append( ydat.min() ) 
                                Emax.append( ydat.max() ) 
                    elif 'figprop' in p.keys():
                        figsuptitle = p.get('figsuptitle',  None)
                        figGS.suptitle(figsuptitle, y=kwargs.get('suptitleY',1.0),fontsize=14) 
                        figGS.text(0.5,kwargs.get('foottextY',1.0),p.get('foottext',None),fontsize=14,
                                   ha='center') 
                          
                else:
                    ax1.plot(t,p); Emin.append(p.min()); Emax.append(p.max())
            
        elif len(EVAL.shape) > 1 :
            for i,E in enumerate(EVAL):
                ax1.plot(t,E); Emin.append(E.min()); Emax.append(E.max())
                
        else:
            ax1.plot( t, EVAL); Emin.append(E.min()), Emax.append(E.max())
            
        Emin = min(Emin); Emax=max(Emax)
        dE = Emax-Emin
        
        ax1.grid()
        ax1.set_xlabel('$\mu\mathrm{m}$', fontsize=14)
        ax1.set_ylabel( self.unitlabel, rotation=0, fontsize=14, labelpad=-5 )
        
        
        # Finalize figure
        if gs is None:
            gs.tight_layout(fig, rect=[0.,0.,1.0,1.0])
            return Emin-0.05*dE, Emax+0.05*dE, ax1
        else:
            ax2.xaxis.set_major_locator( matplotlib.ticker.MultipleLocator(20) ) 
            ax2.xaxis.set_minor_locator( matplotlib.ticker.MultipleLocator(10) ) 
            return Emin-0.05*dE, Emax+0.05*dE, ax1, ax2
        
    def plot3Line(self,**kwargs):
        lines = kwargs.get('lines','all')
        if lines == '111':
            fig = plt.figure(figsize=(6.,5.))
            gs = matplotlib.gridspec.GridSpec(1,1) 
            lineiter = [3] 
            gsindex = [ None, None, None, (0,0) ] 
            tightrect = [0.,0.06, 1.0, 0.92]
            suptitleY = 0.96
            foottextY = 0.04
        elif lines == '100':
            fig = plt.figure(figsize=(6.,5.))
            gs = matplotlib.gridspec.GridSpec(1,1) 
            lineiter = [0] 
            gsindex = [ (0,0), None, None, None ] 
            tightrect = [0.,0.06, 1.0, 0.92]
            suptitleY = 0.96
            foottextY = 0.04
        else:
            fig = plt.figure(figsize=(12,9.))
            gs = matplotlib.gridspec.GridSpec( 2,2, wspace=0.2)
            lineiter = range(4) 
            gsindex = [ (0,0), (1,0), (0,1), (1,1) ] 
            tightrect = [0.,0., 1.0, 0.95]
            suptitleY = 0.975
            foottextY = 0.025
        
        direcs = [ (np.pi/2, 0.), (np.pi/2, np.pi/2), (0., np.pi), (np.arctan(np.sqrt(2)), np.pi/4) ]
        ax0labels = ['$\mathbf{100}$', '$\mathbf{010}$', '$\mathbf{001}$', '$\mathbf{111}$' ]
                     
        
        
        Ax1 = []; Ax2 = []
        Ymin =[]; Ymax=[]
        for i in lineiter:
            ymin, ymax, ax1, ax2 = self.plotLine( direc=direcs[i], 
                                                  gs= gs[gsindex[i]], 
                                                  figGS=fig, suptitleY=suptitleY,
                                                  foottextY=foottextY,
                                                  ax0label=ax0labels[i], **kwargs) 
            
            Ymin.append(ymin); Ymax.append(ymax); Ax1.append(ax1); Ax2.append(ax2)

        Ymin = min(Ymin); Ymax = max(Ymax)
        for ax in Ax1:
            ax.set_ylim( Ymin, Ymax)
            
        Ax1[0].legend( bbox_to_anchor=(1.06,0.005), \
            loc='lower right', numpoints=1, labelspacing=0.2,\
             prop={'size':10}, handlelength=1.1, handletextpad=0.5 )
        Ax2[0].legend( bbox_to_anchor=(1.10,1.10), \
            loc='upper right', numpoints=1, labelspacing=0.2, \
             prop={'size':10}, handlelength=1.1, handletextpad=0.5 )
            
        gs.tight_layout(fig, rect=tightrect)
        return fig
         
    
        
    def plotCross(self,
            axes = None,
            func = None,
            origin  = vec3(0.,0.,0.), 
            normal   = (np.pi/2, 0),
            lims0   = (-50,50), 
            lims1   = (-50,50),
            extents = None,
            npoints = 240 ):
        
        if func is None:
            func = self.evalpotential
        
        if extents is not None:
            lims0 = (-extents, extents)
            lims1 = (-extents, extents)
        
        # Make the unit vectors that define the plane
        unit = vec3()
        th = normal[0]
        ph = normal[1]
        unit.set_spherical( 1, th, ph) 
        #orth1 = -1.*vec3( np.cos(th)*np.cos(ph) , np.cos(th)*np.sin(ph), -1.*np.sin(th) )
        orth0 = vec3( -1.*np.sin(ph), np.cos(ph), 0. )
        orth1 = cross(unit,orth0)
        
        t0 = np.linspace( lims0[0], lims0[1], npoints )
        t1 = np.linspace( lims1[0], lims1[1], npoints ) 
        
        # Obtain points on which function will be evaluated
        T0,T1 = np.meshgrid(t0,t1)
        X = origin[0] + T0*orth0[0] + T1*orth1[0] 
        Y = origin[1] + T0*orth0[1] + T1*orth1[1]
        Z = origin[2] + T0*orth0[2] + T1*orth1[2] 
        
    
        if axes is None:
            # Prepare figure
            fig = plt.figure(figsize=(9,4.5))
            gs = matplotlib.gridspec.GridSpec( 1,2, wspace=0.2)
        
            ax0 = fig.add_subplot( gs[0,0], projection='3d')
            ax1 = fig.add_subplot( gs[0,1])
        else:
            ax0 = axes[0]
            ax1 = axes[1]
        
        
        # Plot the reference surface
        ax0.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3, linewidth=0.)
        ax0.set_xlabel('X')
        ax0.set_ylabel('Y')
        ax0.set_zlabel('Z')
        lmin = min([ ax0.get_xlim()[0], ax0.get_ylim()[0], ax0.get_zlim()[0] ] )
        lmax = max([ ax0.get_xlim()[1], ax0.get_ylim()[1], ax0.get_zlim()[1] ] )
        ax0.set_xlim( lmin, lmax )
        ax0.set_ylim( lmin, lmax )
        ax0.set_zlim( lmin, lmax )
        ax0.set_yticklabels([])
        ax0.set_xticklabels([])
        ax0.set_zticklabels([])
        
        
        # Evaluate function at points and plot
        EVAL = func(X,Y,Z)
  
        T1_1d = np.ravel(T1)
        EVAL_1d = np.ravel(EVAL)
        
        im =ax1.pcolormesh(T0, T1, EVAL, cmap = plt.get_cmap('jet')) # cmaps:  rainbow, jet
        plt.axes( ax1)
        cbar = plt.colorbar(im)
        cbar.set_label(self.unitlabel, rotation=0 )#self.unitlabel
        
        if axes is None:
            # Finalize figure
            gs.tight_layout(fig, rect=[0.,0.,1.0,1.0])
            
        return EVAL.min(), EVAL.max(), im

    def plot3Cross(self, **kwargs):
        fig = plt.figure(figsize=(8,8))
        gs = matplotlib.gridspec.GridSpec( 3,2, wspace=0.2)
        
        normals = [ (np.pi/2, 0.), (np.pi/2, -np.pi/2), (0., -1.*np.pi/2) ] 
        yMin = 1e16
        yMax =-1e16
        Ims = []
        for i in range(3):
            ax0 = fig.add_subplot( gs[i,0], projection='3d')
            ax1 = fig.add_subplot( gs[i,1]); 
            ymin, ymax, im = self.plotCross( normal=normals[i], axes=(ax0,ax1), **kwargs) 
            Ims.append(im)
            if ymin < yMin: yMin = ymin
            if ymax > yMax: yMax = ymax
        #for im in Ims:
        #    im.set_clim( vmin=yMin, vmax=yMax)
        gs.tight_layout(fig, rect=[0.,0.,1.0,1.0])
        return fig

    def plotLineQty(self, **kwargs):
        line_direction  = kwargs.pop('line_direction', '111')
        direcs = { \
                   '100':(np.pi/2, 0.), \
                   '010':(np.pi/2, np.pi/2), \
                   '001':(0., np.pi), \
                   '111':(np.arctan(np.sqrt(2)), np.pi/4) } 

        t, X,Y,Z, lims = self.line_points( direc= direcs[line_direction] )

        ax  = kwargs.pop('ax', None)
        qty = kwargs.pop('qty', 'density')

        if not self.EvalThermo:
            self.EvalThermoQuantities( X, Y, Z ) 

        if qty == 'density':
            EVAL = self.density_eval 
        elif qty == 'doublons':
            EVAL = self.doublons_eval 
        elif qty == 'entropy':
            EVAL = self.entropy_eval
        elif qty == 'entropy_per_particle':
            EVAL = self.entropy_eval / self.density_eval

        

        plotkwargs = kwargs.pop( 'plotkwargs',None)
        if plotkwargs is not None: 
            ax.plot( t, EVAL, **plotkwargs )
        else:
            ax.plot( t, EVAL )
                  
            
## Load the interpolation data for band structure, onsite interactions, and phase diagrams

# Here the interpolation data for the band structure is loaded from disk
v0 = np.loadtxt('banddat/interpdat_B1D_v0.dat')
NBands = 3
from scipy.interpolate import interp1d
interp0 = []
interp1 = []
for n in range( NBands ):
    interp0.append( interp1d(v0, np.loadtxt('banddat/interpdat_B1D_0_%d.dat'%n) ))
    interp1.append( interp1d(v0, np.loadtxt('banddat/interpdat_B1D_1_%d.dat'%n) ))
    
# Using the interpolation data calculate a function that will get the bottom and top
# of the 3D band in a vectorized way. 

def bands3dvec( v0, NBand=0 ):
    assert len(v0)==3
    bandbot = np.zeros_like( v0[0] ) 
    bandtop = np.zeros_like( v0[0] ) 
    if NBand == 0:
        nband = [0, 0, 0]
    elif NBand == 1:
        v0.sort(axis=0)
        nband = [1, 0, 0]
    else:
        return None
    for i in range(3):
        in1d = nband[i] 
        if in1d%2 ==0:
            bandbot += interp0[in1d](v0[i]) 
            bandtop += interp1[in1d](v0[i])
        else:
            bandbot += interp1[in1d](v0[i])
            bandtop += interp0[in1d](v0[i])
    return np.array((bandbot,bandtop))

#Here the interpolation data for the on-site interactions is loaded from disk
from scipy.interpolate import interp1d
wFInterp = interp1d( np.loadtxt('banddat/interpdat_wF_v0.dat'), np.loadtxt('banddat/interpdat_wF_wF.dat'))

# Using the interpolation data calculate a function that will get the on-site
# interactions in a vectorized way. 

def Onsite( v0,  a_s=300., lattice_spacing=0.532):
    assert len(v0)==3
    wint = np.ones_like( v0[0] ) 
    for i in range(3):
        wint *= wFInterp( v0[i] )
    # The lattice spacing is given in um
    a0a = 5.29e-11 / (lattice_spacing*1e-6)
    return a_s * a0a * np.power(wint, 1./3.)


# Loading up the phase diagram interpolation functions and some examples of how to use them

from HubbardPhaseDiagramInterp import getFuchsInterp, getHTSEInterp, getFuchsInterp2, getHTSEInterp2
# Notice here that the ones with the suffix 2 are not vectorized.  They can be used for floats. 
# On the otherh and, the ones without the suffix only work for arrays. 

if __name__ == '__main__':
    Temperature = 2.4
    fdens = getFuchsInterp2( Temperature, name="density")
    fdoub = getFuchsInterp2( Temperature, name="doublons")
    fentr = getFuchsInterp2( Temperature, name="entropy")
    
    fHdens = getHTSEInterp2( Temperature, name="density")
    fHdoub = getHTSEInterp2( Temperature, name="doublons")
    fHentr = getHTSEInterp2( Temperature, name="entropy")
    
    print "Temperature = {0:4,.2f}".format(Temperature)
    Uval = 4. ; muval = 10. 
    print "U/t = {0:4,.2f}".format(Uval)
    print "mu  = {0:4,.2f}\n".format(muval)
    print "Method   {0:>10s}{1:>10s}".format( "Fuchs", "HTSE")
    print "-"*40
    print "Density  {0:10,.2f}{1:10,.2f}".format( float(fdens(Uval, muval)), float(fHdens(Uval, muval)) )
    print "Doublons {0:10,.2f}{1:10,.2f}".format( float(fdoub(Uval, muval)), float(fHdoub(Uval, muval)) )
    print "Entropy  {0:10,.2f}{1:10,.2f}".format( float(fentr(Uval, muval)), float(fHentr(Uval, muval)) )    


## Define the simpleCubic class, which inherits from the potential class

from scipy import integrate
from scipy import optimize
from scipy.interpolate import interp1d

class simpleCubic( potential ):
    def __init__(self, **kwargs):
        """Simple cubic lattice potential """
        
        # Initialize lattice part 
        axes= [ (np.pi/2,0.), (np.pi/2, np.pi/2), (0,0) ] 
        self.l  = kwargs.get('wavelength', 1.064)
        self.m  = kwargs.get('mass', 6.)
        self.w  = kwargs.get('waists', ((47.,47.), (47.,47.), (47.,47.)) )
        self.r  = kwargs.get('retro', (1.,1.,1.) )
        self.a  = kwargs.get('alpha', (1.,1.,1.) )
        self.scale = kwargs.get('scale', 10.)
        self.EvalThermo = False 
        
        self.Er0 = Erecoil(self.l, self.m)
        
        if 'allIR' in kwargs.keys():
            self.Er = [kwargs.get('allIR', 7.0 )]*3
        else:
            self.Er = kwargs.get('Er', (7.0, 7.0, 7.0) ) 
        
        lattbeams = [ LatticeBeam( axis=axes[i], Er=self.Er[i], wavelength=self.l, scale=self.scale,\
                                   waists=self.w[i], retro=self.r[i], alpha=self.a[i] ) for i in range(3) ] 
        
        potential.__init__(self, lattbeams, units=('$E_{R}$', 1/self.Er0) )
        
        self.GRw  = kwargs.get('green_waists', ((40.,40.), (40.,40.), (40.,40.)) ) 
        if 'allGR' in kwargs.keys():
            self.GREr = [kwargs.get('allGR', 4.0 )]*3
        else:
            self.GREr = kwargs.get('green_Er', (4.0, 4.0, 4.0) ) 
        self.GRl  = kwargs.get('green_wavelength', 0.532)
        
        # Express the power requiered for each GR beam, given the Er compensation value
        # in units of the lattice recoil, and given the GR beam waists
        GRmW = [ 1000.* self.GREr[i]*  self.Er0/np.abs(U2uK(self.GRl)*2/np.pi) \
                      * self.GRw[i][0]*self.GRw[i][1]  for i in range(3)]
        
        self.greenbeams = [  GaussBeam( axis=axes[i], mW=GRmW[i], waists=self.GRw[i], wavelength=self.GRl) for i in range(3) ]
        
        # Initialize quantities that will be used for the phase diagram
        #
        self.T = kwargs.get('Temperature', 1.6 )
        self.fHdens = getHTSEInterp( self.T, name="density")
        self.fHdoub = getHTSEInterp( self.T, name="doublons")
        self.fHentr = getHTSEInterp( self.T, name="entropy" )
        self.a_s = kwargs.get('a_s',300.)
        #
        
        # Make a cut line along 111 to calculate integrals of the
        # thermodynamic quantities
        direc111 = (np.arctan(np.sqrt(2)), np.pi/4) 
        unit = vec3(); th = direc111[0]; ph = direc111[1] 
        unit.set_spherical(1, th, ph); unitArr = np.array(unit)
        t = np.linspace( -80, 80, 320 )
        XYZ = np.outer( t, unitArr)
        self.X111 = XYZ[:,0]
        self.Y111 = XYZ[:,1]
        self.Z111 = XYZ[:,2] 
        # Below we get the signed distance from the origin
        self.r111 =  self.X111*unit[0] + self.Y111*unit[1] + self.Z111*unit[2] 
        
        
        # Go ahead and calculate all relevant quantities along the 111
        # direction
        self.V0_111 = self.V0(self.X111,self.Y111,self.Z111) 
        self.Bottom_111 = self.Bottom(self.X111,self.Y111,self.Z111)
        self.bands_111 = bands3dvec( self.V0_111, NBand=0 ) 
        
        self.Ezero_111 = (self.bands_111[1]+self.bands_111[0])/2. + self.Bottom_111
        self.Ezero0_111 = self.Ezero_111.min()
        
        self.tunneling_111 = (self.bands_111[1]-self.bands_111[0])/12. 
        self.onsite_t_111 = Onsite( self.V0_111, a_s=self.a_s, lattice_spacing= self.l/2. ) / self.tunneling_111
        
        
        
        self.verbose = kwargs.get('verbose', False)
        
        
        # Initialize global chemical potential and atom number
        # globalMu can be given directly or can be specified via the 
        # number of atoms.  If the Natoms is specified we calculate 
        # the required gMu using this function: 
        muBrent = kwargs.get('muBrent', (-1, 6.))
        if 'Natoms' in kwargs.keys():
            self.Number = kwargs.get('Natoms', 3e5)
            fN = lambda x : self.getNumber(x)- self.Number
            if self.verbose:
                print "Searching for globalMu => N=%.0f, "% self.Number,
            self.globalMu, brentResults = optimize.brentq(fN, muBrent[0], muBrent[1], xtol=1e-2, rtol=2e4, full_output=True) 
            if self.verbose:
                print "gMu=%.3f, " % brentResults.root,
                print "n_iter=%d, " % brentResults.iterations,
                print "n eval=%d, " % brentResults.function_calls,
                print "conv?=", brentResults.converged
        else :
            # globalMu is given in Er, and is measured from the value
            # of Ezero at the center of the potential
            # When using it in the phase diagram it has to be changed to
            # units of the tunneling
            self.globalMu = kwargs.get('globalMu', 0.15)
            if  self.globalMu == 'halfMott':
                self.globalMu = (self.onsite_t_111 * self.tunneling_111).max()/2.
             
    
        
        # After the chemical potential is established the local chemical
        # potential along 111 can be defined.  It is used for column density
        # plots and for calculating other thermodynamic quantities
        gMuZero = self.Ezero0_111 + self.globalMu
        self.localMu_t_111= (gMuZero - self.Ezero_111) / self.tunneling_111
        
        
        
        # Obtain trap integrated values of the thermodynamic quantities
        self.Number  = self.getNumber( self.globalMu )
        self.NumberD = self.getNumberD()
        self.Entropy = self.getEntropy()
        
        # MAKE FIGURE LABELS
        # V Lattice
        if len(np.unique(self.Er))==1:
            VLlabel = '$V_{L}=%.1fE_{R}$' % self.Er[0] 
        else:
            VLlabel = '$V_{Lx}=%.1f, V_{Ly}=%.1f, V_{Lz}=%.1f$' % self.Er 
        # V Green
        if len(np.unique(self.GREr))==1:
            VGlabel = '$V_{G}=%.1fE_{R}$' % self.GREr[0] 
        else:
            VGlabel = '$V_{Gx}=%.1f, V_{Gy}=%.1f, V_{Gz}=%.1f$' % self.GREr 
        # Scattering length
        aslabel = '$a_{s}=%.0f$' % self.a_s 
        # U/t label 
        Utlabel = '$U/t=%.1f$' % self.onsite_t_111.max()
        # Temperature label
        Tlabel = '$T/t=%.1f$' % self.T
        
        self.figlabel = ',  '.join([VLlabel, VGlabel, aslabel, Utlabel, Tlabel])

        self.Nlabel = r'$N=%.2f\times 10^{5}$' % (self.Number/1e5)
        self.Dlabel = r'$D=%.3f$' % ( self.NumberD / self.Number )
        self.Slabel = r'$S/N=%.2fk_{\mathrm{B}}$' % ( self.Entropy / self.Number )
        self.foottext = ',  '.join([self.Nlabel, self.Dlabel, self.Slabel]) 

        # Calculate energies to estimate eta parameter for evaporation
        self.globalMuZ0 = self.Ezero0_111 + self.globalMu

        # Make a cut line along 100 to calculate the threshold for evaporation
        direc100 = (np.pi/2, 0.) 
        unit = vec3(); th = direc100[0]; ph = direc100[1] 
        unit.set_spherical(1, th, ph); unitArr = np.array(unit)
        t = np.linspace( -100, 100, 320 )
        XYZ100 = np.outer( t, unitArr)
        self.X100 = XYZ100[:,0]
        self.Y100 = XYZ100[:,1]
        self.Z100 = XYZ100[:,2]
        self.V0_100 = self.V0(self.X100,self.Y100,self.Z100) 
        bandBot_100 = bands3dvec( self.V0_100, NBand = 0)[0] + self.Bottom( self.X100, self.Y100, self.Z100 ) 
        self.evapTH0_100 = bandBot_100.max() 


        self.evapTH0 = bands3dvec( self.V0( 100., 0., 0. ), NBand=0 )[0] + self.Bottom(100.,0.,0.)
        self.LowestE0 = np.amin( self.bands_111[0] +  self.Bottom_111)

        #  The correct evaporation thresholds is evapTH0_100, this one accounts for
        #  situations in which there is a local barrier and then the band along 100 goes down
        #  for large 100 distance 

        #print " evapTH0_100 = %.2f      evapTH0 = %.2f"%(self.evapTH0_100, self.evapTH0)


        # Estimation of eta is done below 
        #print "mu global = %.3g" % self.globalMuZ0 
        #print "evap th   = %.3g" % self.evapTH0
        #print "lowest E  = %.3g" % self.LowestE0
        mu = self.globalMuZ0 - self.LowestE0 
        th = self.evapTH0_100 - self.LowestE0
        self.EtaEvap = th/mu 
        #print "mu = %.3g" % mu
        #print "th = %.3g" % th
        #print "eta = %.3g" % (th/mu)
        #print "th-mu = %.3g" % (th-mu)


        # Calculate second derivative of the band bottom to assess whether
        # or not the potential is a valid potential (no split by compensation )
        positive_r =  self.r111  >= 0 
        # absolute energy of the lowest band, elb
        elb  = (self.bands_111[0] + self.Bottom_111)[ positive_r ]
            
        if elb[1] - elb[0] >= 0: 
            self.bandBot_LocalMin = True
        else:
            self.bandBot_LocalMin = False

        
        
        
        
    def Bottom0( self ):
        EVAL = 0.
        for b in self.beams:
            EVAL += b.getBottom( 0., 0., 0.)
        for g in self.greenbeams:
            EVAL += g(0.,0.,0.)
        EVAL = EVAL*self.unitfactor
        return EVAL
            
        
    def Bottom( self, X, Y, Z):
        EVAL = np.zeros_like(X)
        for b in self.beams:
            EVAL += b.getBottom( X, Y, Z)
        for g in self.greenbeams:
            EVAL += g(X,Y,Z)
        return EVAL*self.unitfactor
    
    def V0( self, X, Y, Z):
        EVAL = []
        for b in self.beams:
            EVAL.append( b.getV0( X, Y, Z)*self.unitfactor )
        return np.array(EVAL)
            
    
    def LatticeMod( self, X, Y, Z):
        V0s = self.V0( X, Y, Z )
        Mod = np.amin(V0s, axis=0)
        return self.Bottom(X,Y,Z) + Mod*np.power( np.cos( 2.*np.pi*np.sqrt(X**2 + Y**2 + Z**2 ) / self.l / self.scale ), 2)
    
    def getNumber( self, gMu, verbose=False):
        gMuZero = self.Ezero0_111 + gMu
        localMu = gMuZero - self.Ezero_111
        localMu_t = localMu / self.tunneling_111
        density = self.fHdens( self.onsite_t_111, localMu_t )


        # Under some circumnstances the green compensation can 
        # cause dips in the density profile
        # Experimentally we have seen that we do not handle these very
        # well, so we want to avoid them at all cost 

        # The occurence of this is flagged by a change in the derivative
        # of the radial density.  This derivative should always be negative
        # if the derivative is positive it means that the density is 
        # increasing for larger radii. 
        #
        # Here we find the changes of the density for radii larger
        # that 1/4 of the IR beam waist.  If the density slope is positive
        # at some point then we report that this is not a valid point 
        # in parameter space. 
        # 
        # find the point at which the density changes derivative
        radius_check = 0.25 *  \
                       sum([ (wi[0] + wi[1]) for wi in self.w]  , 0. ) /6. 
        posradii = self.r111 > radius_check 
        posdens =  density[ posradii ]
        neg_slope = np.diff( posdens ) > 1e-4
        n_neg_slope = np.sum( neg_slope )
        if n_neg_slope > 0:  
            msg = "Radial density profile along 111 has a positive slope"
            if verbose:
                print "radius check start = ", radius_check
                print msg
                print posdens
                print np.diff( posdens ) > 1e-4
            raise ValueError(msg) 

        if verbose:
            print " posdens len = ",len(posdens)
            print " n_neg_slope = ",n_neg_slope
         
        dens = density[~np.isnan(density)]
        r = self.r111[~np.isnan(density)]
        return np.power(self.l/2.,-3)*2*np.pi*integrate.simps(dens*(r**2),r)

    def getNumberD( self):
        doublons = self.fHdoub( self.onsite_t_111, self.localMu_t_111 ) 
        doub = doublons[~np.isnan(doublons)]
        r = self.r111[~np.isnan(doublons)]
        return 2.*np.power(self.l/2.,-3)*2*np.pi*integrate.simps(doub*(r**2),r)
    
    def getEntropy( self):
        entropy  = self.fHentr( self.onsite_t_111, self.localMu_t_111 ) 
        entr = entropy[~np.isnan(entropy)]
        r = self.r111[~np.isnan(entropy)]
        return np.power(self.l/2.,-3)*2*np.pi*integrate.simps(entr*(r**2),r) 
        
        
    
    def column( self, plotlist):
        # Calculates the column density for the thermodynamic quantities
        qtys = [] 
        titles = []
        colors = []
        if "density" in plotlist:
            qtys.append( self.fHdens( self.onsite_t_111, self.localMu_t_111 ) ) 
            titles.append(r'$n_{\mathrm{col}}$') 
        if "doublons" in plotlist:
            qtys.append( self.fHdoub( self.onsite_t_111, self.localMu_t_111 ) ) 
            titles.append(r'$d_{\mathrm{col}}$')
        if "entropy" in plotlist:
            qtys.append( self.fHentr( self.onsite_t_111, self.localMu_t_111 ) )
            titles.append(r'$s_{\mathrm{col}}$')
        if "moment" in plotlist:
            qtys.append( self.fHdens( self.onsite_t_111, self.localMu_t_111 ) \
                         - 2.*self.fHdoub( self.onsite_t_111, self.localMu_t_111 ) )
            titles.append(r'$\langle\ m \rangle_{\mathrm{col}}$')
      
        
        
        # Prepare the figure
        fig = plt.figure(figsize=(3*len(qtys),6.))
        
        gs = matplotlib.gridspec.GridSpec( 2,2*len(qtys), wspace=0.2,
                                           width_ratios=[15,1]*len(qtys))
        
        axs = [ fig.add_subplot(gs[0,2*i]) for i in range(len(qtys))]
        #axcbar = [ fig.add_subplot(gs[0,2*i+1]) for i in range(len(qtys))]
        axl = [ fig.add_subplot(gs[1,2*i], sharex=axs[i]) for i in range(len(qtys))]
        
        for i,qty in enumerate(qtys):
        
            # First make an interpolation function of the radial function
            qr = qty[~np.isnan(qty)]
            r  = self.r111[~np.isnan(qty)]
            fq = interp1d(r,qr, bounds_error=False, fill_value=0.)
            
            # Then define the cartesian version
            def fCartesian( X, Y, Z):
                rarray = np.sqrt( X**2 + Y**2 + Z**2)
                return fq( rarray )
            
            # With the cartesian in hand it is easy to do the column integral
            z = r
            qrho = np.array([ integrate.simps( fCartesian( rhoval, 0., z ), z ) for rhoval in r ])
            #plt.plot( r, qrho )
            # save the result as an interpolation
            fqrho = interp1d( r, qrho, bounds_error=False, fill_value=0.)
            axl[i].plot( r, fqrho(r) )
            axl[i].set_xlabel('$\mu \mathrm{m}$')
            axl[i].grid()
            # Evaluate function on mesh, and plot
            Xcol,Ycol = np.meshgrid(r,r)
            COL = fqrho( np.sqrt( Xcol**2 + Ycol**2 ) )
            im =axs[i].pcolormesh(Xcol, Ycol, COL, cmap = plt.get_cmap('jet'))
            
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes


            cax = inset_axes(axs[i],
                 width="5%",
                 height="100%",
                 bbox_transform=axs[i].transAxes,
                 bbox_to_anchor=(0.025, 0.05, 1.12, 0.95),
                 loc= 1)
            
            
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            
            plt.axes( axs[i])
            plt.colorbar(im, cax=cax)
            
            #plt.colorbar(im, cax=axcbar[i])
            
            #divider = make_axes_locatable(plt.gca())
            #cax = divider.append_axes("right", "5%", pad="3%")
            #plt.colorbar(im, cax=cax)
            
            
            axs[i].set_title(titles[i],fontsize=14)
            
        # Add the header and footer labels
        fig.suptitle(self.figlabel, y=0.95,fontsize=14) 
        fig.text(0.5,0.05,self.foottext,fontsize=14,
                   ha='center')
        
        # Finalize figure
        gs.tight_layout(fig, rect=[0.,0.08,1.0,0.92])
        return fig
    
    def CheckInhomog( self, **kwargs ):
        """This function will make a plot along 111 of the model parameters:
           U, t, U/t, v0.  It is useful to assess the degree of inhomogeneity in our system"""
        
        # Prepare the figure
        fig = plt.figure(figsize=(8.,4.8))
        fig.suptitle( ' , '.join([ self.figlabel, self.Nlabel]) )
        fig.text( 0.05, 0.86, "Sample is divided in 5 bins, all containing the same number of atoms (see panel 2).\n" + \
                              "Average Fermi-Hubbard parameters $n$, $U$, $t$, and $U/t$ are calculated in each bin (see panels 1, 3, 4, 5 )" )
        
        gs = matplotlib.gridspec.GridSpec( 2,3, wspace=0.2,\
                 left=0.1, right=0.9, bottom=0.05, top=0.9)
        
        # Setup axes
        axn  = fig.add_subplot(gs[0,0])
        axnInt = fig.add_subplot(gs[0,1])
        
        axU  = fig.add_subplot(gs[0,2])
        axt  = fig.add_subplot(gs[1,0])
        axUt = fig.add_subplot(gs[1,1]) 
        axv0 = fig.add_subplot(gs[1,2])

        # Set xlim
        x0 = -60.; x1 = 60.
        axn.set_xlim( x0, x1)
        axnInt.set_xlim( 0., x1 )
        axU.set_xlim( x0, x1 )
        axU.set_ylim( 0., np.amax( self.onsite_t_111 * self.tunneling_111 *1.05 ) )
        axt.set_xlim( x0, x1 )
        axt.set_ylim( 0., 0.3)
        axUt.set_xlim( x0, x1 )
        axUt.set_ylim( 0., np.amax( self.onsite_t_111 * 1.05 )) 
        axv0.set_xlim( x0, x1 )
        
        lw0 = 2.5
        # Plot relevant quantities 
        density_111 = self.fHdens( self.onsite_t_111, self.localMu_t_111 )
        axn.plot( self.r111, density_111, lw=lw0 , color='black')
        
        axU.plot( self.r111, self.onsite_t_111 * self.tunneling_111 , \
                      lw=lw0, label='$U$', color='black') 

        axt.plot( self.r111, self.tunneling_111,lw=lw0, label='$t$', \
                      color='black')


        axUt.plot( self.r111, self.onsite_t_111, lw=lw0, color='black')
        #print "shape of V0 = ", self.V0_111.shape
        axv0.plot( self.r111, self.V0_111[0], lw=lw0, color='black', label='$V_{0}$')
        bandgap_111 = bands = bands3dvec( self.V0_111, NBand=1 )[0] \
                              - bands3dvec( self.V0_111, NBand=0 )[1] 
        axv0.plot( self.r111, bandgap_111, lw=lw0, linestyle=':', color='black', label='$\mathrm{Band\ gap}$') 

        axv0.legend( bbox_to_anchor=(0.06,0.025), \
            loc='lower left', numpoints=3, labelspacing=0.2,\
             prop={'size':7}, handlelength=1.5, handletextpad=0.5 )

        # Define function to calculate cummulative atom number
        
        def NRadius( Radius ):
            # This function calculates the fraction of the atom number 
            # up to a certain Radius
            valid = np.logical_and( np.abs(self.r111) < Radius, ~np.isnan(density_111) )
            r    = self.r111[ valid ] 
            dens = density_111[ valid ] 
            return np.power( self.l/2, -3)*2*np.pi*integrate.simps( dens*(r**2), r) / self.Number
        
        radii = self.r111[ self.r111 > 4. ] 
        NInt = []
        for radius in radii:
            NInt.append( NRadius( radius ) ) 
        NInt = np.array( NInt ) 
        axnInt.plot( radii, NInt, lw=lw0, color='black') 

       
        # Define function to numerically solve for y in a pair of x,y arrays     
        
        def x_solve( x_array, y_array,  yval ):
            # Convert the array to a function and then solve for y==yval
            yf = interp1d( x_array, y_array-yval, kind='cubic') 
            return optimize.brentq( yf, x_array.min(), x_array.max() ) 

        radius1e = x_solve( self.r111[ self.r111 > 0 ] , \
                            density_111[ self.r111 > 0 ] , density_111.max()/np.exp(1.) ) 

        
        # Find the various radii that split the cloud into slots of 20% atom number
        rcut = []
        for Ncut in [0.2, 0.4, 0.6, 0.8 ]:
            rcut.append( x_solve( radii, NInt, Ncut ) )

        # Define functions to average over the shells        
        def y_average( y_array,  x0, x1):
            # Average y_array over the radii x0 to x1,  weighted by density 
            valid = np.logical_and( np.abs(self.r111) < 70., ~np.isnan(density_111) )
            
            r    = self.r111[ valid ] 
            dens = density_111[ valid ]
            y    = y_array[ valid ] 
            
            shell = np.logical_and( r >= x0, r<x1 ) 
            r    = r[shell]
            dens = dens[shell]
            y    = y[shell] 
            
            num = integrate.simps( y* dens*(r**2), r) 
            den = integrate.simps(  dens*(r**2), r) 
            return num/den 
        
        # Define a function here that makes a piecewise function with the average
        # values of a quantity so that it can be plotted
        def binned( x, yqty ):
            x = np.abs(x)
            yavg = [] 
            cond = []
            for x0,x1 in zip( [0.]+rcut,  rcut+[rcut[-1]+20.]):
                cond.append(np.logical_and( x >= x0 , x<x1 ) )
                yavg.append( y_average( yqty, x0, x1) ) 
              
            return np.piecewise( x, cond, yavg ), yavg

        # Calculate and plot the binned quantities
        dens_binned = binned( self.r111, density_111 ) 
        Ut_binned   = binned( self.r111, self.onsite_t_111 )
        U_binned    = binned( self.r111, self.onsite_t_111 * self.tunneling_111 )
        t_binned    = binned( self.r111, self.tunneling_111 )

        peak_dens = np.amax( density_111 )
        peak_t = np.amin( self.tunneling_111 )
        
        axn.fill_between( self.r111, dens_binned[0], 0., \
                          lw=2, color='red', facecolor='red', zorder=2, alpha=0.8)
        axUt.fill_between( self.r111, Ut_binned[0],  0., \
                          lw=2, color='red', facecolor='red', zorder=2, alpha=0.8  )
        axU.fill_between( self.r111, U_binned[0], 0., \
                          lw=2, color='red', facecolor='red',label='$U$', zorder=2, alpha=0.8) 
        axt.fill_between( self.r111, t_binned[0], 0., \
                          lw=2, color='red', facecolor='red',linestyle=':',label='$t$', zorder=2, alpha=0.8)
                         
           
        
        # Set y labels
        axn.set_ylabel(r'$n$')
        axnInt.set_ylabel(r'$N_{<R}$')
        axU.set_ylabel(r'$U\,(E_{R})$')
        axt.set_ylabel(r'$t\,(E_{R})$')
        axUt.set_ylabel(r'$U/t$')
        axv0.set_ylabel(r'$E_{R}$')
        
        for i,ax in enumerate([axn, axnInt, axU, axt, axUt, axv0]):
            ax.text( 0.08,0.86, '%d'%(i+1), transform=ax.transAxes, fontsize=14)
            ax.yaxis.grid()
            ax.set_xlabel(r'$\mu\mathrm{m}$')
            for n,r in enumerate(rcut):
                if n % 2 == 0:
                    ax.axvspan( r, rcut[n+1], facecolor='lightgray') 
                    if i != 1:
                        ax.axvspan(-rcut[n+1], -r, facecolor='lightgray') 
                ax.axvline( r, lw=1.0, color='gray', zorder=1 )
                if i != 1:
                    ax.axvline(-r, lw=1.0, color='gray', zorder=1 )
                
            ax.xaxis.set_major_locator( matplotlib.ticker.MultipleLocator(20) ) 
            ax.xaxis.set_minor_locator( matplotlib.ticker.MultipleLocator(10) )
            
            #labels = [item.get_text() for item in ax.get_xticklabels()]
            #print labels
            #labels = ['' if float(l) % 40 != 0 else l for l in labels ] 
            #ax.set_xticklabels(labels)

        axnInt.xaxis.set_major_locator( matplotlib.ticker.MultipleLocator(10) ) 
        axnInt.xaxis.set_minor_locator( matplotlib.ticker.MultipleLocator(5) )
        
        # Finalize figure
        gs.tight_layout(fig, rect=[0.,0.0,1.0,0.84])

        if kwargs.get('closefig', False):
            plt.close()
        
        return fig, dens_binned[1], Ut_binned[1], U_binned[1], t_binned[1], peak_dens, radius1e, peak_t
            
    
    def EvalThermoQuantities( self, X, Y, Z, **kwargs):
        NBand = kwargs.get('NBand',0.)
        
        # Make 1d array with distances 
        R_unit = vec3( X[-1]-X[0], Y[-1]-Y[0], Z[-1]-Z[0] ); R_unit = R_unit / abs(R_unit)
        # Below we get the signed distance from the origin
        self.R = X*R_unit[0]+Y*R_unit[1]+Z*R_unit[2]
        
        self.V0_ = self.V0(X,Y,Z) 
        self.Bottom_ = self.Bottom(X,Y,Z)
        
        self.LatticeMod_ = self.Bottom_ + np.amin(self.V0_,axis=0)*\
                           np.power( np.cos( 2.*np.pi*self.R / self.l / self.scale ), 2)
            
        self.bands = bands3dvec( self.V0_, NBand=0 ) 
        self.excband = bands3dvec( self.V0_, NBand=1 )
        
        # The zero of energy for this problem is exactly at the center
        # of the lowest band.  This comes from the t_{ii} matrix element
        # which is generally neglected in the Hamiltonian. 
        self.Ezero = (self.bands[1]+self.bands[0])/2. + self.Bottom_
        self.Ezero0 = self.Ezero.min()

        
        # The threshold for evaporation can ge obtaind from 
        # the bottom of the band going far along one beam
        self.evapTH = bands3dvec( self.V0( 100., 0., 0. ), NBand=0 )[0] + self.Bottom(100.,0.,0.)
        self.LowestE = np.amin( self.bands[0] +  self.Bottom_)
        
        # Tunneling, onsite interactions, and localMu for phase diagram
        self.tunneling = (self.bands[1]-self.bands[0])/12. 
        
        self.onsite = Onsite( self.V0_, a_s=self.a_s, lattice_spacing= self.l/2. )
        self.onsite_t = self.onsite / self.tunneling
        
        # Offset the chemical potential for use in the phase diagram
        self.globalMuZ = self.Ezero0 + self.globalMu
        self.localMu_t = ( self.globalMuZ - self.Ezero )/ self.tunneling
        
        # Obtain the thermodynamic quantities
        self.density_eval  = self.fHdens( self.onsite_t, self.localMu_t ) 
        self.doublons_eval = self.fHdoub( self.onsite_t, self.localMu_t ) 
        self.entropy_eval  = self.fHentr( self.onsite_t, self.localMu_t ) 

        self.EvalThermo = True
        return   

    def Bands( self, X, Y, Z, **kwargs):
        NBand = kwargs.get('NBand',0.)
        
        # Make 1d array with distances 
        R_unit = vec3( X[-1]-X[0], Y[-1]-Y[0], Z[-1]-Z[0] ); R_unit = R_unit / abs(R_unit)
        # Below we get the signed distance from the origin
        self.R = X*R_unit[0]+Y*R_unit[1]+Z*R_unit[2]
        
        self.V0_ = self.V0(X,Y,Z) 
        self.Bottom_ = self.Bottom(X,Y,Z)
        
        self.LatticeMod_ = self.Bottom_ + np.amin(self.V0_,axis=0)*\
                           np.power( np.cos( 2.*np.pi*self.R / self.l / self.scale ), 2)
            
        bands = bands3dvec( self.V0_, NBand=0 ) 
        excband = bands3dvec( self.V0_, NBand=1 )
        
        # The zero of energy for this problem is exactly at the center
        # of the lowest band.  This comes from the t_{ii} matrix element
        # which is generally neglected in the Hamiltonian. 
        self.Ezero = (bands[1]+bands[0])/2. + self.Bottom_
        self.Ezero0 = self.Ezero.min()

        
        # The threshold for evaporation can ge obtaind from 
        # the bottom of the band going far along one beam
        self.evapTH = bands3dvec( self.V0( 100., 0., 0. ), NBand=0 )[0] + self.Bottom(100.,0.,0.)
        self.LowestE = np.amin( bands[0] +  self.Bottom_)
        
        # Tunneling, onsite interactions, and localMu for phase diagram
        self.tunneling = (bands[1]-bands[0])/12. 
        
        self.onsite = Onsite( self.V0_, a_s=self.a_s, lattice_spacing= self.l/2. )
        self.onsite_t = self.onsite / self.tunneling
        
        # Offset the chemical potential for use in the phase diagram
        self.globalMuZ = self.Ezero0 + self.globalMu
        self.localMu_t = ( self.globalMuZ - self.Ezero )/ self.tunneling
        
        # Obtain the thermodynamic quantities
        density  = self.fHdens( self.onsite_t, self.localMu_t ) 
        doublons = self.fHdoub( self.onsite_t, self.localMu_t ) 
        entropy  = self.fHentr( self.onsite_t, self.localMu_t ) 
    
        
        # Higher zorder puts stuff in front
        toplot = [ 
                 {'y':(bands[0] + self.Bottom_, self.Ezero ), 'color':'blue', 'lw':2., \
                  'fill':True, 'fillcolor':'blue', 'fillalpha':0.75,'zorder':10, 'label':'$\mathrm{band\ lower\ half}$'},
                 
                 {'y':(self.Ezero + self.onsite, bands[1] + self.Bottom_+self.onsite), 'color':'purple', 'lw':2., \
                  'fill':True, 'fillcolor':'plum', 'fillalpha':0.75,'zorder':10, 'label':'$\mathrm{band\ upper\ half}+U$'},
                  
                 {'y':(excband[0] + self.Bottom_, excband[1] + self.Bottom_ ), 'color':'red', 'lw':2., \
                  'fill':True, 'fillcolor':'pink', 'fillalpha':0.75,'zorder':10, 'label':'$\mathrm{exc\ band}$'},
                 
                 {'y':np.ones_like(X)*self.globalMuZ, 'color':'limegreen','lw':2,'zorder':1.9, 'label':'$\mu_{0}$'},
                 {'y':np.ones_like(X)*self.evapTH0_100, 'color':'#FF6F00', 'lw':2,'zorder':1.9, 'label':'$\mathrm{evap\ th.}$'},
                 
                 {'y':self.Bottom_,'color':'gray', 'lw':0.5,'alpha':0.5},
                 {'y':self.LatticeMod_,'color':'gray', 'lw':1.5,'alpha':0.5,'label':r'$\mathrm{lattice\ potential\ \ }\lambda\times10$'}]  

        entropy_per_particle = kwargs.pop('entropy_per_particle', False)
        if entropy_per_particle:
            toplot = toplot + [                 
                 {'y':entropy/density,  'color':'black', 'lw':1.75, 'axis':2, 'label':'$s_{N}$'} ] 
        else:
            toplot = toplot + [
                 {'y':density,  'color':'blue', 'lw':1.75, 'axis':2, 'label':'$n$'},
                 {'y':doublons, 'color':'red', 'lw':1.75, 'axis':2, 'label':'$d$'},
                 {'y':entropy,  'color':'black', 'lw':1.75, 'axis':2, 'label':'$s_{L}$'},
                 #{'y':entropy/density,  'color':'black', 'lw':1.75, 'axis':2, 'label':'$s_{N}$'},

                 #{'y':density-2*doublons,  'color':'green', 'lw':1.75, 'axis':2, 'label':'$n-2d$'},
                 #{'y':self.localMu_t,  'color':'cyan', 'lw':1.75, 'axis':2, 'label':r'$\mu$'},
                 ]

        toplot = toplot + [ {'figprop':True,'figsuptitle':self.figlabel, 'foottext':self.foottext} ] 
        return toplot  
        

    def get_eta_evap( self ):
        print "mu global = %.3g" % self.globalMuZ 
        print "evap th   = %.3g" % self.evapTH 
        print "lowest E  = %.3g" % self.LowestE
        mu = self.globalMuZ - self.LowestE 
        th = self.evapTH - self.LowestE 
        print "mu = %.3g" % mu
        print "th = %.3g" % th
        print "eta = %.3g" % (th/mu)
        print "th-mu = %.3g" % (th-mu)
        return 


    def PlotBands( self, **kwargs):
        figGS = plt.figure(figsize=(6.,5.))
        gs3Line = matplotlib.gridspec.GridSpec(1,1) 
        tightrect = [0.,0.06, 1.0, 0.92]

        Ax1 = []; Ax2 = []
        Ymin =[]; Ymax=[]

        kwargs['direc'] = (np.arctan(np.sqrt(2)), np.pi/4) 
        kwargs['ax0label']='$\mathbf{111}$'
        kwargs['suptitleY'] = 0.96
        kwargs['foottextY'] = 0.04

        t, X,Y,Z, lims = self.line_points( **kwargs ) 
       
        gs= gs3Line[0,0] 
        
        gsSub0 = matplotlib.gridspec.GridSpecFromSubplotSpec( 2,2, subplot_spec=gs,\
                     width_ratios=[1,1.6], height_ratios=[1,1],\
                     wspace=0.25)
 
        ax0 = figGS.add_subplot( gsSub0[0,0], projection='3d')
        ax1 = figGS.add_subplot( gsSub0[0:2,1] )
        ax2 = figGS.add_subplot( gsSub0[1,0] )
        
        ax1.set_xlim( lims[0],lims[1])
        ax2.set_xlim( lims[0]/2.,lims[1]/2.)
        ax2.grid()
        ax2.set_xlabel('$\mu\mathrm{m}$', fontsize=14)
            
            
        ax0.plot(X, Y, Z, c='blue', lw=3)
        ax0.set_xlabel('X')
        ax0.set_ylabel('Y')
        ax0.set_zlabel('Z')
        lmin = min([ ax0.get_xlim()[0], ax0.get_ylim()[0], ax0.get_zlim()[0] ] )
        lmax = max([ ax0.get_xlim()[1], ax0.get_ylim()[1], ax0.get_zlim()[1] ] )
        ax0.set_xlim( lmin, lmax )
        ax0.set_ylim( lmin, lmax )
        ax0.set_zlim( lmin, lmax )
        LMIN = np.ones_like(X)*lmin
        LMAX = np.ones_like(X)*lmax
        ax0.plot(X, Y, LMIN, c='gray', lw=2,alpha=0.3)
        ax0.plot(LMIN, Y, Z, c='gray', lw=2,alpha=0.3)
        ax0.plot(X, LMAX, Z, c='gray', lw=2,alpha=0.3)
        ax0.set_yticklabels([])
        ax0.set_xticklabels([])
        ax0.set_zticklabels([])
        ax0.text2D(0.05, 0.87, kwargs.get('ax0label',None),transform=ax0.transAxes)
        
        # Evaluate function at points and make plot
        EVAL = self.Bands(X,Y,Z, **kwargs)
        # EVAL can be of various types, handled below
        Emin =[]; Emax=[]
        for p in EVAL:
            if isinstance(p,dict):
                if 'y' in p.keys():
                    whichax = p.get('axis',1)
                    axp = ax2 if whichax ==2 else ax1
                    porder = p.get('zorder',2)
                    labelstr = p.get('label',None)
                    
                    fill = p.get('fill', False)
                    if fill:
                        ydat = p.get('y',None)
                        if ydat is not None:
                            axp.plot(t,ydat[0],
                                     lw=p.get('lw',2.),\
                                     color=p.get('color','black'),\
                                     alpha=p.get('fillalpha',0.5),\
                                     zorder=porder,\
                                     label=labelstr
                                     )
                            axp.fill_between( t, ydat[0], ydat[1],\
                                              lw=p.get('lw',2.),\
                                              color=p.get('color','black'),\
                                              facecolor=p.get('fillcolor','gray'),\
                                              alpha=p.get('fillalpha',0.5),\
                                              zorder=porder
                                            ) 
                            Emin.append( min( ydat[0].min(), ydat[1].min() ))
                            Emax.append( max( ydat[0].max(), ydat[1].max() )) 
                    else:
                        ydat = p.get('y',None)
                        if ydat is not None:
                            axp.plot( t, ydat,\
                                      lw=p.get('lw',2.),\
                                      color=p.get('color','black'),\
                                      alpha=p.get('alpha',1.0),\
                                      zorder=porder,\
                                      label=labelstr
                                    )
                            Emin.append( ydat.min() ) 
                            Emax.append( ydat.max() ) 
                elif 'figprop' in p.keys():
                    figsuptitle = p.get('figsuptitle',  None)
                    figGS.suptitle(figsuptitle, y=kwargs.get('suptitleY',1.0),fontsize=14) 
                    figGS.text(0.5,kwargs.get('foottextY',1.0),p.get('foottext',None),fontsize=14,
                               ha='center') 
                      
            else:
                ax1.plot(t,p); Emin.append(p.min()); Emax.append(p.max())
            
            
        Emin = min(Emin); Emax=max(Emax)
        dE = Emax-Emin
        
        ax1.grid()
        ax1.set_xlabel('$\mu\mathrm{m}$', fontsize=14)
        ax1.set_ylabel( self.unitlabel, rotation=0, fontsize=14, labelpad=-5 )
        
        # Finalize figure
        ax2.xaxis.set_major_locator( matplotlib.ticker.MultipleLocator(20) ) 
        ax2.xaxis.set_minor_locator( matplotlib.ticker.MultipleLocator(10) ) 
        ymin, ymax, ax1, ax2 =  Emin-0.05*dE, Emax+0.05*dE, ax1, ax2
    
        Ymin.append(ymin); Ymax.append(ymax); Ax1.append(ax1); Ax2.append(ax2)

        Ymin = min(Ymin); Ymax = max(Ymax)
        for ax in Ax1:
            ax.set_ylim( Ymin, Ymax)
            
        Ax1[0].legend( bbox_to_anchor=(1.06,0.005), \
            loc='lower right', numpoints=1, labelspacing=0.2,\
             prop={'size':10}, handlelength=1.1, handletextpad=0.5 )
        Ax2[0].legend( bbox_to_anchor=(1.10,1.10), \
            loc='upper right', numpoints=1, labelspacing=0.2, \
             prop={'size':10}, handlelength=1.1, handletextpad=0.5 )
            
        gs3Line.tight_layout(figGS, rect=tightrect)
        return figGS

    def PlotBandsSimple( self, **kwargs):
        figGS = plt.figure(figsize=(5.2,3.6))
        gs3Line = matplotlib.gridspec.GridSpec(2,2,\
                     width_ratios=[1.6, 1.], height_ratios=[1.8,1],\
                     wspace=0.25)
        tightrect = [0.,0.00, 0.95, 0.82]

        Ax1 = []; 
        Ymin =[]; Ymax=[]


        line_direction  = kwargs.pop('line_direction', '111')
        direcs = { \
                   '100':(np.pi/2, 0.), \
                   '010':(np.pi/2, np.pi/2), \
                   '001':(0., np.pi), \
                   '111':(np.arctan(np.sqrt(2)), np.pi/4) } 
        labels = { \
                   '100':'$\mathbf{111}$', \
                   '010':'$\mathbf{111}$', \
                   '001':'$\mathbf{111}$', \
                   '111':'$\mathbf{111}$' } 

        kwargs['direc'] = direcs[ line_direction ] 
        kwargs['ax0label']= labels[ line_direction ]   

        #kwargs['direc'] = (np.arctan(np.sqrt(2)), np.pi/4) 
        #kwargs['ax0label']='$\mathbf{111}$'

        kwargs['suptitleY'] = 0.96
        kwargs['foottextY'] = 0.84

        t, X,Y,Z, lims = self.line_points( **kwargs ) 
       
        
        ax1 = figGS.add_subplot( gs3Line[0:3,0] )
        ax1.set_xlim( lims[0],lims[1])
        ax1.grid()
        ax1.grid(which='minor')
        ax1.set_xlabel('$\mu\mathrm{m}$', fontsize=16)
        ax1.set_ylabel( self.unitlabel, rotation=0, fontsize=16, labelpad=15 )
        ax1.xaxis.set_major_locator( matplotlib.ticker.MultipleLocator(40) ) 
        ax1.xaxis.set_minor_locator( matplotlib.ticker.MultipleLocator(20) ) 


        ax2 = figGS.add_subplot( gs3Line[0,1] )
        ax2.set_xlim( lims[0]/2.,lims[1]/2.)
        y0,y1 = ax2.get_ylim()
        ax2.set_ylim( y0 , y1 + (y1-y0)*1.1)
        ax2.grid()
        ax2.set_xlabel('$\mu\mathrm{m}$', fontsize=14, labelpad=0)
        #ax2.set_ylabel('$n$', rotation=0, fontsize=14, labelpad=11 )
        ax2.xaxis.set_major_locator( matplotlib.ticker.MultipleLocator(20) ) 
        ax2.xaxis.set_minor_locator( matplotlib.ticker.MultipleLocator(10) ) 
            
        exclude = ['$\mathrm{exc\ band}$', '$n-2d$', '$d$', '$s$' ]
        #exclude = ['$n-2d$', '$d$', '$s$' ]
            
        # Evaluate function at points and make plot
        EVAL = self.Bands(X,Y,Z, **kwargs)
        # EVAL can be of various types, handled below
        Emin =[]; Emax=[]
        for p in EVAL:
            if isinstance(p,dict):
                if 'y' in p.keys():
                    whichax = p.get('axis',1)
                    #if whichax == 2 : continue
                    axp = ax2 if whichax ==2 else ax1

                    labelstr = p.get('label',None)
                    if labelstr is not None:
                        if 'lattice' in labelstr:
                            labelstr = None
                    skip = False
                    for e in exclude:
                        try:
                            if e == labelstr: 
                                skip = True
                        except:
                            continue 
                    if skip: continue
 
                    porder = p.get('zorder',2)
                    
                    fill = p.get('fill', False)
                    if fill:
                        ydat = p.get('y',None)
                        if ydat is not None:
                            axp.plot(t,ydat[0],
                                     lw=p.get('lw',2.),\
                                     color=p.get('color','black'),\
                                     alpha=p.get('fillalpha',0.5),\
                                     zorder=porder,\
                                     label=labelstr
                                     )
                            axp.fill_between( t, ydat[0], ydat[1],\
                                              lw=p.get('lw',2.),\
                                              color=p.get('color','black'),\
                                              facecolor=p.get('fillcolor','gray'),\
                                              alpha=p.get('fillalpha',0.5),\
                                              zorder=porder
                                            ) 
                            Emin.append( min( ydat[0].min(), ydat[1].min() ))
                            Emax.append( max( ydat[0].max(), ydat[1].max() )) 
                    else:
                        ydat = p.get('y',None)
                        if ydat is not None:
                            axp.plot( t, ydat,\
                                      lw=p.get('lw',2.),\
                                      color=p.get('color','black'),\
                                      alpha=p.get('alpha',1.0),\
                                      zorder=porder,\
                                      label=labelstr
                                    )
                            Emin.append( ydat.min() ) 
                            Emax.append( ydat.max() ) 
                elif 'figprop' in p.keys():
                    figsuptitle = p.get('figsuptitle',  None)
                    figGS.suptitle(figsuptitle, y=kwargs.get('suptitleY',1.0),fontsize=14) 
                    figGS.text(0.5,kwargs.get('foottextY',1.0),p.get('foottext',None),fontsize=14,
                               ha='center') 
                      
            else:
                ax1.plot(t,p); Emin.append(p.min()); Emax.append(p.max())
            
        ax2.legend( bbox_to_anchor=(1.20,1.10), \
            loc='upper right', numpoints=1, labelspacing=0.2, \
             prop={'size':10}, handlelength=1.1, handletextpad=0.5 )
            
            
        Emin = min(Emin); Emax=max(Emax)
        dE = Emax-Emin
        
        
        # Finalize figure
        ymin, ymax =  Emin-0.05*dE, Emax+0.05*dE
    
        Ymin.append(ymin); Ymax.append(ymax); Ax1.append(ax1)

        Ymin = min(Ymin); Ymax = max(Ymax)
        for ax in Ax1:
            ax.set_ylim( Ymin, Ymax)
            
        Ax1[0].legend( bbox_to_anchor=(1.1,-0.2), \
            loc='lower left', numpoints=1, labelspacing=0.2,\
             prop={'size':11}, handlelength=1.1, handletextpad=0.5 )
            
        gs3Line.tight_layout(figGS, rect=tightrect)
        return figGS


if __name__ == '__main__':
    latt3d = simpleCubic( allGR=4., Natoms=3.5e5, a_s=650., )
    #latt3d.plot3Line()
    #latt3d.plotCross( normal=(0.,0.), lims0=(-50,50), lims1=(-50,50), npoints=240)
    #latt3d.plot3Line(func=latt3d.Bottom)
    #latt3d.plot3Line(func=latt3d.LatticeMod)
    #latt3d.plot3Line(func=latt3d.Bands, globalMu=0.23, a_s=300., lines='111')
    latt3d.column( ["density","doublons","entropy"] )
    latt3d.plot3Line(func=latt3d.Bands,  lines='111')

if __name__ == '__main__':
    scattlen = 589.
    latt3d = simpleCubic( allIR=7.0, allGR=3.85, Natoms=3.5e5, a_s=scattlen, Temperature=1.6)
    fig100 = latt3d.plot3Line(func=latt3d.Bands,  lines='100')
    fig100.savefig('Ut_Comp/100_as%04d.png'%(scattlen), dpi=200)
    
    




