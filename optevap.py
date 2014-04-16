
import scubic 
import lda
from scipy.optimize import minimize_scalar
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from matplotlib import rc
rc('font',**{'family':'serif'})
rc('text', usetex=True)

 

def optimal( **kwargs ) :

    s0 = kwargs.get('s0', 7. ) 
    wL = kwargs.get('wL', 47. )
    wC = kwargs.get('wC', 40. )
    alpha = wL/wC
    a_s = kwargs.get('a_s', 650. )

    if 'Number' in kwargs.keys():  
        N0 = kwargs['Number']  

    def Npenalty( Num ):
        p = 4.
        if Num > N0: 
            return np.exp( (Num - N0)/1e5 )**p 
        else:
            return 1. 

    def penalty(x,p):
        """
        This function is used to penalyze  EtaF < 1 , which amounts to 
        spilling out along the lattice beams.
        """ 
        if x < 1.:
            return np.exp(-(x-1.))**p
        else:
            return x 
    
        #return np.piecewise(x, [x < 1., x >= 1.], \
        #           [lambda x: , lambda x: x])        

    def merit( g0 ) :
        try:
            pot = scubic.sc(allIR  = s0, \
                        allGR  = g0, \
                        allIRw = wL, \
                        allGRw = wC )

            lda0 = lda.lda(potential = pot, a_s=a_s, globalMu='halfMott')
    
            if 'Number' in kwargs.keys():
                return  penalty( lda0.EtaEvap, 4 ) * Npenalty( lda0.Number) 
            else:
                return  penalty( lda0.EtaEvap, 4 )
        except:
            #print "Fail at g0=%.2f"% g0 
            #raise
            return 1e4

    res = minimize_scalar( merit, bounds=(0., s0/(alpha**2.)), tol=4e-2, \
              method='bounded' )
    gOptimal =  res.x 

    print "%.2f"%gOptimal,

    potOpt = scubic.sc( allIR=s0, allGR=gOptimal, allIRw=wL, allGRw=wC ) 
    ldaOpt = lda.lda( potential = potOpt, a_s=a_s, globalMu='halfMott')  
    return [ gOptimal, ldaOpt.EtaEvap, ldaOpt.Number, \
             ldaOpt.Entropy/ldaOpt.Number ]


def plotOptimal( datfile, **kwargs ) : 
 
    results = np.loadtxt( datfile )
 
    optimal_alpha = 1.10
    x0=44.
    x1=80.
    y0=28.
    y1=71.
    
    def meshplot( ax, i, j, k, contours = None, dashed=None, base=1.):
        x = results[:,i]
        y = results[:,j]
        z = results[:,k]/base
        xi = np.linspace( x.min(), x.max(), 300)
        yi = np.linspace( y.min(), y.max(), 300)
        zq = matplotlib.mlab.griddata(x, y, z, xi,yi)
        im0 =ax.pcolormesh( xi, yi, zq , cmap = plt.get_cmap('rainbow'))
        plt.axes( ax)
        plt.colorbar(im0) 

        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)

        ax.axhline( 40., lw=3., color='lightgray', alpha=0.30)
        ax.axvline( 47., lw=3., color='lightgray', alpha=0.30)
        
        if contours is not None:
            c0 = ax.contour(xi, yi, zq, contours, linewidths = 0.5, colors = 'k')
            plt.clabel(c0, inline=1, fontsize=8)
        if dashed is not None:
            for m in dashed:
                x = np.linspace( x0, x1, 100)
                ax.plot( x, x/m, '--',lw=2, color='black', alpha=1.0)
        
    fig = plt.figure(figsize=(8.,6.5))
    gs  = matplotlib.gridspec.GridSpec(2,2, wspace=0.2, hspace=0.3,\
            left=0.07, right=0.98, bottom=0.08, top=0.94)
    
    
    eta_contours = kwargs.get('eta_contours', [1.87,2.8, 4.4, 6. ])
    ax  = fig.add_subplot( gs[0,0] )
    i=1; j=2; k=4
    meshplot( ax, i, j, k, contours = eta_contours, \
              dashed=[1., optimal_alpha] )     
    ax.set_title('$\eta_{F}$', fontsize=16)
    ax.set_xlabel('$\mathrm{Lattice\ beam\ waist}\ w_{L}\ (\mu\mathrm{m})$')
    ax.set_ylabel('$\mathrm{Compensation\ beam\ waist}\ w_{C}\ (\mu\mathrm{m})$')

    g0_contours = kwargs.get('g0_contours', [1.87,2.8, 4.4, 6. ])
    ax  = fig.add_subplot( gs[0,1] )
    i=1; j=2; k=3
    meshplot( ax, i, j, k, contours = g0_contours, \
              dashed=[1., optimal_alpha] )     
    ax.set_title('$g_{0}$', fontsize=16)
    ax.set_xlabel('$\mathrm{Lattice\ beam\ waist}\ w_{L}\ (\mu\mathrm{m})$')
    ax.set_ylabel('$\mathrm{Compensation\ beam\ waist}\ w_{C}\ (\mu\mathrm{m})$')
   
    sn_contours = kwargs.get('sn_contours', [1.2, 1.4, 2.4, 3.] ) 
    ax  = fig.add_subplot( gs[1,0] )
    i=1; j=2; k=6
    meshplot( ax, i, j, k, contours = sn_contours, \
              dashed=[1., optimal_alpha] )     
    ax.set_title('$S/N$', fontsize=16)
    ax.set_xlabel('$\mathrm{Lattice\ waist}\ w_{L}\ (\mu\mathrm{m})$')
    ax.set_ylabel('$\mathrm{Compensation\ waist}\ w_{C}\ (\mu\mathrm{m})$')
    
    num_contours = kwargs.get('num_contours', [1.0, 2.0, 3.4, 4.8, 5.8])
    ax  = fig.add_subplot( gs[1,1] )
    i=1; j=2;  k=5 
    meshplot( ax, i, j, k, contours = num_contours, \
              dashed=[1., optimal_alpha], base=1e5 )     
    ax.set_title('$N/10^{5}$', fontsize=16)
    ax.set_xlabel('$\mathrm{Lattice\ waist}\ w_{L}\ (\mu\mathrm{m})$')
    ax.set_ylabel('$\mathrm{Compensation\ waist}\ w_{C}\ (\mu\mathrm{m})$')
   
    return fig  
    


