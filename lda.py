import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from matplotlib import rc
rc('font',**{'family':'serif'})
rc('text', usetex=True)


from vec3 import vec3, cross
import scipy.constants as C 


"""
This file provides a way of calculating trap profiles in the local density
approximation.    It needs to have a way of calculating:
 
* local band structure
* local tunneling rate, t
* local onsite interactions, U

From these thre quantities it can go ahead an use the solutions to the
homogeneous Fermi-Hubbard (FH)  model to calculate the LDA. 

In the homogeenous FH problem the chemical potential and the zero of
energy are always specified with respect to some point in the local band
structure.  This point depends on how the Hamiltonian is written down:  

A.  Traditional hamiltonian.   

  i, j  :  lattice sites 
  <i,j> :  nearest neighbors 
  s     :  spin 
  su    :  spin-up
  sd    :  spin-down


      Kinetic energy = -t \sum_{s} \sum_{<i,j>} a_{i,s}^{\dagger} a_{j,s}   

       Onsite energy =  U \sum_{i}  n_{i,su} n_{i,sd}   

  Using the traditional hamiltonian half-filling occurrs at a chemical 
  potential  mu = U/2.  

  The zero of energy in the traditional hamiltonian is exactly midway through
  the lowest band of the U=0 hamiltonian.


B.  Half-filling hamiltonian

      Kinetic energy = -t \sum_{s} \sum_{<i,j>} a_{i,s}^{\dagger} a_{j,s}   

       Onsite energy =  U \sum_{i} ( n_{i,su} - 1/2 )( n_{i,sd} - 1/2 )  

  Using the half-filling hamiltonian half-filling occurrs at a chemical
  potential  mu = 0,  a convenient value.   

  The zero of energy in the half-filling hamiltonian is shifted by U/2  
  with respect to the zero in the traditional hamiltonian. 

....
Considerations for LDA
....


When doing the local density approximation (LDA) we will essentially have a
homogenous FH model that is shifted in energy by the enveloping potential of
the trap and by the local band structure.  In the LDA the zero of energy  is
defined as the energy of an atom at a point where there are no external
potentials.   A global chemical potential will be defined with respect to the
LDA zero of energy.  

To calculate the local thermodynamic quantities, such as density, entropy,
double occupancy, etc.  we will use theoretical results for a homogeneous FH
model.  The local chemical potential will be determined based on the local
value of the enveloping potential and the local band structure (which can be
obtained from the local lattice depth).   

""" 

import udipole
from mpl_toolkits.mplot3d import axes3d

from scipy import integrate
from scipy import optimize
from scipy.interpolate import interp1d


# Load up the HTSE solutions 
import htse  


#...............
# LDA CLASS 
#...............

class lda:
    """ 
    This class provides the machinery to do the lda.  It provides a way to 
    determine the global chemical potential for a given number or for a half
    filled sample.  
    """ 
 
    def __init__( self, **kwargs ): 
        self.verbose = kwargs.get('verbose', False)
  
        # Flag to ignore errors related to the slope of the density profile
        # or the slope of the band bottom 
        self.ignoreSlopeErrors = kwargs.get( 'ignoreSlopeErrors',False)

        # Flag to ignore errors related to the global chemical potential
        # spilling into the beams 
        self.ignoreMuThreshold = kwargs.get('ignoreMuThreshold', False )

        # The potential needs to offer a way of calculating the local band 
        # band structure via provided functions.  The following functions
        # and variables must exist:
        # 
        #  To calculate lda: 
        #  -  pot.l 
        #  -  pot.bandStructure( X,Y,Z )
        #
        #  To make plots  
        #  -  pot.unitlabel
        #  -  pot.Bottom( X,Y,Z )
        #  -  pot.LatticeMod( X,Y,Z )
        #  -  pot.Info() 
        #  -  pot.EffAlpha()
        #  -  pot.firstExcited( X,Y,Z )
        #  -  pot.S0( X,Y,Z )
 
        self.pot = kwargs.pop( 'potential', None) 
        if self.pot is None: 
            raise ValueError(\
                    'A potential needs to be defined to carry out the LDA')  
        # The potential also contains the lattice wavelength, which defines
        # the lattice spacing 
        self.a = self.pot.l / 2. 
 


        # Initialize temperature.  Temperature is specified in units of 
        # Er.  For a 7 Er lattice  t = 0.04 Er 
        self.T = kwargs.get('Temperature', 0.10 ) 
        # Initialize interactions.
        self.a_s = kwargs.get('a_s',300.)
 
        # Make a cut line along 111 to calculate integrals of the
        # thermodynamic quantities
        direc111 = (np.arctan(np.sqrt(2)), np.pi/4)
        unit = vec3(); th = direc111[0]; ph = direc111[1] 
        unit.set_spherical( 1., th, ph); 
        t111, self.X111, self.Y111, self.Z111, lims = \
            udipole.linecut_points( direc=direc111)
        # Below we get the signed distance from the origin
        self.r111 =  self.X111*unit[0] + self.Y111*unit[1] + self.Z111*unit[2]

 
        # Obtain band structure and interactions along the 111 direction
        bandbot_111, bandtop_111,  \
        self.Ezero_111, self.tunneling_111, self.onsite_t_111 = \
            self.pot.bandStructure( self.X111, self.Y111, self.Z111)

        # The onsite interactions are scaled up by the scattering length
        self.onsite_t_111 = self.a_s * self.onsite_t_111
        self.onsite_111 = self.onsite_t_111 * self.tunneling_111

        # Lowst value of E0 is obtained 
        self.LowestE0 = np.amin( bandbot_111 )  

        self.Ezero0_111 = self.Ezero_111.min()

        #---------------------
        # CHECK FOR NO BUMP IN BAND BOTTOM 
        #---------------------
        # Calculate first derivative of the band bottom at small radii, to 
        # assess whether or not the potential is a valid potential 
        # (no bum in the center due to compensation )
        positive_r =  np.logical_and( self.r111  > 0. ,  self.r111 < 10. ) 
        # absolute energy of the lowest band, elb
        elb = bandbot_111[ positive_r ]  
        elb_slope = np.diff( elb ) < -1e-4
        n_elb_slope = np.sum( elb_slope )
        if n_elb_slope > 0:
            msg = "ERROR: Bottom of the band has a negative slope"
            if self.verbose:
                print msg
                print elb
                print np.diff(elb) 
                print elb_slope
            if not self.ignoreSlopeErrors:  
                raise ValueError(msg) 
        else: 
            if self.verbose:
                print "OK: Bottom of the band has positive slope up to r111 = 10 um"

        #------------------------------
        # SET GLOBAL CHEMICAL POTENTIAL 
        #------------------------------
        # Initialize global chemical potential and atom number
        # globalMu can be given directly or can be specified via the 
        # number of atoms.  If the Natoms is specified we calculate 
        # the required gMu using this function: 
        if 'globalMu' in kwargs.keys(): 
            # globalMu is given in Er, and is measured from the value
            # of Ezero at the center of the potential
            # When using it in the phase diagram it has to be changed to
            # units of the tunneling
            self.globalMu = kwargs.get('globalMu', 0.15)
            if  self.globalMu == 'halfMott':
                self.globalMu = self.onsite_111.max()/2.
        else :
            self.Number = kwargs.get('Natoms', 3e5)
            fN = lambda x : self.getNumber(x)- self.Number
            if self.verbose:
                print "Searching for globalMu => N=%.0f, "% self.Number,
            muBrent = kwargs.get('muBrent', (-1, 6.))
            self.globalMu, brentResults = \
                optimize.brentq(fN, muBrent[0], muBrent[1], \
                                xtol=1e-2, rtol=2e4, full_output=True) 
            if self.verbose:
                print "gMu=%.3f, " % brentResults.root,
                print "n_iter=%d, " % brentResults.iterations,
                print "n eval=%d, " % brentResults.function_calls,
                print "converge?=", brentResults.converged

        #---------------------
        # EVAPORATION ENERGIES
        #---------------------
        # Calculate energies to estimate eta parameter for evaporation
        self.globalMuZ0 = self.Ezero0_111 + self.globalMu

        # Make a cut line along 100 to calculate the threshold for evaporation
        direc100 = (np.pi/2, 0.) 
        t100, self.X100, self.Y100, self.Z100, lims = \
            udipole.linecut_points( direc=direc100, extents = 200.)

        # Obtain band structure along the 100 direction
        bandbot_100, bandtop_100,  self.Ezero_100, self.tunneling_100 = \
            self.pot.bandStructure( self.X100, self.Y100, self.Z100, \
                getonsite=False)
        self.Ezero0_100 = self.Ezero_100.min()

        # evapTH0_100 accounts for situations in which there is a local barrier 
        # as you move along 100 to the edge 
        self.evapTH0_100 = bandbot_100.max()

        # Once past the local barrier we calculate the bandbot energy along 
        # a beam
        self.beamBOT_100 = bandbot_100[-1]

        #------------------------------------------------
        # CONTROL THE CHEMICAL POTENTIAL SO THAT IT STAYS 
        # BELOW THE THRESHOLD FOR EVAPORATION
        #------------------------------------------------
        # For a valid scenario we need 
        #   self.globalMuZ0 < self.beamBOT_100
        #   self.globalMuZ0 < self.evapTH0_100  
        # Otherwise the density distribution will spill out into the beams
        # and the assumption of spherical symmetry won't be valid.
        if self.globalMuZ0 > self.evapTH0_100:
            msg = "ERROR: Chemical potential exceeds the evaporation threshold "
            if self.verbose:
                print msg
            if not self.ignoreMuThreshold : 
                raise ValueError(msg) 
        elif self.verbose:
            print "OK: Chemical potential is below evaporation threshold."

        if self.globalMuZ0 > self.beamBOT_100:
            msg = "ERROR: Chemical potential exceeds the bottom of the band " +\
                  "along 100"
            if self.verbose:
                print msg
            if not self.ignoreMuThreshold : 
                raise ValueError(msg) 
        elif self.verbose:
            print "OK: Chemical potential is below the bottom of the band " +\
                  "along 100"


        #-----------------------
        # ESTIMATION OF ETA EVAP
        #-----------------------
        mu = self.globalMuZ0 - self.LowestE0 
        th = self.evapTH0_100 - self.LowestE0
        self.EtaEvap = th/mu
        if False: 
            print "mu global = %.3g" % self.globalMuZ0 
            print "evap th   = %.3g" % self.evapTH0_100
            print "lowest E  = %.3g" % self.LowestE0
            print "mu = %.3g" % mu
            print "th = %.3g" % th
            print "eta = %.3g" % (th/mu)
            print "th-mu = %.3g" % (th-mu)

     
    
        
        # After the chemical potential is established the local chemical
        # potential along 111 can be defined.  It is used to calculate other
        # thermodynamic quantities
        gMuZero = self.Ezero0_111 + self.globalMu
        self.localMu_t_111= (gMuZero - self.Ezero_111) / self.tunneling_111

        # Obtain trap integrated values of the thermodynamic quantities
        self.Number  = self.getNumber( self.globalMu )
        self.NumberD = self.getNumberD()
        self.Entropy = self.getEntropy()

 

             

    def Info( self ):
        """
        Returns a latex string with the information pertinent to the 
        hubbard parameters
        """
        # Scattering length
        aslabel = '$a_{s}=%.0fa_{0}$' % self.a_s 
        # U/t label 
        Utlabel = '$U/t=%.1f$' % self.onsite_t_111.max()
        # Temperature label
        Tlabel = '$T/t=%.1f$' % self.T

        LDAlabel = '\n'.join( [aslabel, Utlabel, Tlabel ] ) 
        return LDAlabel    

    def ThermoInfo( self ):
        """
        Returns a latex string with the information pertinent to the 
        calculated  thermodynamic quantities. 
        """ 
        Nlabel = r'$N=%.2f\times 10^{5}$' % (self.Number/1e5)
        Dlabel = r'$D=%.3f$' % ( self.NumberD / self.Number )
        Slabel = r'$S/N=%.2fk_{\mathrm{B}}$' % ( self.Entropy / self.Number )
        return '\n'.join([Nlabel, Dlabel, Slabel]) 
    
    def TrapFreqs( self ): 
        """
        This function calculates the effective harmonic trapping frequencies
        of the potential.  It fits the bottom of the lowest band to a second
        degree polynomial. 
        """

        # Fit the first +/- 15 um of the band bottom to a second order
        # polynomial 
        direc100 = (np.pi/2., 0.) 
        direc010 = (np.pi/2., np.pi/2.) 
        direc001 = (0., np.pi) 
        nu = []
        for d in [direc100, direc010, direc001]: 
            # Make a cut line along d
            td, Xd, Yd, Zd, limsd = \
                udipole.linecut_points( direc=d, extents = 15.)

            # Obtain band structure along the d direction
            bandbot_d, bandtop_d,  Ezero_d, tunneling_d = \
                self.pot.bandStructure( Xd, Yd, Zd, \
                    getonsite=False)

            # Fit with poly
            z = np.polyfit( td, bandbot_d , 2 )
            c2 = z[0]  

            # A factor equal to h/(m*lambda)  comes out in front 
            # here we use lengths in um and freqs in Hz.  
            factor =  C.h  / C.physical_constants['atomic mass constant'][0] \
                      * 1e12  \
                     / self.pot.m  / self.pot.l  

            nu.append( (1./2./np.pi) * np.sqrt(c2) * factor )
        print nu 
    

    def getNumber( self, gMu, verbose=False):
        """ 
        This function calculates and returns the total number of atoms.  
        It integrates along 111 assuming a spherically symmetric sample. 

        Parameters 
        ----------
        gMu         :  global chemical potential
 
        """
        gMuZero = self.Ezero0_111 + gMu
        localMu = gMuZero - self.Ezero_111
        localMu_t = localMu / self.tunneling_111
        density = htse_dens( self.T, self.tunneling_111, localMu, self.onsite_111)
        density = self.fHdens( self.onsite_t_111, localMu_t )


        # Under some circumnstances the green compensation can 
        # cause dips in the density profile.  This can happen only 
        # if the green beam waist is smaller than the IR beam waist 
        # Experimentally we have seen that we do not handle these very
        # well, so we want to avoid them at all cost 
        # The occurence of this is flagged by a change in the derivative
        # of the radial density.  This derivative should always be negative. 

        # If the green beam waist is larger than the IR beam waist, then 
        # the problem with the non-monotonic density can also be found
        # when trying to push the compensation such that muGlobal gets 
        # close to the evaporation threshold 
        # This can be pathological because it leads to an accumulation of atoms
        # that are not trapped but this lda code integrates over them and counts
        # them anyways.  
        
        # To avoid any of the two situations desribed above we force the
        # density to decrease monotonically over the extent of our calculation. 

        # If the density slope is positive the we report it as an error 
        # 
        # find the point at which the density changes derivative
        radius_check = 1e-3 
        posradii = self.r111 > radius_check 
        posdens =  density[ posradii ]
        neg_slope = np.diff( posdens ) > 1e-4
        n_neg_slope = np.sum( neg_slope )


        if n_neg_slope > 0:  
            msg = "ERROR: Radial density profile along 111 has a positive slope"
            if self.verbose:
                print msg
                print "\n\nradius check start = ", radius_check
                print posdens
                print np.diff( posdens ) > 1e-4
            if not self.ignoreSlopeErrors:
                raise ValueError(msg)
        elif self.verbose:
            print "OK: Radial density profile along 111 decreases " + \
                  "monotonically."

        if False:
            print " posdens len = ",len(posdens)
            print " n_neg_slope = ",n_neg_slope
         
        dens = density[~np.isnan(density)]
        r = self.r111[~np.isnan(density)]
        return np.power(self.a,-3)*2*np.pi*integrate.simps(dens*(r**2),r)

    def getNumberD( self):
        """ 
        This function calculates and returns the total number of doublons. 
        It integrates along 111 assuming a spherically symmetric sample. 

        """
        doublons = self.fHdoub( self.onsite_t_111, self.localMu_t_111 ) 
        doub = doublons[~np.isnan(doublons)]
        r = self.r111[~np.isnan(doublons)]
        return 2.*np.power(self.a,-3)*2*np.pi*integrate.simps(doub*(r**2),r)
    
    def getEntropy( self):
        """ 
        This function calculates and returns the total entropy.  
        It integrates along 111 assuming a spherically symmetric sample. 

        """
        entropy  = self.fHentr( self.onsite_t_111, self.localMu_t_111 ) 
        entr = entropy[~np.isnan(entropy)]
        r = self.r111[~np.isnan(entropy)]
        return np.power(self.a,-3)*2*np.pi*integrate.simps(entr*(r**2),r)



def plotLine(  lda0, **kwargs):

    figGS = plt.figure(figsize=(6.0,4.2))
    gs3Line = matplotlib.gridspec.GridSpec(2,2,\
                 width_ratios=[1.6, 1.], height_ratios=[2.0,1],\
                 wspace=0.25)
    tightrect = [0.,0.00, 0.95, 0.88]

    Ax1 = []; 
    Ymin =[]; Ymax=[]

    line_direction  = kwargs.pop('line_direction', '111')
    direcs = { \
               '100':(np.pi/2, 0.), \
               '010':(np.pi/2, np.pi/2), \
               '001':(0., np.pi), \
               '111':(np.arctan(np.sqrt(2)), np.pi/4) } 
    labels = { \
               '100':'$(\mathbf{100})$', \
               '010':'$(\mathbf{010})$', \
               '001':'$(\mathbf{001})$', \
               '111':'$(\mathbf{111})$' } 

    cutkwargs = kwargs.pop( 'cutkwargs', {} ) 
    cutkwargs['direc'] = direcs[ line_direction ] 
    cutkwargs['ax0label']= labels[ line_direction ]   
    t, X,Y,Z, lims = udipole.linecut_points( **cutkwargs ) 


    kwargs['suptitleY'] = 0.96
    kwargs['foottextY'] = 0.84
   
    
    ax1 = figGS.add_subplot( gs3Line[0:3,0] )
    ax1.set_xlim( lims[0],lims[1] )
    ax1.grid()
    ax1.grid(which='minor')
    ax1.set_xlabel('$\mu\mathrm{m}$ '+cutkwargs['ax0label'], fontsize=16)
    ax1.set_ylabel( lda0.pot.unitlabel, rotation=0, fontsize=16, labelpad=15 )
    ax1.xaxis.set_major_locator( matplotlib.ticker.MultipleLocator(40) ) 
    ax1.xaxis.set_minor_locator( matplotlib.ticker.MultipleLocator(20) ) 

    ax1.yaxis.set_major_locator( matplotlib.ticker.MaxNLocator(7) ) 
    ax1.yaxis.set_minor_locator( matplotlib.ticker.MultipleLocator(1.) ) 

    ax2 = figGS.add_subplot( gs3Line[0,1] )
    ax2.set_xlim( lims[0]/2.,lims[1]/2.)
    ax2.grid()
    ax2.set_xlabel('$\mu\mathrm{m}$', fontsize=14, labelpad=0)
    #ax2.set_ylabel('$n$', rotation=0, fontsize=14, labelpad=11 )
    ax2.xaxis.set_major_locator( matplotlib.ticker.MultipleLocator(20) ) 
    ax2.xaxis.set_minor_locator( matplotlib.ticker.MultipleLocator(10) ) 
        

    #----------------------------------
    # CALCULATE ALL RELEVANT QUANTITIES
    #----------------------------------
    # All the relevant lines are first calculated here 
    bandbot_XYZ, bandtop_XYZ,  \
    Ezero_XYZ, tunneling_XYZ, onsite_t_XYZ = \
        lda0.pot.bandStructure( X, Y, Z ) 
    # The onsite interactions are scaled up by the scattering length
    onsite_t_XYZ = lda0.a_s * onsite_t_XYZ

    onsite_XYZ = onsite_t_XYZ * tunneling_XYZ
    Ezero0_XYZ = Ezero_XYZ.min()

    bottom = lda0.pot.Bottom( X, Y, Z ) 
    lattmod = lda0.pot.LatticeMod( X, Y, Z ) 

    excbot_XYZ, exctop_XYZ = lda0.pot.firstExcited( X, Y, Z ) 

    # Offset the chemical potential for use in the phase diagram
    localMu_t_XYZ =  ( lda0.globalMu + lda0.Ezero0_111 - Ezero_XYZ ) \
                      /  tunneling_XYZ

    # Obtain the thermodynamic quantities
    density_XYZ = lda0.fHdens( onsite_t_XYZ, localMu_t_XYZ ) 
    doublon_XYZ = lda0.fHdoub( onsite_t_XYZ, localMu_t_XYZ ) 
    entropy_XYZ = lda0.fHentr( onsite_t_XYZ, localMu_t_XYZ ) 


     
    #--------------------------
    # SETUP LINES TO BE PLOTTED 
    #--------------------------
    # A list of lines to plot is generated 
    # Higher zorder puts stuff in front
    toplot = [ 
             {'y':(bandbot_XYZ, Ezero_XYZ ), 'color':'blue', 'lw':2., \
              'fill':True, 'fillcolor':'blue', 'fillalpha':0.75,\
               'zorder':10, 'label':'$\mathrm{band\ lower\ half}$'},
             
             {'y':(Ezero_XYZ + onsite_XYZ, bandtop_XYZ + onsite_XYZ), \
              'color':'purple', 'lw':2., \
              'fill':True, 'fillcolor':'plum', 'fillalpha':0.75,\
              'zorder':10, 'label':'$\mathrm{band\ upper\ half}+U$'},
              
             {'y':(excbot_XYZ, exctop_XYZ ), 'color':'red', 'lw':2., \
              'fill':True, 'fillcolor':'pink', 'fillalpha':0.75,\
               'zorder':2, 'label':'$\mathrm{first\ excited\ band}$'},
             
             {'y':np.ones_like(X)*lda0.globalMuZ0, 'color':'limegreen',\
              'lw':2,'zorder':1.9, 'label':'$\mu_{0}$'},

             {'y':np.ones_like(X)*lda0.evapTH0_100, 'color':'#FF6F00', \
              'lw':2,'zorder':1.9, 'label':'$\mathrm{evap\ threshold}$'},
             
             {'y':bottom,'color':'gray', 'lw':0.5,'alpha':0.5},
             {'y':lattmod,'color':'gray', 'lw':1.5,'alpha':0.5,\
              'label':r'$\mathrm{lattice\ potential\ \ }\lambda\times10$'} \
             ]  

    entropy_per_particle = kwargs.pop('entropy_per_particle', False)
    if entropy_per_particle:
        toplot = toplot + [                 
             {'y':entropy_XYZ/density_XYZ,  'color':'black', 'lw':1.75, \
              'axis':2, 'label':'$s_{N}$'} ] 
    else:
        toplot = toplot + [
             {'y':density_XYZ, 'color':'blue', 'lw':1.75, \
              'axis':2, 'label':'$n$'},

             {'y':doublon_XYZ, 'color':'red', 'lw':1.75, \
              'axis':2, 'label':'$d$'},

             {'y':entropy_XYZ, 'color':'black', 'lw':1.75, \
              'axis':2, 'label':'$s_{L}$'},

             #{'y':density-2*doublons,  'color':'green', 'lw':1.75, \
             # 'axis':2, 'label':'$n-2d$'},

             #{'y':self.localMu_t,  'color':'cyan', 'lw':1.75, \
             # 'axis':2, 'label':r'$\mu$'},

             ]

    lattlabel = '\n'.join(  list( lda0.pot.Info() ) + \
                            [lda0.pot.EffAlpha()+\
                             ', $\eta_{F}=%.2f$'%lda0.EtaEvap] )
    toplot = toplot + [ {'text':True, 'x': 0., 'y':1.02, 'tstring':lattlabel,
                         'ha':'left', 'va':'bottom'} ]

    toplot = toplot + [ {'text':True, 'x': 1.0, 'y':1.02, 'tstring':lda0.Info(),
                         'ha':'right', 'va':'bottom'} ]
 
    toplot = toplot + [ {'text':True, 'x': 0., 'y':1.02, \
                         'tstring':lda0.ThermoInfo(), \
                         'ha':'left', 'va':'bottom', 'axis':2} ] 

    #--------------------------
    # ITERATE AND PLOT  
    #--------------------------
        
    Emin =[]; Emax=[]
    for p in toplot:
        if not isinstance(p,dict):
            ax1.plot(t,p); Emin.append(p.min()); Emax.append(p.max())
        else:
            if 'text' in p.keys():
                whichax = p.get('axis',1)
                axp = ax2 if whichax ==2 else ax1

                tx = p.get('x', 0.)
                ty = p.get('y', 1.)
                ha = p.get('ha', 'left')
                va = p.get('va', 'center')
                tstring = p.get('tstring', 'empty') 

                axp.text( tx,ty, tstring, ha=ha, va=va,\
                    transform=axp.transAxes)
            

            elif 'figprop' in p.keys():
                figsuptitle = p.get('figsuptitle',  None)
                figGS.suptitle(figsuptitle, y=kwargs.get('suptitleY',1.0),\
                               fontsize=14)
 
                figGS.text(0.5,kwargs.get('foottextY',1.0),\
                           p.get('foottext',None),fontsize=14,\
                           ha='center') 

            elif 'y' in p.keys():
                whichax = p.get('axis',1)
                #if whichax == 2 : continue
                axp = ax2 if whichax ==2 else ax1

                labelstr = p.get('label',None)
                porder   = p.get('zorder',2)
                fill     = p.get('fill', False)
                ydat     = p.get('y',None)

                if ydat is None: continue
                if fill:
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
                    axp.plot( t, ydat,\
                              lw=p.get('lw',2.),\
                              color=p.get('color','black'),\
                              alpha=p.get('alpha',1.0),\
                              zorder=porder,\
                              label=labelstr
                            )
                    Emin.append( ydat.min() ) 
                    Emax.append( ydat.max() ) 
                  
        
    ax2.legend( bbox_to_anchor=(1.25,1.10), \
        loc='upper right', numpoints=1, labelspacing=0.2, \
         prop={'size':10}, handlelength=1.1, handletextpad=0.5 )
        
        
    Emin = min(Emin); Emax=max(Emax)
    dE = Emax-Emin
    
    
    # Finalize figure
    y0,y1 = ax2.get_ylim()
    ax2.set_ylim( y0 , y1 + (y1-y0)*0.1)


    ymin, ymax =  Emin-0.05*dE, Emax+0.05*dE

    Ymin.append(ymin); Ymax.append(ymax); Ax1.append(ax1)

    Ymin = min(Ymin); Ymax = max(Ymax)
    for ax in Ax1:
        ax.set_ylim( Ymin, Ymax)
  
    if 'ax1ylim' in kwargs.keys():
        ax1.set_ylim( *kwargs['ax1ylim'] ) 
    
        
    Ax1[0].legend( bbox_to_anchor=(1.1,-0.15), \
        loc='lower left', numpoints=1, labelspacing=0.2,\
         prop={'size':11}, handlelength=1.1, handletextpad=0.5 )
        
    gs3Line.tight_layout(figGS, rect=tightrect)
    return figGS



def plotMathy(  lda0, **kwargs):

    figGS = plt.figure(figsize=(5.6,4.2))
    gs3Line = matplotlib.gridspec.GridSpec(3,2,\
                 width_ratios=[1.6, 1.], height_ratios=[2.2,0.8,1.2],\
                 wspace=0.2, hspace=0.24, 
                 left = 0.13, right=0.95, bottom=0.14, top=0.84)
    #tightrect = [0.,0.00, 0.95, 0.88]

    Ax1 = []; 
    Ymin =[]; Ymax=[] 

    line_direction  = kwargs.pop('line_direction', '111')
    direcs = { \
               '100':(np.pi/2, 0.), \
               '010':(np.pi/2, np.pi/2), \
               '001':(0., np.pi), \
               '111':(np.arctan(np.sqrt(2)), np.pi/4) } 
    labels = { \
               '100':'$(\mathbf{100})$', \
               '010':'$(\mathbf{010})$', \
               '001':'$(\mathbf{001})$', \
               '111':'$(\mathbf{111})$' } 

    cutkwargs = kwargs.pop( 'cutkwargs', {} ) 
    cutkwargs['direc'] = direcs[ line_direction ] 
    cutkwargs['ax0label']= labels[ line_direction ]   
    t, X,Y,Z, lims = udipole.linecut_points( **cutkwargs ) 


   
    
    ax1 = figGS.add_subplot( gs3Line[0:2,0] )
    ax1.grid()
    ax1.grid(which='minor')
    ax1.set_ylabel( lda0.pot.unitlabel, rotation=0, fontsize=16, labelpad=15 )
    ax1.xaxis.set_major_locator( matplotlib.ticker.MaxNLocator(8) ) 
    #ax1.xaxis.set_minor_locator( matplotlib.ticker.MultipleLocator(20) ) 

    ax1.yaxis.set_major_locator( matplotlib.ticker.MaxNLocator(7) ) 
    #ax1.yaxis.set_minor_locator( matplotlib.ticker.MultipleLocator(1.) ) 

    ax2 = figGS.add_subplot( gs3Line[0,1] )
    ax2.grid()
    #ax2.set_ylabel('$n$', rotation=0, fontsize=14, labelpad=11 )
    ax2.xaxis.set_major_locator( matplotlib.ticker.MaxNLocator(8) ) 
    ax2.xaxis.set_minor_locator( matplotlib.ticker.MultipleLocator(10) )

    ax3 = figGS.add_subplot( gs3Line[2,0] ) 
    ax3.grid() 
        

    #----------------------------------
    # CALCULATE ALL RELEVANT QUANTITIES
    #----------------------------------
    # All the relevant lines are first calculated here
   
    # In the Mathy plot the x-axis is the local lattice depth
    s0_XYZ = lda0.pot.S0( X, Y, Z)[0] 
    ax1.set_xlim( s0_XYZ.min(), s0_XYZ.max() )
    ax2.set_xlim( s0_XYZ.min(), s0_XYZ.max() )
    ax3.set_xlabel('$s_{0}\,(E_{R}) $', fontsize=16)
    ax2.set_xlabel('$s_{0}\,(E_{R}) $', fontsize=14, labelpad=0)
 
    bandbot_XYZ, bandtop_XYZ,  \
    Ezero_XYZ, tunneling_XYZ, onsite_t_XYZ = \
        lda0.pot.bandStructure( X, Y, Z ) 
    # The onsite interactions are scaled up by the scattering length
    onsite_t_XYZ = lda0.a_s * onsite_t_XYZ

    onsite_XYZ = onsite_t_XYZ * tunneling_XYZ
    Ezero0_XYZ = Ezero_XYZ.min()

    bottom = lda0.pot.Bottom( X, Y, Z ) 
    lattmod = lda0.pot.LatticeMod( X, Y, Z ) 

    excbot_XYZ, exctop_XYZ = lda0.pot.firstExcited( X, Y, Z ) 

    # Offset the chemical potential for use in the phase diagram
    localMu_t_XYZ =  ( lda0.globalMu + lda0.Ezero0_111 - Ezero_XYZ ) \
                      /  tunneling_XYZ

    # Obtain the thermodynamic quantities
    density_XYZ = lda0.fHdens( onsite_t_XYZ, localMu_t_XYZ ) 
    doublon_XYZ = lda0.fHdoub( onsite_t_XYZ, localMu_t_XYZ ) 
    entropy_XYZ = lda0.fHentr( onsite_t_XYZ, localMu_t_XYZ ) 


     
    #--------------------------
    # SETUP LINES TO BE PLOTTED 
    #--------------------------
    # A list of lines to plot is generated 
    # Higher zorder puts stuff in front
    toplot = [ 
             {'y':(bandbot_XYZ, Ezero_XYZ ), 'color':'blue', 'lw':2., \
              'fill':True, 'fillcolor':'blue', 'fillalpha':0.5,\
               'zorder':10, 'label':'$\mathrm{band\ lower\ half}$'},
             
             {'y':(Ezero_XYZ + onsite_XYZ, bandtop_XYZ + onsite_XYZ), \
              'color':'purple', 'lw':2., \
              'fill':True, 'fillcolor':'plum', 'fillalpha':0.5,\
              'zorder':10, 'label':'$\mathrm{band\ upper\ half}+U$'},

             {'y':(Ezero_XYZ, Ezero_XYZ + onsite_XYZ), \
              'color':'black', 'lw':2., \
              'fill':True, 'fillcolor':'gray', 'fillalpha':0.85,\
              'zorder':10, 'label':'$\mathrm{mott\ gap}$'},
              
             #{'y':(excbot_XYZ, exctop_XYZ ), 'color':'red', 'lw':2., \
             # 'fill':True, 'fillcolor':'pink', 'fillalpha':0.75,\
             #  'zorder':2, 'label':'$\mathrm{first\ excited\ band}$'},
             
             {'y':np.ones_like(X)*lda0.globalMuZ0, 'color':'limegreen',\
              'lw':2,'zorder':1.9, 'label':'$\mu_{0}$'},

             {'y':np.ones_like(X)*lda0.evapTH0_100, 'color':'#FF6F00', \
              'lw':2,'zorder':1.9, 'label':'$\mathrm{evap\ threshold}$'},
             
             #{'y':bottom,'color':'gray', 'lw':0.5,'alpha':0.5, 'axis':3},
             {'y':lattmod-bottom,'color':'gray', 'lw':1.5,'alpha':0.5, \
              'axis':3,\
              'label':r'$\mathrm{lattice\ potential\ \ }\lambda\times10$'} \
             ]  

    entropy_per_particle = kwargs.pop('entropy_per_particle', False)
    if entropy_per_particle:
        toplot = toplot + [                 
             {'y':entropy_XYZ/density_XYZ,  'color':'black', 'lw':1.75, \
              'axis':2, 'label':'$s_{N}$'} ] 
    else:
        toplot = toplot + [
             {'y':density_XYZ, 'color':'blue', 'lw':1.75, \
              'axis':2, 'label':'$n$'},

             {'y':doublon_XYZ, 'color':'red', 'lw':1.75, \
              'axis':2, 'label':'$d$'},

             {'y':entropy_XYZ, 'color':'black', 'lw':1.75, \
              'axis':2, 'label':'$s_{L}$'},

             #{'y':density-2*doublons,  'color':'green', 'lw':1.75, \
             # 'axis':2, 'label':'$n-2d$'},

             #{'y':self.localMu_t,  'color':'cyan', 'lw':1.75, \
             # 'axis':2, 'label':r'$\mu$'},

             ]

    lattlabel = '\n'.join(  list( lda0.pot.Info() ) + \
                            [lda0.pot.EffAlpha()+\
                             ', $\eta_{F}=%.2f$'%lda0.EtaEvap] )
    toplot = toplot + [ {'text':True, 'x': 0., 'y':1.02, 'tstring':lattlabel,
                         'ha':'left', 'va':'bottom'} ]

    toplot = toplot + [ {'text':True, 'x': 1.0, 'y':1.02, 'tstring':lda0.Info(),
                         'ha':'right', 'va':'bottom'} ]
 
    toplot = toplot + [ {'text':True, 'x': 0., 'y':1.02, \
                         'tstring':lda0.ThermoInfo(), \
                         'ha':'left', 'va':'bottom', 'axis':2} ] 

    #--------------------------
    # ITERATE AND PLOT  
    #--------------------------
    kwargs['suptitleY'] = 0.96
    kwargs['foottextY'] = 0.84
     
    # For every plotted quantity I use only lthe positive radii  
    Emin =[]; Emax=[]
    positive = t > 0.
    xarray = s0_XYZ[ positive ] 
    for p in toplot:
        if not isinstance(p,dict):
            p = p[positive] 
            ax1.plot(xarray,p); Emin.append(p.min()); Emax.append(p.max())
        else:
            if 'text' in p.keys():
                whichax = p.get('axis',1)
                axp = ax2 if whichax ==2 else ax1

                tx = p.get('x', 0.)
                ty = p.get('y', 1.)
                ha = p.get('ha', 'left')
                va = p.get('va', 'center')
                tstring = p.get('tstring', 'empty') 

                axp.text( tx,ty, tstring, ha=ha, va=va,\
                    transform=axp.transAxes)
            

            elif 'figprop' in p.keys():
                figsuptitle = p.get('figsuptitle',  None)
                figGS.suptitle(figsuptitle, y=kwargs.get('suptitleY',1.0),\
                               fontsize=14)
 
                figGS.text(0.5,kwargs.get('foottextY',1.0),\
                           p.get('foottext',None),fontsize=14,\
                           ha='center') 

            elif 'y' in p.keys():
                whichax = p.get('axis',1)
                #if whichax == 2 : continue
                axp = ax2 if whichax ==2 else ax3 if  whichax == 3 else ax1

                labelstr = p.get('label',None)
                porder   = p.get('zorder',2)
                fill     = p.get('fill', False)
                ydat     = p.get('y',None)

                if ydat is None: continue

                if fill:
                    ydat = ( ydat[0][positive], ydat[1][positive] ) 
                    axp.plot(xarray,ydat[0],
                             lw=p.get('lw',2.),\
                             color=p.get('color','black'),\
                             alpha=p.get('fillalpha',0.5),\
                             zorder=porder,\
                             label=labelstr
                             )
                    axp.fill_between( xarray, ydat[0], ydat[1],\
                                      lw=p.get('lw',2.),\
                                      color=p.get('color','black'),\
                                      facecolor=p.get('fillcolor','gray'),\
                                      alpha=p.get('fillalpha',0.5),\
                                      zorder=porder
                                    )
                    if whichax == 1: 
                        Emin.append( min( ydat[0].min(), ydat[1].min() ))
                        Emax.append( max( ydat[0].max(), ydat[1].max() )) 
                else:
                    ydat = ydat[ positive ] 
                    axp.plot( xarray, ydat,\
                              lw=p.get('lw',2.),\
                              color=p.get('color','black'),\
                              alpha=p.get('alpha',1.0),\
                              zorder=porder,\
                              label=labelstr
                            )
                    if whichax == 1: 
                        Emin.append( ydat.min() ) 
                        Emax.append( ydat.max() ) 
                  
        
    ax2.legend( bbox_to_anchor=(0.03,1.02), \
        loc='upper left', numpoints=1, labelspacing=0.2, \
         prop={'size':10}, handlelength=1.1, handletextpad=0.5 )
        
        
    Emin = min(Emin); Emax=max(Emax)
    dE = Emax-Emin
    
    
    # Finalize figure
    y0,y1 = ax2.get_ylim()
    ax2.set_ylim( y0 , y1 + (y1-y0)*0.1)


    ymin, ymax =  Emin-0.05*dE, Emax+0.05*dE

    Ymin.append(ymin); Ymax.append(ymax); Ax1.append(ax1)

    Ymin = min(Ymin); Ymax = max(Ymax)
    for ax in Ax1:
        ax.set_ylim( Ymin, Ymax)

    if 'ax1ylim' in kwargs.keys():
        ax1.set_ylim( *kwargs['ax1ylim'] ) 
        
    Ax1[0].legend( bbox_to_anchor=(1.1,0.1), \
        loc='upper left', numpoints=1, labelspacing=0.2,\
         prop={'size':11}, handlelength=1.1, handletextpad=0.5 )
        
    #gs3Line.tight_layout(figGS, rect=tightrect)
    return figGS
 
