

import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.interpolate.interpnd import _ndim_coords_from_arrays
import matplotlib.mlab as ml


# Load interpolation data and make interpolations and plots 
    
def getHTSEPoints( Temperature, name="density"):
    try:
        HTSE = np.loadtxt('HTSEdat/HTSEinterp%03d.dat'%int(Temperature*10))
    except:
        saveHTSEInterp( Temperature )
        HTSE = np.loadtxt('HTSEdat/HTSEinterp%03d.dat'%int(Temperature*10))
    namedict = {"density":2, "densfluc":3, "doublons":4, "entropy":5}

    U_   = HTSE[:,0]
    mu_  = HTSE[:,1]
    qty_ = HTSE[:, namedict[name]]
    
    return U_, mu_, qty_

def getHTSEInterp( Temperature, name='density'):
    U_, mu_, qty_ = getHTSEPoints( Temperature, name=name)
    points = _ndim_coords_from_arrays((U_, mu_))
    interp = CloughTocher2DInterpolator(points, qty_)
    return lambda u,mu : interp( np.column_stack(( np.ravel(u),np.ravel(mu) )) ).reshape(u.shape)

def getHTSEInterp2( Temperature, name='density'):
    U_, mu_, qty_ = getHTSEPoints( Temperature, name=name)
    points = _ndim_coords_from_arrays((U_, mu_))
    return CloughTocher2DInterpolator(points, qty_)



doub = {}
entr = {}
enrg = {}
for U in [4, 6, 8, 10, 12]:
    doub[U] = np.loadtxt("FuchsThermodynamics/tables/doubleocc_U%d.dat"%U)
    entr[U] = np.loadtxt("FuchsThermodynamics/tables/entropy_U%d.dat"%U)
    enrg[U] = np.loadtxt("FuchsThermodynamics/tables/energy_U%d.dat"%U)
    
fuchs ={}
fuchs['doublons'] = doub
fuchs['entropy']  = entr
fuchs['energy']   = enrg


def getFuchsPoints( Temperature, name='density', Uset = [4,6,8,10,12]):
    if name == 'density':
        qtydict = fuchs['doublons']
        qd = {'mu':1, 'qty':2}
    else:
        qtydict = fuchs[name]
        qd = {'mu':1, 'qty':4}
    qty = []
    
    
    for U in Uset:
        if name == 'doublons':
            for row in qtydict[U]:
                if row[0] == Temperature:
                    if row[qd['mu']] == 0.:
                        half_fill_doublons = row[qd['qty']]
                    
        for row in qtydict[U]:
            if row[0] == Temperature:
                qty.append(np.array([ row[qd['mu']] + U/2. ,  float(U), row[qd['qty']] ]))
                
                # Mirror quantitites to get positive chemical potential
                # We use the particle-hole symmetry of the Hubbard hamiltonian
                # (http://quest.ucdavis.edu/tutorial/hubbard7.pdf)
                # The entropy, density, and local moment are symmetric about mu=U/2
                # The double occupancy is given by d = (n-m^2)/2  where 
                # n is the density and m^2 is the local moment.  
                if name == 'density':
                    mirrored = 2. - row[qd['qty']]
                if name == 'entropy':
                    mirrored = row[qd['qty']]
                    
                if name == 'doublons':
                    density = row[2]
                    mirroredDensity = 2. - density
                    localMoment = density - 2 * row[qd['qty']] 
                    mirrored = ( mirroredDensity - localMoment  ) / 2.
                qty.append(np.array([ -1.*row[qd['mu']] +U/2.,  float(U), mirrored ]))
                
    qty = np.array(qty)
    #Sort the points by col0, which is chemical potential
    idx = np.argsort(qty[:,0])
    mu_ = qty[ idx, 0]
    U_  = qty[ idx, 1]
    qty_= qty[ idx, 2]
    
    return U_, mu_, qty_

def getFuchsInterp( Temperature, name='density', Uset = [4,6,8,10,12]):
    U_, mu_, qty_ = getFuchsPoints( Temperature, name=name, Uset = Uset)
    points = _ndim_coords_from_arrays((U_, mu_))
    if len ( Uset) > 1:
        interp = CloughTocher2DInterpolator(points, qty_)
        return lambda u,mu : interp( np.column_stack(( np.ravel(u),np.ravel(mu) )) ).reshape(u.shape)
    else:
        return None

def getFuchsInterp2( Temperature, name='density', Uset = [4,6,8,10,12]):
    U_, mu_, qty_ = getFuchsPoints( Temperature, name=name, Uset = Uset)
    points = _ndim_coords_from_arrays((U_, mu_))
    if len ( Uset) > 1:
        return CloughTocher2DInterpolator(points, qty_)
    else:
        return None

