
"""
This file provides the HTSE solution to the Hubbard model.  It gives only
the density, double occupancy and entropy
"""

import numpy as np

def htse_dens( T, t, mu, U, ignoreLowT=False ):
    Tt = np.array(T/t)
    nerror = np.sum( Tt < 1.6 )
    if nerror > 0 :
        msg = "HTSE ERROR: T/t < 1.6. =>  min(T/t) = %.2f"% Tt.min()
        print msg 
        if not ignoreLowT:
            raise ValueError(msg)
        

    z0  = 2.*np.exp(mu/T)  + np.exp(-U/T + 2.*mu/T)  + 1. 
    
    term1 = ( 2.*np.exp(mu/T) + 2.*np.exp(-U/T + 2.*mu/T) ) / z0 

    term2 = 6.*((t/T)**2.)*(-4.*np.exp(mu/T) - 4.*np.exp(-U/T + 2.*mu/T) ) * \
                           ( 2.*T*(1-np.exp(-U/t))*np.exp(2.*mu/T) / U \
                             + np.exp( mu/T ) + np.exp( -U/T + 3.*mu/T)) \
                      / z0**3.

    term3 = 6.*((t/T)**2.)*(4.*T*(1-np.exp(-U/T))*np.exp(2.*mu/T)/U \
                              + np.exp(mu/T) + 3.*np.exp( -U/T + 3.*mu/T) ) \
                      / z0**2.
        
    #print z0    
    #print term1
    #print term2
    #print term3 
    return  term1 + term2 + term3

def htse_doub( T, t, mu, U, ignoreLowT=False):
    Tt = np.array(T/t)
    nerror = np.sum( Tt < 1.6 )
    if nerror > 0 :
        msg = "HTSE ERROR: T/t < 1.6. =>  min(T/t) = %.2f"% Tt.min()
        print msg 
        if not ignoreLowT:
            raise ValueError(msg)
 
    z0  = 2.*np.exp(mu/T)  + np.exp(-U/T + 2.*mu/T)  + 1. 
    
    term1 = np.exp(-U/T + 2.*mu/T) / z0
 
    term2 = -6.*((t/T)**2.)*( -2.*(T**2.)*(1-np.exp(-U/T))\
                                  *np.exp(2.*mu/T) / (U**2.) \
                              + 2.*np.exp(-U/T+2.*mu/T)*T/U  \
                              - np.exp(-U/T+3.*mu/T) ) \
                 / z0**2. 

    term3 = -12.*((t/T)**2.)*( 2.*T*(1-np.exp(-U/T))*np.exp(2.*mu/T) / U \
                               + np.exp(mu/T) + np.exp(-U/T+3.*mu/T) ) \
                             *np.exp(-U/T+2.*mu/T) \
                 / z0**3. 
            
    
    #print z0    
    #print term1
    #print term2
    #print term3 
    return  term1 + term2 + term3


def htse_entr( T, t, mu, U, ignoreLowT=False ):
    Tt = np.array(T/t)
    nerror = np.sum( Tt < 1.6 )
    if nerror > 0 :
        msg = "HTSE ERROR: T/t < 1.6 =>  min(T/t) = %.2f"% Tt.min()
        print msg 
        if not ignoreLowT:
            raise ValueError(msg)
 
    z0  = 2.*np.exp(mu/T)  + np.exp(-U/T + 2.*mu/T)  + 1. 
    
    term0 = (  U*np.exp(-U/T+2.*mu/T) - 2.*mu*np.exp(mu/T) \
              - 2.*mu*np.exp(-U/T+2.*mu/T) ) / T / z0

    term1 = np.log( z0 ) 

    term2 = 6.*((t/T)**2.)*( -2.*U*np.exp(-U/T+2.*mu/T)/T \
                                 + 4.*mu*np.exp(mu/T)/T  \
                                 + 4.*mu*np.exp(-U/T+2.*mu/T)/T ) \
               * ( 2.*T*(1-np.exp(-U/T))*np.exp(2.*mu/T)/U  \
                   + np.exp(mu/T) + np.exp(-U/T+3.*mu/T) ) \
                         / z0**3. 

    term3 = 6.*((t/T)**2.)*( 2.*(1-np.exp(-U/T))*np.exp(2.*mu/T)*T/U \
                            -2.*np.exp(-U/T+2.*mu/T) \
                            -4.*mu*(1-np.exp(-U/T))*np.exp(2.*mu/T)/U \
                            +U*np.exp(-U/T+3.*mu/T)/T \
                            -mu*np.exp(mu/T)/T \
                            -3.*mu*np.exp(-U/T+3.*mu/T)/T ) \
                         / z0**2. 

    term4 = -6.*((t/T)**2.)*( 2.*T*(1-np.exp(-U/T))*np.exp(2.*mu/T)/U \
                             + np.exp(mu/T) + np.exp(-U/T+3.*mu/T) ) \
                         / z0**2. 
    
    #print z0   
    #print term0
    #print term1
    #print term2
    #print term3
    #print term4
    return  term0 + term1 + term2 + term3 + term4


if __name__ == "__main__":
    print htse_dens( 2.4, 1., 10., 20.)
    print htse_doub( 2.4, 1., 10., 20.)
    print htse_entr( 2.4, 1., 10., 20.)

