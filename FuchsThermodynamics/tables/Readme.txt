This directory contains raw data for the 3D Hubbard model as computed
with various methods, for several interactions. 

The observables are:

double occupancies (in files doubleocc_U?)
energies           (in files energy_U?)
entropies          (in files entropy_U?)
spincorrelations   (in files spincorrelation_U?)

The format is:
T - mu - density - error - quantity - error

Files obtained with 10th order series expansion are provided in .series. Files for DMFT in .dmft, those obtained with dca in .dca. The .dca results are extrapolated to the infinite system size limit.

The files ending with .dat contain a combination of the various results (extrapolated dca where available, otherwise dmft where available, otherwise series expansion data for 10th order HTSE)


The subdirectory cluster_data contains the observables as measured directly for each cluster size. 
The observables are:

densities          (in files cluster_data/density_U?_?.dat
double occupancies (in files cluster_data/doubleocc_U?_?.dat)
energies           (in files cluster_data/energy_U?_?.dat)
entropies          (in files cluster_data/entropy_U?_?.dat)
spincorrelations   (in files cluster_data/spincorrelation_U?_?.dat)

The second "?" in each filename denotes the cluster size. Possible values are: 1(DMFT), 18, 26, 36, 48, 56, and 64.  

The format is:

T - mu - quantity
