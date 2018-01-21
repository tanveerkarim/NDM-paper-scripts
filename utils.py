import numpy as np
import numpy.lib.recfunctions as rec
from scipy.ndimage.filters import gaussian_filter

from astropy.io import ascii, fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u

from os import listdir
from os.path import isfile, join

from scipy.stats import multivariate_normal
import scipy.stats as stats
from scipy.integrate import simps, trapz, cumtrapz
import scipy.optimize as opt
import matplotlib.pyplot as plt
import extreme_deconvolution as XD

import numba as nb

from matplotlib.patches import Ellipse

import numpy as np
from scipy.stats import norm, chi2

import time


# Matplot ticks
import matplotlib as mpl
mpl.rcParams['xtick.major.size'] = 15
mpl.rcParams['xtick.major.width'] = 1.
mpl.rcParams['ytick.major.size'] = 15
mpl.rcParams['ytick.major.width'] = 1.
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15

colors = ["orange", "grey", "purple", "red", "black", "blue"]
cnames = ["Gold", "Silver", "NoOII",  "NoZ", "Non-ELG", "DEEP2 unobserved"]

large_random_constant = -999119283571
deg2arcsec=3600


@nb.jit
def check_in_arr2(arr1, arr2):
    """
    Given two sorted integer arrays arr1, arr2, return a boolean vector of size arr1.size,
    where each element i indicate whether the value of arr1[i] is in arr2.
    """
    N_arr1 = arr1.size
    N_arr2 = arr2.size
    
    # Vector to return
    iselect = np.zeros(N_arr1, dtype=bool)
    
    # First, check whether elements from arr1 is within the range of arr2
    if (arr1[-1] < arr2[0]) or (arr1[0] > arr2[-1]):
        return iselect
    else: # Otherwise, for each element in arr2, incrementally search for in arr1 the same elements
        idx = 0
        for arr2_current_el in arr2:
            while arr1[idx] < arr2_current_el: # Keep incrementing arr1 idx until we reach arr2_current value.
                idx+=1
            if arr1[idx] == arr2_current_el:
                while arr1[idx] == arr2_current_el:
                    iselect[idx] = True
                    idx+=1
                
        return iselect


@nb.jit
def tally_objects(N_cell, cell_number, cw, FoM):
    """
    Given number of cells and cell number, completeness weight, and FoM per sample,
    return a tally.

    Note that the total number and FoM are weighted by completeness weight.

    Also, return the number of good objects defined as objects with positive FoM.
    """

    FoM_tally = np.zeros(N_cell, dtype = float)
    Ntotal_tally = np.zeros(N_cell, dtype = float)
    Ngood_tally = np.zeros(N_cell, dtype = float)

    for i, cn in enumerate(cell_number): # cn is cell number, which we can use as index.
#         print (i, cn, FoM[i], cw[i])
        if (cn>=0) and (cn<N_cell):
            FoM_tally[cn] += FoM[i] * cw[i]
            Ntotal_tally[cn] += cw[i]
            if FoM[i] > 0: 
                Ngood_tally[cn] += cw[i]

    return FoM_tally, Ntotal_tally, Ngood_tally



def compute_cell_number(bin_indicies, num_bins):
    """
    Give number of bins in each direction of a multi-dimensional grid, 
    return a cell number corresponding to a particular set of bin indices.
    
    bin_indicies is a list of numpy arrays [bin_num1, bin_num2, ...]
    
    num_bins is a list.
    """
    cell_num = np.zeros(bin_indicies[0].size, dtype=int)
    ND = len(num_bins)
    for i in range(ND):
        if i < ND-1:
            cell_num += bin_indicies[i] * np.multiply.reduce(num_bins[i+1:])
        else:
            cell_num += bin_indicies[i]
    
    return cell_num


@nb.jit
def tally_objects_kernel(N_cell, cell_number, cw, FoM, num_bins):
    """
    Given number of cells and cell number, completeness weight, and FoM per sample,
    return a tally. Use Gaussian kernel approximation for each particle in a cell.

    Strategy:
    - For each sample with a legitimate cell number, compute the bin number. 
    - Determine the cells in the neighborhood of the current cell, and increment
    according to the kernel look up table.

    Note that the total number and FoM are weighted by completeness weight.

    Also, return the number of good objects defined as objects with positive FoM.

    Assumes the grid is three dimensional.
    """
    # Constants
    N1 = np.multiply.reduce(num_bins[1:])
    N2 = np.multiply.reduce(num_bins[2:])

    N_kernel = 11
    gauss_kernel = gen_gauss_kernel_3D(N_kernel) # Normalized to sum to one.

    FoM_tally = np.zeros(N_cell, dtype = float)
    Ntotal_tally = np.zeros(N_cell, dtype = float)
    Ngood_tally = np.zeros(N_cell, dtype = float)


    for i, cn in enumerate(cell_number): # cn is cell number, which we can use as index.
        if (cn>=0) and (cn<N_cell):
            # Compute the bin number corresponding to the cell.
            bin_indicies = [cn//N1, (cn%N1) // N2,  cn % N2]

            # Extract common values
            cw_tmp = cw[i]
            FoM_tmp = FoM[i]

            # Iterate through the neighborhood of cells centered at the current cell cn,
            # increment the appropriate numbers.
            # Indicies for 0, 1, 2 directions: m, n, l
            for m in range(-N_kernel/2, N_kernel/2+1): # e.g., N_kernel=3 gives -1, 0, 1
                for n in range(-N_kernel/2, N_kernel/2+1): # e.g., N_kernel=3 gives -1, 0, 1
                    for l in range(-N_kernel/2, N_kernel/2+1): # e.g., N_kernel=3 gives -1, 0, 1
                        # Cell number computed 
                        cn_iter = (bin_indicies[0]+m)*N1 + (bin_indicies[1]+n)*N2 + (bin_indicies[2]+l)
                        gk_factor = gauss_kernel[m+N_kernel/2, n+N_kernel/2, l+N_kernel/2] # Gaussian kernel factor
                        if (cn_iter>=0) and (cn_iter<N_cell):
                            FoM_tally[cn_iter] += FoM_tmp * cw_tmp * gk_factor
                            Ntotal_tally[cn_iter] += cw_tmp * gk_factor
                            if FoM_tmp > 0: 
                                Ngood_tally[cn_iter] += cw_tmp * gk_factor

    return FoM_tally, Ntotal_tally, Ngood_tally



def multdim_grid_cell_number(samples, ND, limits, num_bins):
    """
    Given samples array, return the cell each sample belongs to, where the cell is an 
    element of a ND-dimensional grid defined by limits and numb_bins.
    
    More specifically, each cell can be identified by its bin indices.
    If there are three variables, v0, v1, v2, which have N0, N1, N2 number 
    of bins, and the cell corresponds to (n0, n1, n2)-th bin,
    then cell_number = (n0* N1 * N2) + (n1 * N2) + n2. 
    
    Note that we use zero indexing. If an object falls outside the binning range,
    it's assigned cell number "-1".
    
    INPUT: All arrays must be numpy arrays. 
        - samples, ND: List of linear numpy arrays [var1, var2, var3, ...] with size [Nsample].
        Cell numbers calculated based on the first ND variables. 
        - limits: List of limits for each dimension. 
        - num_bins: Number of bins to use for each dimension

    Output:
        - cell_number
    """
    Nsample = samples[0].size
    cell_number = np.zeros(Nsample, dtype=int)
    ibool = np.zeros(Nsample, dtype=bool) # For global correction afterwards.
    
    for i in range(ND): # For each variable to be considered.
        X = samples[i]
        Xmin, Xmax = limits[i]
        _, dX = np.linspace(Xmin, Xmax, num_bins[i]+1, endpoint=True, retstep=True)
        X_bin_idx = gen_bin_idx(X, Xmin, dX) # bin_idx of each sample
        if i < ND-1:
            cell_number += X_bin_idx * np.multiply.reduce(num_bins[i+1:])
        else:
            cell_number += X_bin_idx
        
        # Correction. If obj out of bound, assign -1.
        ibool = np.logical_or.reduce(((X_bin_idx < 0), (X_bin_idx >= num_bins[i]), ibool))
        
    cell_number[ibool] = -1
    
    return cell_number



def gen_bin_idx(X, Xmin, dX):
    """
    Given a linear array of numbers and minimum, 
    compute bin index corresponding to each sample.
    
    dX is spacing between the bins.
    """
    
    return np.floor((X-Xmin)/float(dX)).astype(int)


def return_file(fname):
    with open (fname, "r") as myfile:
        data=myfile.readlines()
    return data

def HMS2deg(ra=None, dec=None):
    rs, ds = 1, 1
    if dec is not None:
        D, M, S = [float(i) for i in dec.split(":")]
        if str(D)[0] == '-':
            ds, D = -1, abs(D)
        dec= D + (M/60) + (S/3600)

    if ra is not None:
        H, M, S = [float(i) for i in ra.split(":")]
        if str(H)[0] == '-':
            rs, H = -1, abs(H)
        ra = (H*15) + (M/4) + (S/240)

    if (ra is not None) and (dec is not None):
        return ra, dec 
    elif ra is not None: 
        return ra
    else:
        return dec
    
def MMT_study_color(grz, field, mask=None):
    """
    field:
    - 0 corresponds to 16hr
    - 1 corresponds to 23hr
    """
    g,r,z = grz
    if mask is not None:
        g = g[mask]
        r = r[mask]
        z = z[mask]
    if field == 0:
        return (g<24) & ((g-r)<0.8) & np.logical_or(((r-z)>(0.7*(g-r)+0.2)), (g-r)<0.2)
    else:
        return (g<24) & ((g-r)<0.8) & np.logical_or(((r-z)>(0.7*(g-r)+0.2)), (g-r)<0.2) & (g>20)
    
def MMT_DECaLS_quality(fits, mask=None):
    gany,rany,zany = load_grz_anymask(fits)
    givar, rivar, zivar = load_grz_invar(fits)
    bp = load_brick_primary(fits)
    if bp[0] == 0:
        bp = (bp==0)
    elif type(bp[0])==np.bool_:
        bp = bp # Do nothing    
    else:
        bp = bp=="T"
    r_dev, r_exp = load_shape(fits)
    
    if mask is not None:
        gany, rany, zany = gany[mask], rany[mask], zany[mask]
        givar, rivar, zivar =givar[mask], rivar[mask], zivar[mask]
        bp = bp[mask]
        r_dev, r_exp = r_dev[mask], r_exp[mask]
        
    return (gany==0)&(rany==0)&(zany==0)&(givar>0)&(rivar>0)&(zivar>0)&(bp)&(r_dev<1.5)&(r_exp<1.5)

def load_MMT_specdata(fname, fib_idx=None):
    """
    Given spHect* file address, return wavelength (x),
    flux value (d), inverse variance (divar), and
    AND_mask.

    If fib_idx is not None, then return only spectra
    indicated. 
    """
    table_spec = fits.open(fname)
    x = table_spec[0].data
    d = table_spec[1].data
    divar = table_spec[2].data # Inverse variance
    AND_mask = table_spec[3].data

    if fib_idx is not None:
        x = x[fib_idx,:]
        d = d[fib_idx,:]
        divar = divar[fib_idx,:]
        AND_mask = AND_mask[fib_idx,:]
    return x, d, divar, AND_mask

def box_car_avg(d,window_pixel_size=50,mask=None):
    """
    Take a running average of window_piexl_size pixels.
    Exclude masked pixels from averaging.
    """
    # Filter
    v = np.ones(window_pixel_size)
    
    # Array that tells how many pixels were used
    N_sample = np.ones(d.size)
    if mask is not None:
        N_sample[mask]=0
    N_sample = np.convolve(N_sample, v,mode="same")
    
    # Running sum of the data excluding masked pixels
    if mask is not None:
        d[mask]=0
    d_boxed = np.convolve(d, v,mode="same")
    
    # Taking average
    d_boxed /= N_sample
    
    return d_boxed



def process_spec(d, divar, width_guess, x_mean, mask=None):
    """
    Given the data vector d and its corresopnding inverse
    variance and pix_sigma width for the filter
    compute, integrated flux A, var(A), and chi sq.
    Also, return S2N. width_guess is in Angstrom.
    """
    # Filter
    pix_sigma = width_guess/x_mean # Peak width in terms of pixels
    filter_size = np.ceil(pix_sigma*4*2) # How large the filter has to be to encompass 4-sig
    # If filter size is odd, add one.
    if filter_size%2==0:
        filter_size+=1

    # Centered around the filter, create a gaussian.
    v_center = int(filter_size/2)
    v = np.arange(int(filter_size))-v_center
    v = np.exp(-(v**2)/(2*(pix_sigma**2)))/(pix_sigma*np.sqrt(2*np.pi))
    # Note: v = G(A=1)
    
    # If mask is used, then block out the appropriate
    # portion.
    if mask is not None:
        d[mask]=0
        divar[mask]=0
    
    # varA: Running sum of (ivar*v^2) excluding masked pixels
    varA = np.convolve(divar, v**2, mode="same")
    
    # A_numerator: Running sum of (d*v*ivar)
    A_numerator = np.convolve(d*divar, v, mode="same")
    A = A_numerator/varA
    
    # SN
    S2N = A/np.sqrt(varA)
    
    # Compute reduced chi. sq.
    # To do, compute the number of samples used.
    # Filter
    v_N = np.ones(int(filter_size))    
    N_sample = np.ones(d.size)
    if mask is not None:
        N_sample[mask]=0
    N_sample = np.convolve(N_sample, v_N, mode="same")    

    # Chi sq. # -1 since we are only estimating one parameter.
    chi = -(-2*A_numerator*A+varA*(A**2))/(N_sample-1) 
    
    return A, varA, chi, S2N

def median_filter(data, mask=None, window_pixel_size=50):
    """
    Given the data array and window size, compute median
    without mask.
    """
    array_length = data.size
    
    if window_pixel_size%2==0:
        window_pixel_size+=1
        
    if mask is None:
        mask = np.ones(array_length)
        
    pass_mask = np.logical_not(mask)

    ans = np.zeros(array_length)
    for i in range(array_length):
        idx_l = max(0, i-int(window_pixel_size/2))
        idx_h = min(array_length, i+int(window_pixel_size/2))+1
#         print(idx_l, idx_h)
        tmp = data[idx_l:idx_h][pass_mask[idx_l:idx_h]]
#         print(tmp.size)
        ans[i] = np.median(tmp)
    return ans    

def spec_lines():
    emissions = [3727.3, 4102.8, 4340, 4861.3, 4959,5006.8, 6562.8, 6716]
    absorptions  = [3933.7, 3968.6, 4304.4, 5175.3, 5984.0]
    return emissions, absorptions

def OII_wavelength():
    return 3727.3


def plot_fit(x, d, A, S2N, chi, threshold=5, mask=None, mask_caution=None, xmin=4500, xmax=8500, s=1,\
             plot_show=True, plot_save=False, save_dir=None, plot_title=""):
    """
    Plot a spectrum and its fits.
    """
    if mask is not None:
        S2N[mask] = 0
        A[mask] = 0
        chi[mask] = 0
        d[mask] = 0

    # Limit plot range
    ibool = (x>xmin)&(x<xmax)
    x_masked = x[ibool]
    S2N_masked = S2N[ibool]
    chi_masked = chi[ibool]
    A_masked = A[ibool]
    d_masked = d[ibool]
    if mask_caution is not None:
        mask_caution = mask_caution[ibool]

    # Emission and absorption lines
    emissions, absorptions = spec_lines()
    OII_line = OII_wavelength()
    
    # Find peaks in S2N. Must have 5-sigma.
    isig5 = (S2N_masked>threshold)
    # Create a vector that tells where a peak cluster starts and end. 
    S2N_start_end = np.zeros_like(S2N_masked)
    S2N_start_end[isig5] = 1
    S2N_start_end[1:] = S2N_start_end[1:]-S2N_start_end[:-1]
    S2N_start_end[0] = 0
    # For each [1,...,-1] cluster, finding the idx of maximum and find
    # the corresponding x value.
    starts = np.where(S2N_start_end==1)[0]
    ends = np.where(S2N_start_end==-1)[0]
    z_peak_list = []
    s2n_peak_list = []
    oii_flux_list = []
    for i in range(len(ends)):
        start = starts[i]
        end = ends[i]
        if (start==(end-1)) or (start==end):
            val = S2N_masked[start]
        else:
            val = np.max(S2N_masked[start:end])
        idx = np.where(S2N_masked ==val)[0]
        z_peak_list.append(x_masked[idx]/OII_line-1)
        s2n_peak_list.append(S2N_masked[idx])
        oii_flux_list.append(A_masked[idx])
        
    for guess_num,z_pk in enumerate(z_peak_list):
        info_str = "-".join(["z%.2f"%z_pk,"oii%.2f"%oii_flux_list[guess_num], "s2n%.2f"%s2n_peak_list[guess_num]])
        title_str = "-".join([plot_title, "guess%d"%guess_num , info_str])
        # Create a figure where x-axis is shared
        ft_size = 15        
        fig, (ax0, ax1,ax2,ax3) = plt.subplots(4,figsize=(12,10),sharex=True)

        # Draw lines
        for em in emissions:
            ax0.axvline(x=(em*(z_pk+1)), ls="--", lw=2, c="red")
        for ab in absorptions:
            ax0.axvline(x=(ab*(z_pk+1)), ls="--", lw=2, c="green")            
        ax0.axvline(x=(OII_line*(z_pk+1)), ls="--", lw=2, c="blue")                        
        ax0.set_title(title_str, fontsize=ft_size)
        ax0.plot(x_masked,d_masked,lw=1, c="black")
        ax0.set_xlim([xmin, xmax])
        ax0.set_ylim([max(np.min(d_masked)*1.1,-2),np.max(d_masked)*1.1])
        ax0.set_ylabel(r"Original Flux", fontsize=ft_size)

        # Draw lines
        for em in emissions:
            ax1.axvline(x=(em*(z_pk+1)), ls="--", lw=2, c="red")
        for ab in absorptions:
            ax1.axvline(x=(ab*(z_pk+1)), ls="--", lw=2, c="green")            
        ax1.axvline(x=(OII_line*(z_pk+1)), ls="--", lw=2, c="blue")
        ax1.scatter(x_masked,A_masked,s=s, c="black", edgecolor="none")
        ax1.scatter(x_masked[isig5],A_masked[isig5],s=s, c="red", edgecolor="none")    
        if mask_caution is not None:
            ax1.scatter(x_masked[mask_caution],A_masked[mask_caution],s=s, c="blue", edgecolor="none")                    
        ax1.set_xlim([xmin, xmax])
        ax1.set_ylim([-2,np.max(A_masked)*1.1])
        ax1.set_ylabel(r"Integrated Flux", fontsize=ft_size)

        # Draw lines
        for em in emissions:
            ax2.axvline(x=(em*(z_pk+1)), ls="--", lw=2, c="red")
        for ab in absorptions:
            ax2.axvline(x=(ab*(z_pk+1)), ls="--", lw=2, c="green")            
        ax2.axvline(x=(OII_line*(z_pk+1)), ls="--", lw=2, c="blue") 
        ax2.scatter(x_masked,S2N_masked,s=s, c="black", edgecolor="none")    
        ax2.scatter(x_masked[isig5],S2N_masked[isig5],s=s, c="red", edgecolor="none")        
        ax2.axhline(y=5, ls="--", lw=2, c="blue")
        if mask_caution is not None:
            ax2.scatter(x_masked[mask_caution],S2N_masked[mask_caution],s=s, c="blue", edgecolor="none")       
        ax2.set_xlim([xmin, xmax])
        ax2.set_ylim([-1,np.max(S2N_masked)*1.1])
        ax2.set_ylabel(r"S/N", fontsize=ft_size)

        # Draw lines
        for em in emissions:
            ax3.axvline(x=(em*(z_pk+1)), ls="--", lw=2, c="red")
        for ab in absorptions:
            ax3.axvline(x=(ab*(z_pk+1)), ls="--", lw=2, c="green")            
        ax3.axvline(x=(OII_line*(z_pk+1)), ls="--", lw=2, c="blue")
        ax3.scatter(x_masked,chi_masked,s=s, c="black", edgecolor="none")
        ax3.scatter(x_masked[isig5],chi_masked[isig5],s=s, c="red", edgecolor="none")        
        if mask_caution is not None:
            ax3.scatter(x_masked[mask_caution],chi_masked[mask_caution],s=s, c="blue", edgecolor="none")            
        ax3.set_xlim([xmin, xmax])
        ax3.set_ylim([-0.5,np.max(chi_masked)*1.1])    
        ax3.set_xlabel("Wavelength ($\AA$)", fontsize=ft_size)
        ax3.set_ylabel("neg. reduced $\chi^2$", fontsize=ft_size)    

        fig.subplots_adjust(hspace=0.05)
        if plot_save:
            plt.savefig(save_dir+title_str+".png", bbox_inches="tight", dpi=200)
        if plot_show:
            plt.show()
        plt.close() 
        
    return 

def process_spec_best(d, divar, width_guesses, x_mean, mask=None):
    """
    The same as process_spec(), except returns A, varA, chi, S2N values for
    best chi.
    
    width guesses is either a list or numpy array. 
    """
    width_guesses = np.asarray(width_guesses)
    
    # First
    A, varA, chi, S2N = process_spec(d, divar, width_guesses[0], x_mean, mask=mask)    
    for i in range(1,width_guesses.size):
        A_tmp, varA_tmp, chi_tmp, S2N_tmp = process_spec(d, divar, width_guesses[i], x_mean, mask=mask)
        # Swith values if chi squar is higher. Note the we defined chi sq.
        ibool = (chi_tmp>chi) #& ~np.isnan(chi_tmp) 
        # ibool = (np.abs(1-chi_tmp)<np.abs(1-chi)) #& ~np.isnan(chi_tmp)
        # ibool = S2N_tmp>S2N
        A[ibool] = A_tmp[ibool]
        varA[ibool] = varA_tmp[ibool]
        chi[ibool] = chi_tmp[ibool]
        S2N[ibool] = S2N_tmp[ibool]
        
    return  A, varA, chi, S2N    


def plot_spectrum(x,d,x2=None,d2=None, xmin=4000, xmax=8700, lw=0.25, lw2=1, mask=None, mask2=None):
    """
    Plot a spectrum given x,d.
    """
    if mask is not None:
        d[mask] = 0
    ibool = (x>xmin)&(x<xmax)        
    
    fig = plt.figure(figsize=(10,5))
    plt.plot(x[ibool],d[ibool],lw=lw, c="black")
    
    if (x2 is not None) and (d2 is not None):
        ibool = (x2>xmin)&(x2<xmax)
        if mask2 is not None:
            d2[mask2]=0
        plt.plot(x2[ibool],d2[ibool],lw=lw2, c="red")
        
    ft_size = 15
    
    plt.xlim([xmin, xmax])
    plt.xlabel(r"Wavelength ($\AA$)", fontsize=ft_size)
    plt.ylabel(r"Flux ($10^{-17}$ ergs/cm^2/s/$\AA$)", fontsize=ft_size)
    plt.show()
    plt.close()

def plot_S2N(x, S2N, mask=None, xmin=4500, xmax=8500, s=1):
    """
    Plot a spectrum given x,d.
    """
    if mask is not None:
        S2N[mask] = 0
    ibool = (x>xmin)&(x<xmax)        
    
    fig = plt.figure(figsize=(10,5))
    S2N_masked = S2N[ibool]
    plt.scatter(x[ibool],S2N_masked,s=s, c="black")
        
    ft_size = 15
    
    plt.xlim([xmin, xmax])
    plt.ylim([np.min(S2N_masked)*1.2,np.max(S2N_masked)*1.2])
    plt.xlabel(r"Wavelength ($\AA$)", fontsize=ft_size)
    plt.ylabel(r"S/N", fontsize=ft_size)
    plt.show()
    plt.close()    



    
def MMT_radec(field, MMT_data_directory="./MMT_data/"):
    """
    field is one of [0,1,2]:
        - 0: 16hr observation 1
        - 1: 16hr observation 2
        - 2: 23hr observation
    MMT_data_directory: Where the relevant header files are stored.
    """
    num_fibers = 300

    if field==0:
        # 16hr2_1
        # Header file name
        fname = MMT_data_directory+"config1FITS_Header.txt"
        # Get info corresponding to the fibers
        OnlyAPID = [line for line in return_file(fname) if line.startswith("APID")]
        # Get the object type
        APID_types = [line.split("= '")[1].split(" ")[0] for line in OnlyAPID]
        # print(APID_types)
        # Getting index of targets only
        ibool1 = np.zeros(num_fibers,dtype=bool)
        for i,e in enumerate(APID_types):
            if e.startswith("5"):
                ibool1[i] = True        
        APID_targets = [OnlyAPID[i] for i in range(num_fibers) if ibool1[i]]
        fib = [i+1 for i in range(num_fibers) if ibool1[i]]
        # Extract ra,dec
        ra_str = [APID_targets[i].split("'")[1].split(" ")[1] for i in range(len(APID_targets))]
        dec_str = [APID_targets[i].split("'")[1].split(" ")[2] for i in range(len(APID_targets))]
        ra = [HMS2deg(ra=ra_str[i]) for i in range(len(ra_str))]
        dec = [HMS2deg(dec=dec_str[i]) for i in range(len(ra_str))]
    elif field==1:
        # 16hr2_2
        # Header file name
        fname = MMT_data_directory+"config2FITS_Header.txt"
        # Get info corresponding to the fibers
        OnlyAPID  = return_file(fname)[0].split("= '")[1:]
        # Get the object type
        APID_types = [line.split(" ")[0] for line in OnlyAPID]
        # print(APID_types)
        # Getting index of targets only
        ibool2 = np.zeros(num_fibers,dtype=bool)
        for i,e in enumerate(APID_types):
            if e.startswith("5"):
                ibool2[i] = True        
        APID_targets = [OnlyAPID[i] for i in range(num_fibers) if ibool2[i]]
        fib = [i+1 for i in range(num_fibers) if ibool2[i]]
        # print(APID_targets[0])
        # Extract ra,dec
        ra_str = [APID_targets[i].split(" ")[1] for i in range(len(APID_targets))]
        dec_str = [APID_targets[i].split(" ")[2] for i in range(len(APID_targets))]
        ra = [HMS2deg(ra=ra_str[i]) for i in range(len(ra_str))]
        dec = [HMS2deg(dec=dec_str[i]) for i in range(len(ra_str))]
    elif field==2:
        # 23hr
        # Header file name
        fname = MMT_data_directory+"23hrs_FITSheader.txt"
        # Get info corresponding to the fibers
        OnlyAPID  = return_file(fname)[0].split("= '")[1:]
        # Get the object type
        APID_types = [line.split(" ")[0] for line in OnlyAPID]
        # print(APID_types)
        # Getting index of targets only
        ibool3 = np.zeros(num_fibers,dtype=bool)
        for i,e in enumerate(APID_types):
            if e.startswith("3"):
                ibool3[i] = True        
        APID_targets = [OnlyAPID[i] for i in range(num_fibers) if ibool3[i]]
        fib = [i+1 for i in range(num_fibers) if ibool3[i]]        
        # print(APID_targets[0])
        # Extract ra,dec
        ra_str = [APID_targets[i].split(" ")[1] for i in range(len(APID_targets))]
        dec_str = [APID_targets[i].split(" ")[2] for i in range(len(APID_targets))]
        ra = [HMS2deg(ra=ra_str[i]) for i in range(len(ra_str))]
        dec = [HMS2deg(dec=dec_str[i]) for i in range(len(ra_str))]
    
    return np.asarray(ra), np.asarray(dec), np.asarray(fib)



def plot_dNdz_selection(cn, w, iselect1, redz, area, dz=0.05, gold_eff=1, silver_eff=1, NoZ_eff=0.25, NoOII_eff=0.6,\
    gold_eff2=1, silver_eff2=1, NoZ_eff2=0.25, NoOII_eff2=0.6,\
    cn2=None, w2=None, iselect2=None, redz2=None, plot_total=True, fname="dNdz.png", color1="black", color2="red", color_total="green",\
     label1="Selection 1", label2="Selection 2", label_total="DEEP2 Total", wNoOII=0.1, wNoZ=0.5, lw=1.5, \
     label_np1="nP=1", color_np1="blue", plot_np1 = True):
    """
    Given class number (cn), mask (iselect1), weights (w), redshifts, class efficiencies, plot the redshift
    histogram. 

    dz: Histogram binwidth
    **_eff: Gold and Silver are NOT always equal to one. NoZ and NoOII are objects wtih no redshift
        in DEEP2 but are guessed to have efficiency of about 0.25.
    **_eff2: The efficiencies for the second set.
    iselect2: If not None, used as another set of mask to plot dNdz histogram.
    plot_total: Plots total.
    fname: Saves in fname.
    color1: iselect1 color
    color2: iselect2 color
    color_total: total color
    label1, label2, lbael_total: Labels
    """

    mpl.rcParams['xtick.labelsize'] = 20
    mpl.rcParams['ytick.labelsize'] = 20

    if plot_total:
        ibool = np.logical_or((cn==0),(cn==1)) 
        plt.hist(redz[ibool], bins = np.arange(0.6,1.7,dz), weights=w[ibool]/area,\
                 histtype="step", color=color_total, label=label_total, lw=lw)

        # NoOII:
        ibool = (cn==3) 
        N_NoOII = NoOII_eff*w[ibool].sum();
        plt.bar(left=0.7, height =N_NoOII/(wNoOII/dz), width=wNoOII, bottom=0., alpha=0.5,color=color_total, \
                edgecolor =color_total, label=label_total+" NoOII (Proj.)", hatch="*")
        # NoZ:
        ibool = (cn==5) 
        N_NoZ = NoZ_eff*w[ibool].sum();
        plt.bar(left=1.4, height =N_NoZ/(wNoZ/dz), width=wNoZ, bottom=0., alpha=0.5,color=color_total, \
                edgecolor =color_total, label=label_total+" NoZ (Proj.)")


    if iselect2 is not None:
        # If the new cn, w and redz are given, then use those values. Else, use first set.
        if cn2 is None:
            redz2 = np.copy(redz)
            cn2 = np.copy(cn)
            w2 = np.copy(w)
        # appropriately weighing the objects.
        w_select2 = np.copy(w2)
        w_select2[cn2==0] *= gold_eff2
        w_select2[cn2==1] *= silver_eff2
        w_select2[cn2==3] *= NoOII_eff2
        w_select2[cn2==5] *= NoZ_eff2

        ibool = np.logical_or((cn2==0),(cn2==1)) & iselect2
        plt.hist(redz2[ibool], bins = np.arange(0.6,1.7,dz), weights=w_select2[ibool]/area,\
                 histtype="step", color=color2, label=label2, lw=lw)

        # NoOII:
        ibool = (cn2==3) & iselect2
        N_NoOII = w_select2[ibool].sum();
        plt.bar(left=0.7, height =N_NoOII/(wNoOII/dz), width=wNoOII, bottom=0., alpha=0.5,color=color2, \
                edgecolor =color2, label=label2+ " NoOII (Proj.)", hatch="*")
    
        plt.plot([0.7, 0.7+wNoOII], [N_NoOII/(wNoOII/dz)/NoOII_eff2, N_NoOII/(wNoOII/dz)/NoOII_eff2], color=color2, linewidth=2.0, ls="--")


        # NoZ:
        ibool = (cn2==5) & iselect2
        N_NoZ = w_select2[ibool].sum();
        plt.bar(left=1.4, height =N_NoZ/(wNoZ/dz), width=wNoZ, bottom=0., alpha=0.5,color=color2, \
                edgecolor =color2, label=label2+" NoZ (Proj.)")         

        plt.plot([1.4, 1.4+wNoZ], [N_NoZ/(wNoZ/dz)/NoZ_eff2, N_NoZ/(wNoZ/dz)/NoZ_eff2], color=color2, linewidth=2.0, ls="--")

    # Selection 1.
    # appropriately weighing the objects.
    w_select1 = np.copy(w)
    w_select1[cn==0] *= gold_eff
    w_select1[cn==1] *= silver_eff
    w_select1[cn==3] *= NoOII_eff
    w_select1[cn==5] *= NoZ_eff

    ibool = np.logical_or((cn==0),(cn==1)) & iselect1 # Total
    plt.hist(redz[ibool], bins = np.arange(0.6,1.7,dz), weights=w_select1[ibool]/area,\
             histtype="step", color=color1, label=label1, lw=lw)

    # NoOII:
    ibool = (cn==3) & iselect1
    N_NoOII = w_select1[ibool].sum();
    plt.bar(left=0.7, height =N_NoOII/(wNoOII/dz), width=wNoOII, bottom=0., alpha=0.5,color=color1, \
            edgecolor =color1, label=label1+" NoOII (Proj.)", hatch="*")

    plt.plot([0.7, 0.7+wNoOII], [N_NoOII/(wNoOII/dz)/NoOII_eff, N_NoOII/(wNoOII/dz)/NoOII_eff], color=color1, linewidth=2.0, ls="--")

    # NoZ:
    ibool = (cn==5) & iselect1
    N_NoZ = w_select1[ibool].sum();
    plt.bar(left=1.4, height =N_NoZ/(wNoZ/dz), width=wNoZ, bottom=0., alpha=0.5, color=color1, \
            edgecolor =color1, label=label1+" NoZ (Proj.)")

    plt.plot([1.4, 1.4+wNoZ], [N_NoZ/(wNoZ/dz)/NoZ_eff, N_NoZ/(wNoZ/dz)/NoZ_eff], color=color1, linewidth=2.0, ls="--")

    # Plotting np=1 line
    if plot_np1:
        X,Y = np1_line(dz)
        plt.plot(X,Y, color=color_np1, label=label_np1, lw=lw*2., ls="-.")

 
    plt.xlim([0.5,1.4+wNoZ+0.1])
    plt.legend(loc="upper right", fontsize=15)  
    ymax=260
    if plot_total:
        ymax = 450
    plt.ylim([0,ymax])
    # plt.legend(loc="upper left")
    plt.xlabel("Redshift z", fontsize=20)
    plt.ylabel("dN/d(%.3fz) per sq. degs."%dz, fontsize=20)
    plt.savefig(fname, bbox_inches="tight", dpi=400)
    # plt.show()
    plt.close()


def np1_line(dz=0.5):
    """
    Given the binwidth dz, return np=1 line.
    """
    X, Y = np.asarray([[0.14538014092363039, 1.1627906976744384],
    [0.17035196758073518, 2.906976744186011],
    [0.20560848729069203, 5.8139534883720785],
    [0.2731789775637742, 10.465116279069775],
    [0.340752313629068, 15.697674418604663],
    [0.4083256496943619, 20.930232558139494],
    [0.4729621281972476, 26.16279069767444],
    [0.5405354642625415, 31.395348837209326],
    [0.6081088003278353, 36.62790697674416],
    [0.6756821363931291, 41.860465116279045],
    [0.7403214606882265, 47.67441860465118],
    [0.8078919509613086, 52.32558139534882],
    [0.8754624412343909, 56.97674418604652],
    [0.9430357772996848, 62.209302325581405],
    [1.0106034217805555, 66.27906976744185],
    [1.0811107696160458, 70.93023255813955],
    [1.1486784140969166, 75],
    [1.2162432127855753, 78.48837209302326],
    [1.2867448690366428, 81.97674418604649],
    [1.3543096677253015, 85.46511627906978],
    [1.4248084781841568, 88.37209302325581],
    [1.4953072886430125, 91.27906976744185],
    [1.5687401108720649, 93.6046511627907],
    [1.6392389213309202, 96.51162790697674],
    [1.7097320402053522, 98.2558139534884],
    [1.7802280048719963, 100.58139534883719],
    [1.8507211237464292, 102.32558139534885],
    [1.9212113968286495, 103.48837209302326],
    [1.9917045157030815, 105.23255813953489]]).T 

    return X, Y*dz/0.1


def flux2asinh_mag(flux, band = "g"):
    """
    Returns asinh magnitude. The b parameter is set following discussion surrounding
    eq (9) of the paper on luptitude. b = 1.042 sig_f. 
    
    Sig_f for each fitler has been obtained based on DEEP2 deep fields and is in nanomaggies.

    Sig_oii is based on DEEP2 OII flux values in 10^-17 ergs/cm^2/s unit.
    """
    b = None
    if band == "g":
        b = 1.042 * 0.0285114
    elif band == "r":
        b = 1.042 * 0.0423106
    elif band == "z":
        b = 1.042 * 0.122092
    elif band == "oii":
        b = 1.042 * 0.581528277909
    return 22.5-2.5 * np.log10(b) - 2.5 * np.log10(np.e) * np.arcsinh(flux/(2*b))



def asinh_mag2flux(mu, band = "g"):
    """
    Invsere of flux2asinh_mag
    """
    b = None
    if band == "g":
        b = 1.042 * 0.0285114
    elif band == "r":
        b = 1.042 * 0.0423106
    elif band == "z":
        b = 1.042 * 0.122092
    elif band == "oii":
        b = 1.042 * 0.581528277909
        
    flux = 2* b * np.sinh((22.5-2.5 * np.log10(b) - mu) / (2.5 * np.log10(np.e)))
    return flux

def FDR_cut(grz):
    """
    Given a list [g,r,z] magnitudes, apply the cut and return an indexing boolean vector.
    """
    g,r,z=grz; yrz = (r-z); xgr = (g-r)
    ibool = (r<23.4) & (yrz>.3) & (yrz<1.6) & (xgr < (1.15*yrz)-0.15) & (xgr < (1.6-1.2*yrz))
    return ibool




def sample_GMM(Sxamp,Sxmean, Sxcovar, ycovar):
    """
    Return a sample based on the GMM input.
    """
    N = ycovar.shape[0] # Number of data points. 
    sample = []
    # For each data point, generate a sample based on the specified GMM. 
    for i in range(N):
        sample.append(sample_GMM_generate(Sxamp,Sxmean, Sxcovar, ycovar[i]))
    sample = np.asarray(sample)
#     print sample.shape, sample
    xgr_sample, yrz_sample = sample[:,0], sample[:,1]
    return xgr_sample, yrz_sample

def sample_GMM_generate(Sxamp,Sxmean, Sxcovar, cov):
    """
    sample from a gaussian mixture
    """
    # Number of components.
    K = Sxamp.size
    if K == 1:
#         print(Sxmean[0], (Sxcovar+cov)
        one_sample = np.random.multivariate_normal(Sxmean[0], (Sxcovar+cov)[0], size=1)[0]
        return one_sample
    
    # Choose from the number based on multinomial
    m = np.where(np.random.multinomial(1,Sxamp)==1)[0][0]
    # Draw from the m-th gaussian.
    one_sample = np.random.multivariate_normal(Sxmean[m], Sxcovar[m]+cov, size=1)[0]
    return one_sample



def plot_XD_fit_K(ydata, ycovar, Sxamp, Sxmean, Sxcovar, fname=None, pt_size=5, mask=None, show=False):
    """
    Used for model selection.
    """
    bnd_lw = 1.
    # Unpack the colors.
    xgr = ydata[:,0]; yrz = ydata[:,1]
    if mask is not None:
        yrz = yrz[mask]
        xgr = xgr[mask]
        
    # # Broad boundary
    # xbroad, ybroad = generate_broad()
    # Figure ranges
    grmin = -1.
    rzmin = -.75
    grmax = 2.5
    rzmax = 2.75
    
    # Create figure 
    f, axarr = plt.subplots(2, 2, figsize=(14,14))

    # First panel is the original.
    axarr[0,0].scatter(xgr,yrz, c="black",s=pt_size, edgecolors="none")
    # FDR boundary:
    axarr[0,0].plot([-4, 0.195], [0.3, 0.30], 'k-', lw=bnd_lw, c="red")
    axarr[0,0].plot([0.195, 0.706],[0.3, 0.745], 'k-', lw=bnd_lw, c="red")
    axarr[0,0].plot([0.706, -0.32], [0.745, 1.6], 'k-', lw=bnd_lw, c="red")
    axarr[0,0].plot([-0.32, -4],[1.6, 1.6], 'k-', lw=bnd_lw, c="red")
    # # Broad
    # axarr[0,0].plot(xbroad,ybroad, linewidth=bnd_lw, c='blue')
    # Decoration
    axarr[0,0].set_xlabel("$g-r$",fontsize=18)
    axarr[0,0].set_ylabel("$r-z$",fontsize=18)
    axarr[0,0].set_title("Data",fontsize=20)        
    axarr[0,0].axis("equal")
    axarr[0,0].axis([grmin, grmax, rzmin, rzmax]) 
    
    
    # The remaining three are simulation based on the fit.
    sim_counter = 1
    for i in range(1,4):
        xgr_sample, yrz_sample = sample_GMM(Sxamp,Sxmean, Sxcovar, ycovar)
        axarr[i//2, i%2].scatter(xgr_sample,yrz_sample, c="black",s=pt_size, edgecolors="none")
        # FDR boundary:
        axarr[i//2, i%2].plot([-4, 0.195], [0.3, 0.30], 'k-', lw=bnd_lw, c="red")
        axarr[i//2, i%2].plot([0.195, 0.706],[0.3, 0.745], 'k-', lw=bnd_lw, c="red")
        axarr[i//2, i%2].plot([0.706, -0.32], [0.745, 1.6], 'k-', lw=bnd_lw, c="red")
        axarr[i//2, i%2].plot([-0.32, -4],[1.6, 1.6], 'k-', lw=bnd_lw, c="red")
        # Broad
        # axarr[i//2, i%2].plot(xbroad,ybroad, linewidth=bnd_lw, c='blue')
        # Decoration
        axarr[i//2, i%2].set_xlabel("$g-r$",fontsize=18)
        axarr[i//2, i%2].set_ylabel("$r-z$",fontsize=18)
        axarr[i//2, i%2].set_title("Simulation %d" % sim_counter,fontsize=20); sim_counter+=1
        axarr[i//2, i%2].axis("equal")
        axarr[i//2, i%2].axis([grmin, grmax, rzmin, rzmax])     

    if fname is not None:
#         plt.savefig(fname+".pdf", bbox_inches="tight",dpi=200)
        plt.savefig(fname+".png", bbox_inches="tight",dpi=200)    
    if show:
        plt.show()
    plt.close()    

def save_params(Sxamp_init, Sxmean_init, Sxcovar_init, Sxamp, Sxmean, Sxcovar, i, K, tag=""):
    fname = ("%d-params-fit-amps-glim24-K%d"+tag) %(i, K)
    np.save(fname, Sxamp)
    fname = ("%d-params-fit-means-glim24-K%d"+tag) %(i, K)
    np.save(fname, Sxmean)
    fname = ("%d-params-fit-covars-glim24-K%d"+tag) %(i, K)
    np.save(fname, Sxcovar)
    # Initi parameters
    fname = ("%d-params-init-amps-glim24-K%d"+tag) %(i, K)
    np.save(fname, Sxamp_init)
    fname = ("%d-params-init-means-glim24-K%d"+tag) %(i, K)
    np.save(fname, Sxmean_init)
    fname = ("%d-params-init-covars-glim24-K%d"+tag) %(i, K)
    np.save(fname, Sxcovar_init)
    return

def amp_init(K):
    return np.ones(K,dtype=np.float)/np.float(K)

def mean_init(K, ydata):
    S = np.random.randint(low=0,high=ydata.shape[0],size=K)
    return ydata[S]

def covar_init(K, init_var):
    covar = np.zeros((K, 2,2))
    for i in range(K):
        covar[i] = np.diag((init_var, init_var))
    return covar 

def XD_init(K, ydata, init_var):
    xamp_init = amp_init(K)
    # print xamp_init, xamp_init.shape
    xmean_init = mean_init(K, ydata)
    # print xmean_init, xmean_init.shape
    xcovar_init = covar_init(K, init_var)
    # print xcovar_init, xcovar_init.shape
    
    return xamp_init, xmean_init, xcovar_init






def grz2gr_rz(grz):
    return np.transpose(np.asarray([grz[0]-grz[1], grz[1]-grz[2]]))

def grz2rz_gr(grz):
    return np.transpose(np.asarray([grz[1]-grz[2], grz[0]-grz[1]]))    

def fvar2mvar(f, fivar):
    return (1.08574)**2/(f**2 * fivar)
    
def gr_rz_covariance(grzflux, grzivar):
    gflux = grzflux[0]
    rflux = grzflux[1]
    zflux = grzflux[2]
    givar = grzivar[0]
    rivar = grzivar[1]
    zivar = grzivar[2]
    
    gvar = fvar2mvar(gflux,givar)
    rvar = fvar2mvar(rflux,rivar)
    zvar = fvar2mvar(zflux,zivar)
    
    gr_rz_covar = np.zeros((gvar.size ,2,2))
    for i in range(gvar.size):
#         if i % 100 == 0:
#             print i
        gr_rz_covar[i] = np.asarray([[gvar[i]+rvar[i], rvar[i]],[rvar[i], rvar[i]+zvar[i]]])
    
    return gr_rz_covar

def rz_gr_covariance(grzflux, grzivar):
    gflux = grzflux[0]
    rflux = grzflux[1]
    zflux = grzflux[2]
    givar = grzivar[0]
    rivar = grzivar[1]
    zivar = grzivar[2]
    
    gvar = fvar2mvar(gflux,givar)
    rvar = fvar2mvar(rflux,rivar)
    zvar = fvar2mvar(zflux,zivar)
    
    rz_gr_covar = np.zeros((gvar.size ,2,2))
    for i in range(gvar.size):
#         if i % 100 == 0:
#             print i
        rz_gr_covar[i] = np.asarray([[rvar[i]+zvar[i], rvar[i]],[rvar[i], gvar[i]+rvar[i]]])
    
    return rz_gr_covar    


def pow_legend(params_pow):
    alpha, A = params_pow
    return r"$A=%.2f,\,\, \alpha=%.2f$" % (A, alpha)

def broken_legend(params_broken):
    alpha, beta, fs, phi = params_broken
    return r"$\alpha=%.2f, \,\, \beta=%.2f, \,\, f_i=%.2f, \,\, \phi=%.2f$" % (alpha, beta, fs, phi)


def broken_pow_phi_init(flux_centers, best_params_pow, hist,bw, fluxS):
    """
    Return initial guess for phi.
    """
    # selecting one non-zero bin
    c_S = 0;
    while c_S == 0:
        S = np.random.randint(low=0,high=flux_centers.size,size=1)
        f_S = flux_centers[S]
        c_S = hist[S]
        
    alpha = -best_params_pow[0]
    beta = best_params_pow[0]
    phi = c_S/broken_pow_law([alpha, beta, fluxS, 1.], f_S)/bw
    
    # phi
    return phi[0]


def pow_law(params, flux):
    A = params[1]
    alpha = params[0]
    return A* flux**alpha

def broken_pow_law(params, flux):
    """
    If alpha < beta, then for f > fs, f**alpha, and for f < fs, f**beta.
    fs: The break point where the behavior of the function changes from one power law to the other.
    phi: Sets the overall height of the function    
    """

    alpha = params[0]
    beta = params[1]
    fs = params[2]
    phi = params[3]
    return phi/((flux/fs)**(-alpha)+(flux/fs)**(-beta) + 1e-12)

def integrate_broken_pow_law(params, fmin, fmax, df):
    farray = np.arange(fmin, fmax+df/2., df)
    dNdf_grid = broken_pow_law(params, farray)
    return trapz(dNdf_grid, farray)


def dNdm2dNdf(m):
    """
    dNdm/dNdf_m (m) factor
    """
    return (2*np.log(10)/5.)*mag2flux(m)

def pow_mag_param_init(bin_centers, left_hist, right_hist, bw, area):
    """
    Return initial guess for the exponent and normalization.
    """
    # Size of the left and right histogram length
    N_left = left_hist.size
    N_right = right_hist.size
    
    # Randomly pick a bin from the left and the right
    idx_left = np.random.choice(range(N_left))
    idx_right = np.random.choice(range(N_right))
    
    # Corresponding mag values and counts
    mag_left = bin_centers[idx_left]
    mag_right= bin_centers[idx_right+N_left]
    c_left = left_hist[idx_left]
    c_right = right_hist[idx_right]

    # print mag_left, mag_right, c_left, c_right
    # assert False
    
    # Solve for A and alpha 
    alpha =  np.log((c_left+1)/float(c_right+1)) / np.log(mag_left/float(mag_right)) # +1 for regularization
    A = c_left / ((mag_left ** alpha) * bw * area)
    
    return A, alpha




def dNdm_fit_broken_pow(mag, weight, bw, magmin, magmax, area, niter = 5, pow_tol =1e-5):
    """
    Given the magnitudes and the corresponding weight, and the parameters for the histogram, 
    return the best fit parameters for a broken power law.
    
    Note: This function could be much more modular. But for now I keep it as it is.
    """
    # Computing the histogram.
    bins = np.arange(magmin, magmax+bw/2., bw)
    hist, bin_edges = np.histogram(mag, weights=weight, bins=bins)

    # Compute the median magnitude
    magmed = np.median(mag)

    # Compute bin centers. Left set and right set.
    bin_centers = (bin_edges[:-1]+bin_edges[1:])/2.
    ileft = bin_centers < magmed
    # left and right counts
    left_hist = hist[ileft]
    right_hist = hist[~ileft]

    # Place holder for the best parameters. phi, ms, alpha, beta 
    best_params_pow = np.zeros(4, dtype=np.float) 

    # Empty list for the negative log-likelihood
    list_nloglike = []
    best_nloglike = -np.inf # -100

    # Define negative total loglikelihood function given the histogram.
    def ntotal_loglike_pow(params):
        """
        Total log likelihood.
        """
        total_loglike = 0
        for i in range(bin_centers.size):
            total_loglike += stats.poisson.logpmf(hist[i].astype(int), mag_broken_pow_law(params, bin_centers[i]) * bw * area)

        return -total_loglike

    # fit for niter times 
    counter = 0
    while counter < niter:
        print "Try %d" % counter
        # Generate initial parameters
        init_params = pow_mag_param_init(bin_centers, left_hist, right_hist, bw, area)
        A, alpha = init_params
        # assert False
        
        # Optimize the parameters.
        ms_guess = 22.5

        init_params = [max(A*ms_guess**2, 100), ms_guess, min(alpha, 10), np.random.choice([-0.1, 0.1])]           
        res = opt.minimize(ntotal_loglike_pow, init_params, tol=pow_tol,method="Powell")
        counter+=1

        if res["success"]:
            fitted_params = res["x"]

            # Calculate the negative total likelihood
            nloglike = ntotal_loglike_pow(fitted_params)
            list_nloglike.append(nloglike)

            # If loglike is the highest among seen, then update the parameters.
            if nloglike > best_nloglike:
                best_nloglike = nloglike
                best_params_pow = fitted_params
            print "Optimization suceed."
        else:
            fitted_params = res["x"]
            print "Stopped at", fitted_params
            print "Optimization failed."

    print(best_params_pow)

    return best_params_pow


def mag_broken_pow_law(params, mags):
    phi, ms, alpha, beta = params    
    return phi/((mags/ms)**(-alpha) +  (mags/ms)**(-beta))    

def mag_pow_law(params, mags):
    A, alpha = params
    return A * mags**alpha

def integrate_mag_broken_pow_law(params, mag_min, mag_max, area=1):
    dm = 5e-3
    mag_bins = np.arange(mag_min, mag_max, dm)
    return trapz(mag_broken_pow_law(params, mag_bins), mag_bins) * area



def dNdm_fit(mag, weight, bw, magmin, magmax, area, niter = 5, pow_tol =1e-5):
    """
    Given the magnitudes and the corresponding weight, and the parameters for the histogram, 
    return the best fit parameters for a power law.
    
    Note: This function could be much more modular. But for now I keep it as it is.
    """
    # Computing the histogram.
    bins = np.arange(magmin, magmax+bw/2., bw)
    hist, bin_edges = np.histogram(mag, weights=weight, bins=bins)

    # Compute the median magnitude
    magmed = np.median(mag)

    # Compute bin centers. Left set and right set.
    bin_centers = (bin_edges[:-1]+bin_edges[1:])/2.
    ileft = bin_centers < magmed
    # left and right counts
    left_hist = hist[ileft]
    right_hist = hist[~ileft]

    # Place holder for the best parameters A, alpha
    best_params_pow = np.zeros(2,dtype=np.float) 

    # Empty list for the negative log-likelihood
    list_nloglike = []
    best_nloglike = -np.inf # -100

    # Define negative total loglikelihood function given the histogram.
    def ntotal_loglike_pow(params):
        """
        Total log likelihood.
        """
        total_loglike = 0
        A, alpha = params
        for i in range(bin_centers.size):
            total_loglike += stats.poisson.logpmf(hist[i].astype(int), A * bin_centers[i]**alpha * bw * area)

        return -total_loglike

    # fit for niter times 
    counter = 0
    while counter < niter:
        print "Try %d" % counter
        # Generate initial parameters
        init_params = pow_mag_param_init(bin_centers, left_hist, right_hist, bw, area)
        print init_params
        # assert False
        
        # Optimize the parameters.
        res = opt.minimize(ntotal_loglike_pow, init_params,tol=pow_tol,method="Nelder-Mead" )
        counter+=1

        if res["success"]:
            fitted_params = res["x"]

            # Calculate the negative total likelihood
            nloglike = ntotal_loglike_pow(fitted_params)
            list_nloglike.append(nloglike)

            # If loglike is the highest among seen, then update the parameters.
            if nloglike > best_nloglike:
                best_nloglike = nloglike
                best_params_pow = fitted_params
            print "Optimization suceed."
        else:
            print "Optimization failed."

#     print(best_params_pow)

    return best_params_pow


def gen_mag_pow_law_samples(params, mag_min, mag_max, Nsample):
    """
    Generate Nsample power law samples in the specified range.
    """
    A, alpha = params
    
    if Nsample > 1e5:
        N_per_iter = int(1e5)
    else:
        N_per_iter = Nsample
        
    N_counter = 0 # Tallying the number of samples generate so far
    mag_list = [] 
    while N_counter < Nsample:
        uni = np.random.random(N_per_iter) # Sample uniform randoms
        mag_tmp = mag_max * uni**(1/float(alpha+1)) # Use inverse cdf function to compute samples
#         print mag_tmp
        
        ibool = (mag_tmp > mag_min) & (mag_tmp < mag_max) # Check if in the range
        mag_tmp = mag_tmp[ibool]
        mag_list.append(mag_tmp)
        N_counter += mag_tmp.size # Update the siae
#         assert False
    mag_list = np.concatenate(mag_list)[:Nsample]
    
    return mag_list


def integrate_mag_pow_law(params, mag_min, mag_max, area = 1):
    # Returns integrated number given the law
    A, alpha = params
    return (A/float(alpha+1)) * (mag_max**(alpha+1) - mag_min**(alpha+1)) * area





def integrate_pow_law(alpha, A, fmin, fmax):
    """
    Given power law model [alpah, A] of A x f**alpha, 
    return the integrated number.
    """
    return A * (fmax**(1 + alpha) - fmin**(1 + alpha))/(1 + alpha)


def gen_pow_law_sample(fmin, nsample, alpha, exact=False, fmax=None, importance_sampling=False, alpha_importance=None):
    """
    Note the convention f**alpha, alpha>0.??
    
    If exact, then return nsample number of sample exactly between fmin and fmax.

    If importance_sampling, then generate the samples using the alpha_importance,
    and return the corresponding importance weights along with the sample.
    iw = f_sample^(alpha-alpha_importance)
    """
    flux = None
    if importance_sampling:
        if exact:
            assert (fmax is not None)
            flux = fmin * np.exp(np.log(np.random.rand(nsample))/(alpha_importance+1))
            ibool = (flux>fmin) & (flux<fmax)
            flux = flux[ibool]
            nsample_counter = np.sum(ibool)
            while nsample_counter < nsample:
                flux_tmp = fmin * np.exp(np.log(np.random.rand(nsample))/(alpha_importance+1))
                ibool = (flux_tmp>fmin) & (flux_tmp<fmax)
                flux_tmp = flux_tmp[ibool]
                nsample_counter += np.sum(ibool)
                flux = np.concatenate((flux, flux_tmp))
            flux = flux[:nsample]# np.random.choice(flux, nsample, replace=False)
            iw = flux**(alpha-alpha_importance)
        else:
            pass

        return flux, iw
    else:
        if exact:
            assert (fmax is not None)
            flux = fmin * np.exp(np.log(np.random.rand(nsample))/(alpha+1))
            ibool = (flux>fmin) & (flux<fmax)
            flux = flux[ibool]
            nsample_counter = np.sum(ibool)
            while nsample_counter < nsample:
                flux_tmp = fmin * np.exp(np.log(np.random.rand(nsample))/(alpha+1))
                ibool = (flux_tmp>fmin) & (flux_tmp<fmax)
                flux_tmp = flux_tmp[ibool]
                nsample_counter += np.sum(ibool)
                flux = np.concatenate((flux, flux_tmp))
            flux = flux[:nsample]# np.random.choice(flux, nsample, replace=False)
        else:
            assert False # This mode is not supported.
            # flux = fmin * np.exp(np.log(np.random.rand(nsample))/(alpha+1))
        
        return flux

def sample_MoG(amps, means, covs, nsample, importance_sampling=False, factor_importance = 1.5):
    """
    Given MoG parameters, return a sample specified by nsample.

    If importance_sampling, then generate sample from the MoG with a covar matrix enlarged by
    factor_importance and then return the corresponding importance weight factor.
    """
    nsample_per_component = np.random.multinomial(nsample, amps)
    
    if importance_sampling:
        sample = []
        iw = []
        for i, ns in enumerate(nsample_per_component):
            sample_tmp = np.random.multivariate_normal(means[i], (factor_importance**2)*covs[i], ns)
            iw_tmp = multivariate_normal.pdf(sample_tmp, mean=means[i], cov=covs[i])/multivariate_normal.pdf(sample_tmp, mean=means[i], cov=(factor_importance**2)*covs[i])
            sample.append(sample_tmp)            
            iw.append(iw_tmp)
        sample = np.vstack(sample)
        iw = np.concatenate(iw)
        return sample, iw
    else:
        sample = []
        for i, ns in enumerate(nsample_per_component):
            sample.append(np.random.multivariate_normal(means[i], covs[i], ns))
        sample = np.vstack(sample)
        
        return sample


def pow_param_init_dNdf(left_hist, left_f, right_hist, right_f, bw, area):
    """
    Return initial guess for the exponent and normalization.
    """
    # selecting non-zero bin one from left and one from right. 
    c_L = 0; c_R = 0
    while c_L==0 or c_R == 0 or c_L <= c_R:
        L = np.random.randint(low=0,high=left_hist.size,size=1)
        f_L = left_f[L]
        c_L = left_hist[L]
        R = np.random.randint(low=0,high=right_hist.size,size=1)
        f_R = right_f[R]
        c_R = right_hist[R]
#     print(L,R)
    # exponent
    alpha_init = np.log(c_L/np.float(c_R))/np.log(f_L/np.float(f_R))
    A_init = c_R/(f_R**alpha_init * bw * area)
    
    ans = np.zeros(2, dtype=np.float)
    ans[0] = alpha_init
    ans[1] = A_init
    return ans


def dNdf_fit_broken_pow(flux, weight, bw, fmin, fmax, area, niter = 5, pow_tol =1e-5):
    """
    Given the fluxes and the corresponding weight, and the parameters for the histogram, 
    return the best fit parameters for a *broken* pow law.
    
    Note: This function could be much more modular. But for now I keep it as it is.
    """
    # Computing the histogram.
    bins = np.arange(fmin, fmax+bw/2., bw)
    hist, bin_edges = np.histogram(flux, weights=weight, bins=bins)

    # Compute the median magnitude
    fmed = np.median(flux)

    # Compute bin centers. Left set and right set.
    bin_centers = (bin_edges[:-1]+bin_edges[1:])/2.
    ileft = bin_centers < fmed
    # left and right counts
    left_hist = hist[ileft]
    right_hist = hist[~ileft]
    # left and right flux
    left_f = bin_centers[ileft]
    right_f = bin_centers[~ileft]

    # Place holder for the best parameters
    best_params_pow = np.zeros(4, dtype=np.float) 

    # Empty list for the negative log-likelihood
    list_nloglike = []
    best_nloglike = -np.inf # -100

    # Define negative total loglikelihood function given the histogram.
    def ntotal_loglike_pow(params):
        """
        Total log likelihood.
        """
        total_loglike = 0

        for i in range(bin_centers.size):
            total_loglike += stats.poisson.logpmf(hist[i].astype(int), broken_pow_law(params, bin_centers[i]) * bw * area)

        return -total_loglike

    # fit for niter times 
    counter = 0
    while counter < niter:
        print "Try %d" % counter

        # Generate initial parameters
        init_params = pow_param_init_dNdf(left_hist, left_f, right_hist, right_f, bw, area)
#         print init_params
        init_params = np.array([init_params[0], -1.05, mag2flux(23.), 1]) #init_params[1]*5])
    
        # Optimize the parameters.
        res = opt.minimize(ntotal_loglike_pow, init_params,tol=pow_tol,method="Nelder-Mead" )
        counter+=1
#         if counter % 2 == 0:
        if res["success"]:
            fitted_params = res["x"]

            # Calculate the negative total likelihood
            nloglike = ntotal_loglike_pow(fitted_params)
            list_nloglike.append(nloglike)

            # If loglike is the highest among seen, then update the parameters.
            if nloglike > best_nloglike:
                best_nloglike = nloglike
                best_params_pow = fitted_params
            print "Optimization success"
        else:
            print "Optimization failed."

#     print(best_params_pow)

    return best_params_pow

def dNdf_fit(flux, weight, bw, fmin, fmax, area, niter = 5, pow_tol =1e-5):
    """
    Given the fluxes and the corresponding weight, and the parameters for the histogram, 
    return the best fit parameters for a power law.
    
    Note: This function could be much more modular. But for now I keep it as it is.
    """
    # Computing the histogram.
    bins = np.arange(fmin, fmax+bw/2., bw)
    hist, bin_edges = np.histogram(flux, weights=weight, bins=bins)

    # Compute the median magnitude
    fmed = np.median(flux)

    # Compute bin centers. Left set and right set.
    bin_centers = (bin_edges[:-1]+bin_edges[1:])/2.
    ileft = bin_centers < fmed
    # left and right counts
    left_hist = hist[ileft]
    right_hist = hist[~ileft]
    # left and right flux
    left_f = bin_centers[ileft]
    right_f = bin_centers[~ileft]

    # Place holder for the best parameters
    best_params_pow = np.zeros(2,dtype=np.float) 

    # Empty list for the negative log-likelihood
    list_nloglike = []
    best_nloglike = -np.inf # -100

    # Define negative total loglikelihood function given the histogram.
    def ntotal_loglike_pow(params):
        """
        Total log likelihood.
        """
        total_loglike = 0

        for i in range(bin_centers.size):
            total_loglike += stats.poisson.logpmf(hist[i].astype(int), pow_law(params, bin_centers[i]) * bw * area)

        return -total_loglike

    # fit for niter times 
    counter = 0
    while counter < niter:
        # Generate initial parameters
        init_params = pow_param_init_dNdf(left_hist, left_f, right_hist, right_f, bw, area)

        # Optimize the parameters.
        res = opt.minimize(ntotal_loglike_pow, init_params,tol=pow_tol,method="Nelder-Mead" )
        counter+=1
        if counter % 2 == 0:
            print(counter)
        if res["success"]:
            fitted_params = res["x"]

            # Calculate the negative total likelihood
            nloglike = ntotal_loglike_pow(fitted_params)
            list_nloglike.append(nloglike)

            # If loglike is the highest among seen, then update the parameters.
            if nloglike > best_nloglike:
                best_nloglike = nloglike
                best_params_pow = fitted_params
        else:
            print "Optimization failed."

#     print(best_params_pow)

    return best_params_pow

    

def mag2flux(mag):
    return 10**(0.4*(22.5-mag))
        




def combine_grz(list1,list2,list3=None):
    """
    Convenience function for combining two or three sets data in a list.
    """
    if list3 is not None:
        g = np.concatenate((list1[0], list2[0], list3[0]))
        r = np.concatenate((list1[1], list2[1], list3[1]))
        z = np.concatenate((list1[2], list2[2], list3[2]))
    else:
        g = np.concatenate((list1[0], list2[0]))
        r = np.concatenate((list1[1], list2[1]))
        z = np.concatenate((list1[2], list2[2]))

    return [g, r,z]


def true_false_fraction(ibool):
    """
    Given boolean index array count true and false proportion and print.
    """
    counts = np.bincount(ibool)
    tot = np.sum(counts).astype(float)
    print("True: %d (%.4f)| False: %d (%.4f)" % (counts[1], counts[1]/tot, counts[0], counts[0]/tot))
    return [counts[1], counts[1]/tot, counts[0], counts[0]/tot]


def load_cn(fits):
    return fits["cn"].astype(int)


def load_DEEP2matched(table):
    return table["DEEP2_matched"][:]    


def window_mask(ra, dec, w_fname):
    """
    Given the ra,dec of objects and a window function file name, the function 
    returns an boolean array whose elements equal True when the corresponding 
    objects lie within regions where the map is positive.
    
    Note: the windowf.**.fits files were not set up in a convenient format 
    so below I had to perform a simple but tricky affine transformation. 
    """
    
    # Import the window map and get pixel limits.
    window = fits.open(w_fname)[0].data.T # Note that transpose.
    px_lim = window.shape[1]
    py_lim = window.shape[0]      
    
    # Creating WCS object for the window.  
    w = WCS(w_fname)
    
    # Convert ra/dec to pixel values and round.
    px, py = w.wcs_world2pix(ra, dec, 0)
    px_round = np.round(py).astype(int)
    py_round = np.round(px).astype(int)
  
    # Creating the array.
    idx = np.zeros(px_round.size, dtype=bool)
    for i in range(px.size):
        if (px_round[i]>=0) and (px_round[i]<px_lim) and (py_round[i]>=0) and (py_round[i]<py_lim): # Check if the object lies within the window frame. 
            if (window[py_round[i],px_round[i]]>0): # Check if the object is in a region where there is spectroscopy.
                idx[i] = True
    
    return idx


def est_spec_area(w_fname):
    """
    The following function estiamtes the spectroscopic area given a window 
    function file.
    """
    # Creating WCS object for the window.  
    w = WCS(w_fname)
    
    # Importing the window
    window = fits.open(w_fname)[0].data
    px_lim = window.shape[0]
    py_lim = window.shape[1]
   
    
    # Convert ra/dec to pixel values and round.
    ra, dec = w.wcs_pix2world([0,px_lim], [0,py_lim], 0)
    
    # Calculating the area
    area = (ra[1]-ra[0])*(dec[1]-dec[0])
    
    # Calculating the fraction covered by spectroscopy
    frac = (window>0).sum()/(px_lim*py_lim+1e-12)
    
    return frac*area
        
        
    
def import_zcat(z_fname):
    """
    Given DEEP2 redshift catalog filename, import and return relevant fields.
    """
    data = fits.open(z_fname)[1].data
    
    return data["OBJNO"], data["RA"], data["DEC"], data["OII_3727"], data["OII_3727_ERR"], data["ZHELIO"], data["ZHELIO_ERR"], data["ZQUALITY"], data["TARG_WEIGHT"]
    


def match_objno(objno1, objno2):
    """
    Given two objno arrays, return idx of items that match. This algorithm can be slow, O(N^2), but it should work.
    The input arrays have to be a set, meaning a list of unique items.
    """    
    global large_random_constant
    # Finding the intersection
    intersection = np.intersect1d(objno1, objno2)
    print("# of elements in intersection: %d"% intersection.size)
    
    # Creating placeholders for idx's to be returned.
    idx1 = np.ones(intersection.size,dtype=int)*large_random_constant
    idx2 = np.ones(intersection.size,dtype=int)*large_random_constant
    
    # Creating objno1, objno2 copies with integer tags before sorting.
    
    objno1_tagged = np.rec.fromarrays((objno1, range(objno1.size)),dtype=[('id', int), ('tag', int)])
    objno2_tagged = np.rec.fromarrays((objno2, range(objno2.size)),dtype=[('id', int), ('tag', int)])
    
    # Sorting according id
    objno1_tagged = np.sort(objno1_tagged, axis=0, order="id")
    objno2_tagged = np.sort(objno2_tagged, axis=0, order="id")
    
    # tags
    tags1 = objno1_tagged["tag"]
    tags2 = objno2_tagged["tag"]
    
    # values
    objno1_vals = objno1_tagged["id"]    
    objno2_vals = objno2_tagged["id"]
        
    # For each id in the intersection set, find the corresponding indices in objno1 and objno2 and save. 
    for i,e in enumerate(intersection):
        idx1[i] = tags1[np.searchsorted(objno1_vals,e)]
        idx2[i] = tags2[np.searchsorted(objno2_vals,e)]
    
    return idx1, idx2


def pcat_append(pcat, new_col, col_name, idx1, idx2):
    """
    Given DEEP2 pcat recarray and a pair of a field column, and a field name,
    append the new field column to the recarray using OBJNO-matched values and name the appended column the field name.
    Must provide appropriate idx values for both pcats and additional catalogs.
    """
    global large_random_constant
    new_col_sorted = np.ones(pcat.shape[0])*large_random_constant
    new_col_sorted[idx1] = new_col[idx2]
    
    new_pcat = rec.append_fields(pcat, col_name, new_col_sorted, dtypes=new_col_sorted.dtype, usemask=False, asrecarray=True)
    return new_pcat
    


def generate_class_col(pcat):
    """
    Given a pcat array with required fields, produce a column that classify objects into different classes.

    - Gold and Silver objects are any ELGs with a measured OII and a secured redshift. For convenience,
    I do not impose OII>0 condition.
    - NoOII are objects in DESI redshift range but no measured OII.
    - NoZ are DEEP2 color selected objects but no secure redshift information and no OII information.
    - NonELG: DEEP2 rejected, confirmed low-z objects with no OII flux measurement, or objects with no color selection information.
    - DEEP2_unobserved: Objects that were not targeted by DEEP2 survey.
    """
    # Extracting columns
    OII = pcat["OII_3727"]*1e17
    Z = pcat["RED_Z"]
    ZQUALITY = pcat["Z_QUALITY"]
    OII_ERR = pcat["OII_3727_ERR"]
    BRIcut = pcat["BRI_cut"].astype(int)
    
    # Placeholder for the class column.
    class_col = np.ones(pcat.shape[0],dtype=int)*large_random_constant
    
    # Gold, CN=0: Z in [1.1, 1.6]
    ibool = (Z>1.1) & (Z<1.6) & (BRIcut==1) & (ZQUALITY>=3) & (OII_ERR>0) & (OII > 0)
    class_col[ibool] = 0
    
    # Silver, CN=1: Z in [0.6, 1.1]
    ibool = (Z>0.6) & (Z<1.1) & (BRIcut==1) & (ZQUALITY>=3) & (OII_ERR>0) & (OII > 0)
    class_col[ibool] = 1

    # NoOII, CN=2: OII=?, Z in [0.6, 1.6] and secure redshift
    ibool = (Z>0.6) & (Z<1.6) & (BRIcut==1) & (ZQUALITY>=3)  & np.logical_or((OII_ERR<=0), OII <=0)
    class_col[ibool] = 2

    # NoZ, CN=3: OII=NA, Z undetermined.
    ibool = np.logical_or.reduce(((ZQUALITY==-2) , (ZQUALITY==0) , (ZQUALITY==1) ,(ZQUALITY==2))) & (BRIcut==1)  & (OII_ERR<=0)
    class_col[ibool] = 3
    
    # NonELG, CN=4. DEEP2 rejected, confirmed stars (ZQUALITY == -1), or confirmed other low-redshift (or Non-DESI redshift objects)
    ibool_lowZ = np.logical_or((np.logical_or((Z>1.6), (Z<0.6)) & (ZQUALITY>=3)),(ZQUALITY==-1))  & (OII_ERR<=0) & (BRIcut==1)
    ibool_reject = BRIcut==0
    ibool_NoInfo = BRIcut <-1000 # Objects with no color selection information.
    ibool = np.logical_or.reduce((ibool_lowZ, ibool_reject, ibool_NoInfo))
    class_col[ibool] = 4
    
    # DEEP2_unobserved, CN=5
    ibool = (BRIcut==1) & (ZQUALITY<-10)   # Objects that were not assigned color-selection flag are classifed as DEEP2 color rejected objects.
    class_col[ibool] = 5
    
    return class_col


def count_nn(arr):
    """
    Count the number of non-negative elements.
    """   
    return arr[arr>-1].size




def class_breakdown(fn, cn, weight, area, rwd="D"):
    """
    Given a list of class fields and corresponding weights and areas, return the breakdown of object 
    for each class. fn gives the field number. 
    
    R: Raw density
    W: Weighted Density
    """
    
    # Place holder for tallying
    counts = np.zeros(len(cnames))
    
    # Generate counts
    for i in range(len(fn)):
        # Computing counts
        if rwd == "R":
            tmp = generate_raw_breakdown(cn[i], area[i])
        elif rwd == "W":
            tmp = generate_density_breakdown(cn[i], weight[i], area[i])
        
        # Tallying counts
        if rwd in ["R", "W"]:
            counts += tmp
        else:
            counts += tmp/len(fn)
        
        # Printing counts
        print(str_counts(fn[i], tmp))            

    # Total or average counts
    if rwd in ["R", "W"]:
        # print(str_counts("Total", counts))
    # else:
        print(str_counts("Avg.", counts/len(fn)))


    
def str_counts(fn, counts):
    """
    Given the counts of various class of objects return a formated string.
    """
    if type(fn)==str:
        return_str = "%s " % (fn)
    else:
        return_str = "%d " % (fn)
        
    for i in range(counts.size):
        return_str += "& %d " % counts[i]
    
    return_str += "& %d " % np.sum(counts)
    
    return_str += latex_eol()
    
    return return_str
    
    
def generate_raw_breakdown(cn, area):
    counts = np.zeros(len(cnames), dtype=int)
    for i in range(len(cnames)-1):
        counts[i] = np.sum(cn==i)/float(area)
    return counts



def generate_density_breakdown(cn, weight,area):
    counts = np.zeros(len(cnames), dtype=int)
    for i in range(len(cnames)-1):
        if i != 4: # If the chosen class is not DEEP2 reject then 
            counts[i] = np.sum(weight[cn==i])/float(area)
        else:
            counts[i] = np.sum(cn==i)/float(area)
    return counts

    

def generate_table_header():
    return "Field & "+" & ".join(cnames) + " & Total" + latex_eol() + latex_hline()

def latex_eol():
    return "\\\\ \\hline"

def latex_hline():
    return "\\hline"
    



def return_bricknames(ra, dec, br_name, ra_range, dec_range,tol):
    ibool = (ra>(ra_range[0]-tol)) & (ra<(ra_range[1]+tol)) & (dec>(dec_range[0]-tol)) & (dec<(dec_range[1]+tol))
    return  br_name[ibool]






def combine_tractor(fits_directory):
    """
    Given the file directory, find all Tractor fits files combine them and return as a rec-array.
    """
    onlyfiles = [f for f in listdir(fits_directory) if isfile(join(fits_directory, f))]
    print("Number of files in %s %d" % (fits_directory, len(onlyfiles)-1))
    
    
    DR3 = None
    for i,e in enumerate(onlyfiles,start=1):
        # If the file ends with "fits"
        if e[-4:] == "fits":
            print("Combining file %d. %s" % (i,e))
            # If DR3 has been set with something.
            tmp_table = apply_mask(fits.open(fits_directory+e)[1].data)
            if DR3 is not None:
                DR3 = np.hstack((DR3, tmp_table))
            else:
                DR3 = tmp_table
                
    return DR3

def combine_tractor_nocut(fits_directory, all_models=False):
    """
    Given the file directory, find all Tractor fits files combine them and return as a rec-array.

    If all_models is True, search within fits_directory for directory called all-models to find corresponding metrics files.
    """
    onlyfiles = [f for f in listdir(fits_directory) if isfile(join(fits_directory, f))]
    print("Number of files in %s %d" % (fits_directory, len(onlyfiles)))
    
    
    DR3 = None
    for i,e in enumerate(onlyfiles, start=0):
        # If the file ends with "fits"
        if e[-4:] == "fits":
            print("Combining file %d. %s" % (i,e))
            if all_models: # If all_models is true, then combine the tractor file with the corresponding all-models file.
                orig_cols = fits.open(fits_directory+e)[1].data.columns
                tmp_table2 = fits.open(fits_directory+"all-models/all-models-"+e.split("tractor-")[1])[1].data
                new_cols = fits.ColDefs([
                    fits.Column(name='rex_shapeExp_r', format='E',
                        array=tmp_table2["rex_shapeExp_r"]),
                    fits.Column(name='rex_shapeExp_r_ivar', format='E',
                        array=tmp_table2["rex_shapeExp_r_ivar"])])
                hdu = fits.BinTableHDU.from_columns(orig_cols + new_cols)
                hdu.writeto('tmp_table.fits', clobber=True)
                # name = ''; format = 'E'
                # name = 'rex_shapeExp_r_ivar'; format = 'E'
                tmp_table = fits.open("tmp_table.fits")[1].data
            else:
                tmp_table = fits.open(fits_directory+e)[1].data

            if DR3 is not None:
                DR3 = np.hstack((DR3, tmp_table))
            else:
                DR3 = tmp_table
                
    return DR3

def category_vector_generator(z_quality, z_err, oii, oii_err, BRI_cut, cn):
    """
    Given the argument vectors, generate boolean vectors corresponding to
    ELG, NoZ, and NonELG classes
    """
    
    iELG = (z_quality>=3) & (z_err>0) & (oii>0) & (oii_err>0) & BRI_cut
    iNoZ = np.logical_or.reduce(((z_quality==-2) , (z_quality==0) , (z_quality==1) ,(z_quality==2)))  & (oii_err <=0) & BRI_cut

    # D2rejected group
    iNonELG0 = (cn==6)
    # Stars
    iNonELG1 = (z_quality == -1) & BRI_cut
    iNonELG = np.logical_or(iNonELG0, iNonELG1)
        
    return iELG, iNoZ, iNonELG



def inverse_cdf_2D(cvs, dV, PDF):
    """
    Given grid PDF, return the PDF values that correspond to probability mass values cvs.
    cv of 0.98 means there are 98% probability mass within the contour.

    This obviously assumes that PDF integrate to 1.

    Volume is provided by the user as dV.
    """
    
    pdf_sorted = np.sort(PDF.flatten())

    # Prob grid 
    prob_grid = np.arange(0, pdf_sorted[-1], pdf_sorted[-1]/float(1e3))
    
    # Calculating cumulative function (that is probability volume within the boundary)
    cdf = np.zeros(prob_grid.size)
    for i in range(cdf.size):
        cdf[i] = np.sum(pdf_sorted[pdf_sorted>prob_grid[i]]) * dV

    pReturn = []
    for i in range(len(cvs)):
        cv = cvs[i]
        pReturn.append(prob_grid[find_nearest_idx(cdf,cv)])

    return pReturn


        
def find_nearest_idx(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx    


def summed_gm_pdf(pts, mus, covs, amps):
    """
    pts: [Nsample, ND]

    Return PDF fo GMM specified by the parameters. 
    """

    func_val = None
    for i in range(amps.size):
        if func_val is None:
            func_val = amps[i]*stats.multivariate_normal.pdf(pts, mean=mus[i], cov=covs[i])    
        else:
            func_val += amps[i]*stats.multivariate_normal.pdf(pts, mean=mus[i], cov=covs[i])    

    return func_val


def plot_2D_contour_GMM(ax, Xrange, Yrange, amps, means, covs, var_num1, var_num2, levels=[0.95, 0.68], colors=["blue", "blue"]):
    """
    Given GMM parameters amps, means, covs of ND, plot on ax the cumulative contou at chosen levels.
    var_num1, var_num2 represents x and y variables for 2D plotting.

    Levels must be given from high to low. 
    Wrong: [0.02, 0.10, 0.50, 0.9, 0.98]
    Correct: [0.98, 0.9, 0.5, 0.10, 0.02]
    """
    # If the dimension of MoG is higher than 2,
    # find the 2D projections. 
    ND = means.shape[1] 
    if ND > 2:
        means_tmp = []
        covs_tmp = []
        for i in range(amps.size):
            # For each gaussian, find 2D projection
            mu = means[i]
            cov = covs[i]
            mu = np.array([mu[var_num1], mu[var_num2]])
            cov = np.array([[cov[var_num1, var_num1], cov[var_num1, var_num2]], [cov[var_num2, var_num1], cov[var_num2, var_num2]]])
            means_tmp.append(mu)
            covs_tmp.append(cov)
        means = np.asarray(means_tmp)
        covs = np.asarray(covs_tmp)

    # Plot isocontours
    xmin, xmax = Xrange
    ymin, ymax = Yrange
    dx = (xmax-xmin)/float(1e3)
    dy = (ymax-ymin)/float(1e3)
    Xvec = np.arange(xmin, xmax, dx)
    Yvec = np.arange(ymin, ymax, dy)
    X,Y = np.meshgrid(Xvec, Yvec) # grid of point
    X_flat, Y_flat = X.ravel(), Y.ravel()
    PDF = summed_gm_pdf(np.array([X_flat, Y_flat]).T, means, covs, amps) # * dX * dY # evaluation of the function on the grid
    cvs = levels # contour levels
    cvsP = inverse_cdf_2D(cvs, dx*dy, PDF)
    PDF = PDF.reshape(X.shape) # The same shape as X
    # assert False
    ax.contour(X, Y, PDF, cvsP, linewidths=2., colors=colors)

    return

def contour_plot_range(Xrange):
    # Unpack variables
    xmin, xmax = Xrange

    Delta_X = xmax - xmin

    return xmin-2*Delta_X, xmax+2*Delta_X

def gen_guide_line():
    """
    Line to guide eye.
    """
    # Two points 
    x1, y1 = 1, 0.45
    x2, y2 = 2, 1
    m = (y2-y1)/float(x2-x1) # slope
    b = (y2-m*x2) # intercept

    x = np.arange(-1.5, 5.5, 0.01)
    y = m*x+b
    return x, y


def make_corr_plots(ax_list, num_cat, num_vars, variables, lims, binws, var_names, weights=None,\
                    lines=None, pt_sizes=None, lw=1.5, lw_dot=1, ft_size=30, category_names = None,\
                    colors=None, ft_size_legend=15, hist_normed=False, alphas=None,\
                    plot_MoG1=False, amps1=None, means1=None, covs1=None, ND1=0, color_MoG1="blue",\
                    plot_MoG2=False, amps2=None, means2=None, covs2=None, ND2=0, color_MoG2="red",\
                   plot_MoG_general=False, var_num_tuple=None, amps_general=None, means_general=None, covs_general=None, color_general="red",\
                   cum_contour=False, plot_pow=False, pow_model=None, pow_var_num=None,\
                   hist_types=None, guide=False):
    """
    Add correlation plots for each variable pair to a given axis list. 
    Also, make marginalized plots in histogram.
    ax_list: Axis list of required dimension. 
    num_cat: Number of categories to deal with
    num_vars: Number of variables to plot
    variables: Nested list with num_cat lists of num_vars.
    lims: For each variable, specify min and max.
    binws: Binwidths for historgrams for each variable
    var_names: Names of the variables.
    weights: weights to assign when constructing histogram. A list of num_cat vectors. 
    lines: For each variable, draw dotted lines as specified.
    hist_normed: If True, all histograms are normalized.
    cum_contour: If true AND Plot_MoG_general=True, then 
        plot the cum_contour of the "general MoG" instead of the individual components.
    hist_types: A list of histtypes. If None is given, then use "step"

    Plot_MoG1,2:
    The function can also plot MoG specified by the user. amps, means, and covs are required.
    ND represents the dimension of the MoG. If ND=3, then model fit is in var1, var2, var3.
    Arbitrary order is not used by design. The number of components is automatically inferred.
    
    Plot_MoG_general: 
    Allows the user to input MoG fit in select variables specified by var_num_tuple. For example,
    var_num_tuple = (0, 1, 4), means a Gaussian mixture was fit in var1, var2, and var5.

    Plot_pow: 
    If True and if a power law model is supplied than a power law is overlayed on top of the histogram magnitude.
    
    Plot order: 
    - The last row (num_vars-1) is reserved for histogram for variables 1 ... num_vars-1.
    - The last column (num_vars-1) of the penultimate row (num_vars-2) is reserved for the vertical histogram of last variable.
    - This determines the unique order of plots. For example, for three variables.
    (x1, x2), (x2-hist-v)
    (x1, x3), (x2, x3), (x3-hist-v)
    (x1-hist), (x2-hist)
    - At the end, add auxilary lines

    If guide True, then plot a line that can guide the eye in bright orange solid line.
    """
    if guide: 
        x_guide, y_guide = gen_guide_line()

    # Disable panels that won't be used
    for i in range(0, num_vars-2):
        for j in range(i+2, num_vars):
            ax_list[i, j].axis('off')    
    ax_list[num_vars-1, num_vars-1].axis('off')  
    
    # construct a dictionary of what the plot type of each
    # For a given axis (row, col), the dictionary returns
    # plot-type, num_x, num_y
    # corr, x, y
    # hist, x
    # v-hist, x
    
    ax_dict = {}
    for i in range(num_vars): # row
        for j in range(num_vars): # col
            if i<num_vars-1: # If we haven't reached the last row
                if j<i+1:
                    ax_dict[(i, j)] = ("corr", j, i+1)
                elif j==i+1:
                    ax_dict[(i, j)] = ("v-hist", i+1,None)
                else: 
                    ax_dict[(i, j)] = (None,None,None)                    
            else: # for the last row
                if j<num_vars-1: # If we haven't reached the last column
                    ax_dict[(i, j)] = ("hist", j,None)
                else: 
                    ax_dict[(i, j)] = (None,None,None)                    
            
    
    # Plot each category
    for i in range(num_cat): 
        vars_tmp = variables[i]
        if weights is not None:
            w_tmp = weights[i]
        else:
            w_tmp = np.ones_like(vars_tmp[0])
        if colors is not None:
            color = colors[i]
        else:
            color = "black"
        if alphas is not None: 
            alpha = alphas[i]
        else:
            alpha = 1
        if category_names is not None:
            cname = category_names[i]
        else:
            cname = str(i)
        if pt_sizes is None:
            pt_size = 5
        else:
            pt_size = pt_sizes[i]
        if hist_types is not None:
            hist_type_tmp = hist_types[i]
            if hist_type_tmp == "step":
                alpha_hist = 1
            else:
                alpha_hist = 0.5
        else:
            hist_type_tmp = "step"
            alpha_hist = 1
            
        # Plot each axis
        for i in range(num_vars): # row
            for j in range(num_vars): # col
                plot_type, var_num1, var_num2 = ax_dict[(i, j)]
                if plot_type is not None:
                    if plot_type == "corr":
                        ax_list[i, j].scatter(vars_tmp[var_num1], vars_tmp[var_num2], s=pt_size, c=color, label=cname, edgecolor="none", alpha=alpha)
                        if plot_MoG1 and (var_num1<ND1) and (var_num2<ND1):
                            plot_cov_ellipse(ax_list[i, j], means1, covs1, var_num1, var_num2, MoG_color=color_MoG1)
                        if plot_MoG2 and (var_num1<ND2) and (var_num2<ND2):
                            plot_cov_ellipse(ax_list[i, j], means2, covs2, var_num1, var_num2, MoG_color=color_MoG2)
                        if plot_MoG_general and (var_num1 in var_num_tuple) and (var_num2 in var_num_tuple):
                            if cum_contour:
                                Xrange = contour_plot_range(lims[var_num1])
                                Yrange = contour_plot_range(lims[var_num2])
                                plot_2D_contour_GMM(ax_list[i, j], Xrange, Yrange, amps_general, means_general, covs_general, var_num_tuple.index(var_num1), var_num_tuple.index(var_num2))
                            else:
                                plot_cov_ellipse(ax_list[i, j], means_general, covs_general, var_num_tuple.index(var_num1), var_num_tuple.index(var_num2), MoG_color=color_general)
                        # If guide True and i,j = (0,0), then plot a line that can guide eye.
                        if guide and (i==0) and (j==0):
                            ax_list[i,j].plot(x_guide, y_guide, c="orange", lw = 2)

                    elif plot_type == "hist":
                        var_min, var_max = lims[var_num1]
                        bin_width = binws[var_num1]
                        hist_bins = np.arange(var_min, var_max+bin_width/2., bin_width)
                        if plot_pow and (var_num1 == pow_var_num):
                            ax_list[i, j].hist(vars_tmp[var_num1], bins=hist_bins, histtype=hist_type_tmp, alpha=alpha_hist, color=color, weights=w_tmp, lw=lw, label = cname)  
                        else:
                            ax_list[i, j].hist(vars_tmp[var_num1], bins=hist_bins, histtype=hist_type_tmp, alpha=alpha_hist, color=color, weights=w_tmp, lw=lw, label = cname, normed=hist_normed)                          
                        if plot_MoG1 and (var_num1<ND1):
                            plot_1D_gauss(ax_list[i, j], lims[var_num1], amps1, means1, covs1, var_num1, MoG_color=color_MoG1)
                        if plot_MoG2 and (var_num1<ND2):
                            plot_1D_gauss(ax_list[i, j], lims[var_num1], amps2, means2, covs2, var_num2, MoG_color=color_MoG2)                        
                        if plot_MoG_general and (var_num1 in var_num_tuple):
                            plot_1D_gauss(ax_list[i, j], lims[var_num1], amps_general, means_general, covs_general, var_num_tuple.index(var_num1), MoG_color=color_MoG2)
                        if plot_pow and (var_num1 == pow_var_num):
                            xvec = np.arange(var_min, var_max, 1e-3)
                            yvec = pow_law(pow_model, mag2flux(xvec)) * bin_width * dNdm2dNdf(xvec)
                            ax_list[i, j].plot(xvec,yvec, c = color_general, lw=2.)
                            
                    elif plot_type == "v-hist":
                        var_min, var_max = lims[var_num1]
                        bin_width = binws[var_num1]
                        hist_bins = np.arange(var_min, var_max+bin_width/2., bin_width)
                        if plot_pow and (var_num1 == pow_var_num):
                            ax_list[i, j].hist(vars_tmp[var_num1], bins=hist_bins, histtype=hist_type_tmp, alpha=alpha_hist, color=color, weights=w_tmp, lw=lw, label = cname, orientation="horizontal")  
                        else:
                            ax_list[i, j].hist(vars_tmp[var_num1], bins=hist_bins, histtype=hist_type_tmp, alpha=alpha_hist, color=color, weights=w_tmp, lw=lw, label = cname, orientation="horizontal", normed=hist_normed)  
                        if plot_MoG1 and (var_num1<ND1):
                            plot_1D_gauss(ax_list[i, j], lims[var_num1], amps1, means1, covs1, var_num1, vertical=False, MoG_color=color_MoG1)
                        if plot_MoG2 and (var_num1<ND2):
                            plot_1D_gauss(ax_list[i, j], lims[var_num1], amps2, means2, covs2, var_num2, vertical=False, MoG_color=color_MoG2)                        
                        if plot_MoG_general and (var_num1 in var_num_tuple):
                            plot_1D_gauss(ax_list[i, j], lims[var_num1], amps_general, means_general, covs_general, var_num_tuple.index(var_num1), MoG_color=color_MoG2, vertical=False) 
                        if plot_pow and (var_num1 == pow_var_num):
                            xvec = np.arange(var_min, var_max, 1e-3)
                            yvec = pow_law(pow_model, mag2flux(xvec)) * bin_width * dNdm2dNdf(xvec)
                            ax_list[i, j].plot(yvec, xvec, c = color_general, lw=2.)
                            

                    
            
     
    # Deocration
    for i in range(num_vars): # row
        for j in range(num_vars): # col
            plot_type, var_num1, var_num2 = ax_dict[(i, j)]
            if plot_type == "corr":            
                # axes_labels
                ax_list[i, j].set_xlabel(var_names[var_num1], fontsize=ft_size)            
                ax_list[i, j].set_ylabel(var_names[var_num2], fontsize=ft_size)
                ax_list[i, j].legend(loc="upper right", fontsize=ft_size_legend)
                # lines
                for vh_line in lines[var_num1]:
                    ax_list[i, j].axvline(x=vh_line, lw=lw_dot, c="green", ls="--")
                for vh_line in lines[var_num2]:
                    ax_list[i, j].axhline(y=vh_line, lw=lw_dot, c="green", ls="--")                    
                # axes_limits
                ax_list[i, j].set_xlim(lims[var_num1])                            
                ax_list[i, j].set_ylim(lims[var_num2])
                
            elif plot_type == "hist":
                # axes_labels
                ax_list[i, j].set_xlabel(var_names[var_num1], fontsize=ft_size)            
                ax_list[i, j].legend(loc="upper right", fontsize=ft_size_legend)
                # lines
                for vh_line in lines[var_num1]:
                    ax_list[i, j].axvline(x=vh_line, lw=lw_dot, c="green", ls="--")
                # axes_limits
                ax_list[i, j].set_xlim(lims[var_num1])                            
                
            elif plot_type == "v-hist":
                # axes_labels
                ax_list[i, j].set_ylabel(var_names[var_num1], fontsize=ft_size)            
                ax_list[i, j].legend(loc="upper right", fontsize=ft_size_legend)
                # lines
                for vh_line in lines[var_num1]:
                    ax_list[i, j].axhline(y=vh_line, lw=lw_dot, c="green", ls="--")
                # axes_limits
                ax_list[i, j].set_ylim(lims[var_num1])                                            
                                
    return ax_dict




def MoG1_mean_cov():
    """
    First component of MoGs
    """
    mean = np.array([ 1.38378524,  0.73595286,  1.,  3.24835027,  1.01934929]) # Fixed gflux mean by hand.
    cov = np.array([[ 0.26336862,  0.09897078,  0.00093518, -0.09504225, -0.03354253],
       [ 0.09897078,  0.04855362,  0.00229276, -0.03842319, -0.01763364],
       [ 0.00093518,  0.00229276,  0.05609987, -0.03871561, -0.00763744],
       [-0.09504225, -0.03842319, -0.03871561,  0.48560483,  0.01829135],
       [-0.03354253, -0.01763364, -0.00763744,  0.01829135,  0.03618361]])
    
    return mean, cov

def MoG2_mean_cov():
    """
    Second component of 2 component MoG
    """
    mean, cov = MoG1_mean_cov()
    mean = mean*np.array([1.5, 1.5, 1.05, 0.9, 1.1])
    cov *= 0.25
    cov += np.array([[ 0.03447977, -0.1071021 , -0.05680584,  0.2478375 ,  0.10232841],
       [ 0.08766291,  0.18576923, -0.04333036, -0.00487922, -0.03921066],
       [-0.09202787,  0.23828726,  0.10759185, -0.15039897, -0.04463796],
       [-0.09578853, -0.00231927, -0.01224208,  0.08104045, -0.19074252],
       [-0.09726527,  0.07063743,  0.13382492, -0.04058464,  0.1238597 ]])/10. 
    return mean, cov
    

def MoG1(Nsample):
    mean, cov = MoG1_mean_cov()    
    var_x, var_y, var_z, OII, redz = np.random.multivariate_normal(mean, cov, Nsample).T

    return var_x, var_y, var_z, OII, redz
    
def MoG2(Nsample):
    amps = [0.5, 0.5]
    var_x1, var_y1, var_z1, OII1, redz1 = MoG1(int(Nsample*amps[0]))
    
    # Second gaussian
    mean, cov = MoG2_mean_cov()    
    var_x2, var_y2, var_z2, OII2, redz2 = np.random.multivariate_normal(mean, cov, int(amps[1]*Nsample)).T
    
    return np.concatenate((var_x1, var_x2)), np.concatenate((var_y1, var_y2)), np.concatenate((var_z1, var_z2)), np.concatenate((OII1, OII2)), np.concatenate((redz1, redz2))


def gen_init_covar_from_data(Ndim, ydata, K):
    """
    Generate the covariance from the data. 
    Same covar matrix is used for all K components.
    """
    return np.asarray([np.cov(ydata).reshape((Ndim, Ndim))] * K)

def gen_flux_noise(Nsample, flim, sn=5):
    """
    Given the limiting flux and signal to noise, generate Nsample noise sample.
    """
    sig = flim/float(sn)
    return np.random.normal(0, sig, Nsample).T


def gen_err_seed(nsample, sigma=1, return_iw_factor=False):
    """
    If return_iw is True, then return importance weight as well as the error seeds. 
    The importance weights are computed assuming that the true target distribution is
    that of unit normal with zero mean whereas the proposal distribution uses the user
    provided sigma. Note that we return unnormalized weights since the normalization happens down stream.
    """
    if return_iw_factor:
        err_seed = np.random.normal(0, sigma, nsample)
        r_tilde = np.exp(-err_seed**2 * (1/2. - 1/float(2*sigma**2))) # p(x)/q(x), both un-normalized
        iw_factor = r_tilde # Do not use "/r_tilde.sum()" to normaliz
        return err_seed, iw_factor
    else:
        return np.random.normal(0, sigma, nsample)
    
    


def fit_GMM(ydata, ycovar, ND, K, Niter=1, weight=None):
    """
    Given the data matrix ydata [Nsample, ND], the error covariance matrix Ycovar [Nsample, [ND, ND]],
    fit K component MoG and return the fit parameters.
    
    Input: 
    - ND: Number of dimensions of the data points.
    - K: Number of components to fit.
    - Niter: Number of trials for the XD fit.
    - weight: Weight of data points.

    Output:
    - params: A dictionary that contains means, covars, and amps.
    """

    lnL_best = -np.inf # Initially lnL is infinitely terrible
    init_mean = None

    # Initialization of amps and covariance
    init_amp = gen_uni_init_amps(K)
    init_covar = gen_init_covar_from_data(ND, ydata.T, K)

    for j in range(Niter): # Number of trials
        print "Trial num: %d" % j
        # Initialization of means
        # Randomly pick K samples from the generated set.                
        init_mean_tmp = gen_init_mean_from_sample(ND, ydata, K)
        fit_mean_tmp, fit_amp_tmp, fit_covar_tmp = np.copy(init_mean_tmp), np.copy(init_amp), np.copy(init_covar)

        #Set up your arrays: ydata has the data, ycovar the uncertainty covariances
        #initamp, initmean, and initcovar are initial guesses
        #get help on their shapes and other options using
        # ?XD.extreme_deconvolution

        # XD fitting. Minimal regularization for the power law is used here. w=1e-4.
        lnL = XD.extreme_deconvolution(ydata, ycovar, fit_amp_tmp, fit_mean_tmp, fit_covar_tmp, w=1e-4, weight=weight)

        # Take the best result so far
        if lnL > lnL_best:
            fit_mean, fit_amp, fit_covar = fit_mean_tmp, fit_amp_tmp, fit_covar_tmp

    # Save the dictionary after fitting each model so that 
    # if the function crashses partial results are still saved.
    # Format example 
    params = {"means": fit_mean, "amps": fit_amp, "covs": fit_covar}
        
    return params

def apply_mask(table):
    """
    Given a tractor catalog table, apply the standard mask. brick_primary and flux inverse variance. 
    """
    brick_primary = load_brick_primary(table)
    givar, rivar, zivar = load_grz_invar(table)
    ibool = (brick_primary==True) & (givar>0) & (rivar>0) &(zivar>0) 
    table_trimmed = np.copy(table[ibool])

    return table_trimmed
    
def load_grz_anymask(fits):
    g_anymask = fits['DECAM_ANYMASK'][:][:,1]
    r_anymask = fits['DECAM_ANYMASK'][:][:,2]
    z_anymask = fits['DECAM_ANYMASK'][:][:,4]
    
    return g_anymask, r_anymask, z_anymask

def load_grz_allmask(fits):
    g_allmask = fits['DECAM_ALLMASK'][:][:,1]
    r_allmask = fits['DECAM_ALLMASK'][:][:,2]
    z_allmask = fits['DECAM_ALLMASK'][:][:,4]
    
    return g_allmask, r_allmask, z_allmask


def load_radec(fits):
    ra = fits["ra"][:]
    dec= fits["dec"][:]
    return ra, dec

def load_radec_ext(pcat):
    ra = pcat["RA_DEEP"]
    dec = pcat["DEC_DEEP"]    
    return ra, dec

def cross_match_catalogs(pcat, pcat_ref, tol=0.5):
    """
    Match pcat catalog to pcat_ref via ra and dec.
    Incorporate astrometric correction if any.
    """
    # Load radec
    ra, dec = load_radec_ext(pcat)
    ra_ref, dec_ref = load_radec_ext(pcat_ref)
    
    # Create spherematch objects
    c = SkyCoord(ra=ra*u.degree, dec=dec*u.degree)  
    c_ref = SkyCoord(ra=ra_ref*u.degree, dec=dec_ref*u.degree)  
    idx, idx_ref, d2d, d3d = c_ref.search_around_sky(c, 1*u.arcsec)
    
    # Find the median difference
    ra_med_diff = np.median(ra_ref[idx_ref]-ra[idx])
    dec_med_diff = np.median(dec_ref[idx_ref]-dec[idx])
    
    print("ra,dec discrepancy: %.3f, %.3f"%(ra_med_diff*3600, dec_med_diff*3600))
    
    # Finding matches again taking into account astrometric differnce.
    c = SkyCoord(ra=(ra+ra_med_diff)*u.degree, dec=(dec+dec_med_diff)*u.degree)  
    c_ref = SkyCoord(ra=ra_ref*u.degree, dec=dec_ref*u.degree)  
    idx, idx_ref, d2d, d3d = c_ref.search_around_sky(c, 1*u.arcsec)    
    
    return idx, idx_ref    


def load_brick_primary(fits):
    return fits['brick_primary']


def load_bid(fits):
    return fits['brickid']


def load_shape(fits):
    r_dev = fits['SHAPEDEV_R']
    r_exp = fits['SHAPEEXP_R']
    return r_dev, r_exp


def load_star_mask(table):
    return table["TYCHOVETO"][:].astype(int).astype(bool)

def load_oii(fits):
    return fits["OII_3727"]

def new_oii_lim(N_new, N_old=2400):
    """
    Return the new OII low threshold given the updated fiber number in units of
    1e-17 ergs/A/cm^2/s
    """
    return 8*np.sqrt(N_new/N_old)

def frac_above_new_oii(oii, weight, new_oii_lim):
    """
    Given the oii and weights of the objects of interest and the new OII limit, return
    the proportion of objects that meet the new criterion.
    """
    ibool = oii>new_oii_lim
    return weight[ibool].sum()/weight.sum()    


def load_fits_table(fname):
    """Given the file name, load  the first extension table."""
    return fits.open(fname)[1].data

def save_fits(data, fname):
    """
    Given a rec array and a file name (with "fits" filename), save it.
    """
    cols = fits.ColDefs(np.copy(data)) # This is somehow necessary.
    tbhdu = fits.BinTableHDU.from_columns(cols)
    tbhdu.writeto(fname, clobber=True)
    
    return 

def save_fits_join(data1,data2, fname):
    """
    Given a rec array and a file name (with "fits" filename), save it.
    """
    
    data = rec.merge_arrays((data1,data2), flatten=True, usemask=False,asrecarray=True)
    cols = fits.ColDefs(data) 
    tbhdu = fits.BinTableHDU.from_columns(cols)
    tbhdu.writeto(fname, clobber=True)
    
    return 

def load_weight(fits):
    return fits["TARG_WEIGHT"]    

def fits_append(table, new_col, col_name, idx1, idx2, dtype="default", dtype_user=None):
    """
    Given fits table and field column/name pair,
    append the new field to the table using the idx1 and idx2 that correspond to 
    fits table and new column indices.

    If dtype =="default", then the default of float variable type is used.
    If dtype =="user", then user provided data type is used.
    """
    global large_random_constant
    new_col_sorted = np.ones(table.shape[0])*large_random_constant
    new_col_sorted[idx1] = new_col[idx2]
    
    if dtype=="default":
        new_table = rec.append_fields(table, col_name, new_col_sorted, dtypes=new_col_sorted.dtype, usemask=False, asrecarray=True)
    else:
        new_table = rec.append_fields(table, col_name, new_col_sorted, dtypes=dtype_user, usemask=False, asrecarray=True)


    return new_table

def load_fits_table(fname):
    """Given the file name, load  the first extension table."""
    return fits.open(fname)[1].data
    

def apply_star_mask(fits):
    ibool = ~load_star_mask(fits) 
    
    return fits[ibool]

def load_grz_flux(fits):
    """
    Return raw (un-dereddened) g,r,z flux values.
    """
    g = fits['decam_flux'][:][:,1]
    r = fits['decam_flux'][:][:,2]
    z = fits['decam_flux'][:][:,4]
    
    return g,r,z

def load_grz_flux_dereddened(fits):
    # Colors: DECam model flux in ugrizY
    # mag = 22.5-2.5log10(f)
    g = fits['decam_flux'][:][:,1]/fits['decam_mw_transmission'][:][:,1]
    r = fits['decam_flux'][:][:,2]/fits['decam_mw_transmission'][:][:,2]
    z = fits['decam_flux'][:][:,4]/fits['decam_mw_transmission'][:][:,4]
    return g, r, z    

def load_grz_invar(fits):
    givar = fits['DECAM_FLUX_IVAR'][:][:,1]
    rivar = fits['DECAM_FLUX_IVAR'][:][:,2]
    zivar = fits['DECAM_FLUX_IVAR'][:][:,4]
    return givar, rivar, zivar

def load_grz(fits):
    # Colors: DECam model flux in ugrizY
    # mag = 22.5-2.5log10(f)
    g = (22.5 - 2.5*np.log10(fits['decam_flux'][:][:,1]/fits['decam_mw_transmission'][:][:,1]))
    r = (22.5 - 2.5*np.log10(fits['decam_flux'][:][:,2]/fits['decam_mw_transmission'][:][:,2]))
    z = (22.5 - 2.5*np.log10(fits['decam_flux'][:][:,4]/fits['decam_mw_transmission'][:][:,4]))
    return g, r, z


def load_W1W2_flux(fits):
    """
    Return raw (un-dereddened) w1, w2 flux values.
    """
    w1flux = fits["WISE_FLUX"][:][:,1]
    w2flux = fits["WISE_FLUX"][:][:,2]
    return w1flux, w2flux

def load_W1W2_fluxinvar(fits):
    w1_ivar = fits["WISE_FLUX_IVAR"][:][:,1]
    w2_ivar = fits["WISE_FLUX_IVAR"][:][:,2]
    return w1_ivar, w2_ivar

def load_W1W2(fits):
    # Colors: DECam model flux in ugrizY
    # mag = 22.5-2.5log10(f)
    w1 = (22.5 - 2.5*np.log10(fits['WISE_FLUX'][:][:,1]/fits['WISE_MW_TRANSMISSION'][:][:,0]))
    w2 = (22.5 - 2.5*np.log10(fits['WISE_FLUX'][:][:,2]/fits['WISE_MW_TRANSMISSION'][:][:,1]))
    return w1, w2




def load_redz(fits):
    """
    Return redshift
    """
    return fits["RED_Z"]

 

def reasonable_mask(table, decam_mask = "all", SN = True):
    """
    Given DECaLS table, return a boolean index array that indicates whether an object passed flux positivity, reasonable color range, and allmask conditions
    Impose SN>2 and decam_mask cut, only if it's requested.
    """
    grzflux = load_grz_flux(table)
    ibool = is_grzflux_pos(grzflux)
    
    grz = load_grz(table)
    ibool &= is_reasonable_color(grz) 
    
    if decam_mask == "all":
        grz_allmask = load_grz_allmask(table)
        ibool &= pass_grz_decammask(grz_allmask)
    elif decam_mask == "any":
        grz_anymask = load_grz_anymask(table)
        ibool &= pass_grz_decammask(grz_anymask)        
    
    if SN:
        grzivar = load_grz_invar(table)
        ibool &= pass_grz_SN(grzflux, grzivar, thres=2)

    return ibool

def pass_grz_SN(grzflux, grzivar, thres=2):
    gf, rf, zf = grzflux
    gi, ri, zi = grzivar
    
    return ((gf*np.sqrt(gi))>thres)&((rf*np.sqrt(ri))>thres)&((zf*np.sqrt(zi))>thres)

def grz_S2N(grzflux, grzinvar):
    g,r,z = grzflux
    gi,ri,zi = grzinvar
    return g*np.sqrt(gi),r*np.sqrt(ri),z*np.sqrt(zi)

def grz_flux_error(grzinvar):
    """
    Given the inverse variance return flux error.
    """
    gi,ri,zi = grzinvar
    return np.sqrt(1/gi),np.sqrt(1/ri),np.sqrt(1/zi)

def mag_depth_Xsigma(f_err, sigma=5):
    """
    Given flux error, return five sigma depth
    """
    return flux2mag(f_err*sigma)

def shift_flux_DR46_to_DR5(gflux, rflux, zflux):
    """
    Given grz fluxes of DR4 or DR6 data, apply transformation to
    bring fluxes to DR5 system. 
    """
    gflux5 = gflux * 10**(-0.4 * 0.029) * (gflux/rflux)**(-0.068)
    rflux5 = rflux * 10**(+0.4 * 0.012) * (rflux/zflux)**(-0.029)
    zflux5 = zflux * 10**(-0.4 * 0.000) * (rflux/zflux)**(+0.009)

    return glux5, rflux5, zflux5 


def median_mag_depth(f_err, sn=5):
    """
    Given list of flux errors compute median magnitude error.
    """
    return np.median(mag_depth_Xsigma(f_err, sn))    

def flux2mag(flux):
    return 22.5-2.5*np.log10(flux)    
    
def pass_grz_decammask(grz_decammask):
    gm, rm, zm = grz_decammask
    return (gm==0) & (rm==0) & (zm==0)

def is_reasonable_color(grz):
    """
    Given grz mag list, check whether colors lie within a reasonable range.
    """
    g,r,z = grz
    gr = g-r
    rz = r-z
    
    return (gr>-0.75) & (gr<2.5) & (rz>-0.5) &(rz<2.7)


def is_grzflux_pos(grzflux):
    """
    Given a list [gflux, rflux, zflux], return a boolean array that tells whether each object has all good fluxes or not.
    """
    ibool = (grzflux[0]>0) & (grzflux[1]>0) & (grzflux[2]>0)
    return ibool



def check_astrometry(ra1,dec1,ra2,dec2,pt_size=0.3):
    """
    Given two sets of ra/dec's return median difference in degrees.
    """
    ra_diff = ra2-ra1
    dec_diff = dec2-dec1
    ra_med_diff = np.median(ra_diff)
    dec_med_diff = np.median(dec_diff)
    return ra_med_diff, dec_med_diff



def crossmatch_cat1_to_cat2(ra1, dec1, ra2, dec2, tol=1./(deg2arcsec+1e-12)):
    """
    Return indices of cat1 (e.g., DR3) and cat2 (e.g., DEE2) cross matched to tolerance. 

    Note: Function used to cross-match DEEP2 and DR3 catalogs in each field 
    and test for any astrometric discrepancies. That is, for every object in 
    DR3, find the nearest object in DEEP2. For each DEEP2 object matched, 
    pick DR3 object that is the closest. The surviving objects after these 
    matching process are the cross-matched set.
    """
    
    # Match cat1 to cat2 using astropy functions.
    idx_cat1_to_cat2, d2d = match_cat1_to_cat2(ra1, dec1, ra2, dec2)
    
    # Indicies of unique cat2 objects that were matched.
    cat2matched = np.unique(idx_cat1_to_cat2)
    
    # For each cat2 object matched, pick cat1 object that is the closest. 
    # Skip if the closest objects more than tol distance away.
    idx1 = [] # Place holder for indices
    idx2 = []
    tag = np.arange(ra1.size,dtype=int)
    for e in cat2matched:
        ibool = (idx_cat1_to_cat2==e)
        candidates = tag[ibool]
        dist2candidates = d2d[ibool]
        # Index of the minimum distance cat1 object
        if dist2candidates.min()<tol:
            idx1.append(candidates[np.argmin(dist2candidates)])
            idx2.append(e)
    
    # Turning list of indices into numpy arrays.
    idx1 = np.asarray(idx1)
    idx2 = np.asarray(idx2)
    
    # Return the indices of cat1 and cat2 of cross-matched objects.
    return idx1, idx2



def match_cat1_to_cat2(ra1, dec1, ra2, dec2):
    """
    "c = SkyCoord(ra=ra1*u.degree, dec=dec1*u.degree)  
    catalog = SkyCoord(ra=ra2*u.degree, dec=dec2*u.degree)  
    idx, d2d, d3d = c.match_to_catalog_sky(catalog)  

    idx are indices into catalog that are the closest objects to each of the coordinates in c, d2d are the on-sky distances between them, and d3d are the 3-dimensional distances." -- astropy documentation.  

    Fore more information: http://docs.astropy.org/en/stable/coordinates/matchsep.html#astropy-coordinates-matching 
    """    
    cat1 = SkyCoord(ra=ra1*u.degree, dec=dec1*u.degree)  
    cat2 = SkyCoord(ra=ra2*u.degree, dec=dec2*u.degree)  
    idx, d2d, d3d = cat1.match_to_catalog_sky(cat2)
    
    return idx, d2d.degree

def closest_idx(arr, val):
    return np.argmin(np.abs(arr-val))   




##############################################################################
# The following is adpated from the URL indicated below.
# """
#     ImagingLSS
#     https://github.com/desihub/imaginglss/blob/master/imaginglss/analysis/tycho_veto.py

#     veto objects based on a star catalogue.
#     The tycho vetos are based on the email discussion at:
#     Date: June 18, 2015 at 3:44:09 PM PDT
#     To: decam-data@desi.lbl.gov
#     Subject: decam-data Digest, Vol 12, Issue 29
#     These objects takes a decals object and calculates the
#     center and rejection radius for the catalogue in degrees.
#     Note : The convention for veto flags is True for 'reject',
#     False for 'preserve'.

#     apply_tycho takes the galaxy catalog and appends a Tychoveto column
#     the code works fine for ELG and LRGs. For other galaxy type, you need to adjust it!
# """

# Import modules
import sys

def BOSS_DR9(tycho):
    bmag = tycho['BMAG']
    # BOSS DR9-11
    b = bmag.clip(6, 11.5)
    R = (0.0802 * b ** 2 - 1.86 * b + 11.625) / 60. #
    return R

def DECAM_LRG(tycho):
    vtmag = tycho['VTMAG']
    R = 10 ** (3.5 - 0.15 * vtmag) / 3600.
    return R

DECAM_ELG = DECAM_LRG

def DECAM_QSO(tycho):
    vtmag = tycho['VTMAG']
    # David Schlegel recommends not applying a bright star mask
    return vtmag - vtmag

def DECAM_BGS(tycho):
    vtmag = tycho['VTMAG']
    R = 10 ** (2.2 - 0.15 * vtmag) / 3600.
    return R

def radec2pos(ra, dec):
    """ converting ra dec to position on a unit sphere.
        ra, dec are in degrees.
    """
    pos = np.empty(len(ra), dtype=('f8', 3))
    ra = ra * (np.pi / 180)
    dec = dec * (np.pi / 180)
    pos[:, 2] = np.sin(dec)
    pos[:, 0] = np.cos(dec) * np.sin(ra)
    pos[:, 1] = np.cos(dec) * np.cos(ra)
    return pos

def tycho(filename):
    """
    read the Tycho-2 catalog and prepare it for the mag-radius relation
    """
    dataf = fits.open(filename)
    data = dataf[1].data
    tycho = np.empty(len(data),
        dtype=[
            ('RA', 'f8'),
            ('DEC', 'f8'),
            ('VTMAG', 'f8'),
            ('VMAG', 'f8'),
            ('BMAG', 'f8'),
            ('BTMAG', 'f8'),
            ('VARFLAG', 'i8'),
            ])
    tycho['RA'] = data['RA']
    tycho['DEC'] = data['DEC']
    tycho['VTMAG'] = data['MAG_VT']
    tycho['BTMAG'] = data['MAG_BT']
    vt = tycho['VTMAG']
    bt = tycho['BTMAG']
    b = vt - 0.09 * (bt - vt)
    v = b - 0.85 * (bt - vt)
    tycho['VMAG']=v
    tycho['BMAG']=b
    dataf.close()
    return tycho


def txts_read(filename):
    obj = np.loadtxt(filename)
    typeobj = np.dtype([
              ('RA','f4'), ('DEC','f4'), ('COMPETENESS','f4'),
              ('rflux','f4'), ('rnoise','f4'), ('gflux','f4'), ('gnoise','f4'),
              ('zflux','f4'), ('znoise','f4'), ('W1flux','f4'), ('W1noise','f4'),
              ('W2flux','f4'), ('W2noise','f4')
              ])
    nobj = obj[:,0].size
    data = np.zeros(nobj, dtype=typeobj)
    data['RA'][:] = obj[:,0]
    data['DEC'][:] = obj[:,1]
    data['COMPETENESS'][:] = obj[:,2]
    data['rflux'][:] = obj[:,3]
    data['rnoise'][:] = obj[:,4]
    data['gflux'][:] = obj[:,5]
    data['gnoise'][:] = obj[:,6]
    data['zflux'][:] = obj[:,7]
    data['znoise'][:] = obj[:,8]
    data['W1flux'][:] = obj[:,9]
    data['W1noise'][:] = obj[:,10]
    data['W2flux'][:] = obj[:,11]
    data['W2noise'][:] = obj[:,12]
    #datas = np.sort(data, order=['RA'])
    return data

def veto(coord, center, R):
    """
        Returns a veto mask for coord. any coordinate within R of center
        is vet.
        Parameters
        ----------
        coord : (RA, DEC)
        center : (RA, DEC)
        R     : degrees
        Returns
        -------
        Vetomask : True for veto, False for keep.
    """
    from sklearn.neighbors import KDTree
    pos_stars = radec2pos(center[0], center[1])
    R = 2 * np.sin(np.radians(R) * 0.5)
    pos_obj = radec2pos(coord[0], coord[1])
    tree = KDTree(pos_obj)
    vetoflag = ~np.zeros(len(pos_obj), dtype='?')
    arg = tree.query_radius(pos_stars, r=R)
    arg = np.concatenate(arg)
    vetoflag[arg] = False
    return vetoflag



def apply_tycho(objgal, tychofn,galtype='LRG'):
    # reading tycho star catalogs
    tychostar = tycho(tychofn)
    #
    # mag-radius relation
    #
    if galtype == 'LRG' or galtype == 'ELG':    # so far the mag-radius relation is the same for LRG and ELG
        radii = DECAM_LRG(tychostar)
    else:
        sys.exit("Check the apply_tycho function for your galaxy type")
    #
    #
    # coordinates of Tycho-2 stars
    center = (tychostar['RA'], tychostar['DEC'])
    #
    #
    # coordinates of objects (galaxies)
    coord = (objgal['ra'], objgal['dec'])
    #
    #
    # a 0.0 / 1.0 array (1.0: means the object is contaminated by a Tycho-2 star, so 0.0s are good)
    tychomask = (~veto(coord, center, radii)).astype('f4')
    objgal = rec.append_fields(objgal, ['TYCHOVETO'], data=[tychomask], dtypes=tychomask.dtype, usemask=False)
    return objgal

def apply_tycho_radec(ra, dec, tychofn,galtype='LRG'):
    """
    Return tycho mask given ra, dec of objects.
    """
    # reading tycho star catalogs
    tychostar = tycho(tychofn)
    #
    # mag-radius relation
    #
    if galtype == 'LRG' or galtype == 'ELG':    # so far the mag-radius relation is the same for LRG and ELG
        radii = DECAM_LRG(tychostar)
    else:
        sys.exit("Check the apply_tycho function for your galaxy type")
    #
    #
    # coordinates of Tycho-2 stars
    center = (tychostar['RA'], tychostar['DEC'])

    # coordinates of objects (galaxies)
    coord = (ra, dec)
    #
    #
    # a 0.0 / 1.0 array (1.0: means the object is contaminated by a Tycho-2 star, so 0.0s are good)
    tychomask = (~veto(coord, center, radii)).astype('f4')
    return tychomask


def apply_tycho_pcat(objgal, tychofn,galtype='LRG'):
    # reading tycho star catalogs
    tychostar = tycho(tychofn)
    #
    # mag-radius relation
    #
    if galtype == 'LRG' or galtype == 'ELG':    # so far the mag-radius relation is the same for LRG and ELG
        radii = DECAM_LRG(tychostar)
    else:
        sys.exit("Check the apply_tycho function for your galaxy type")
    #
    #
    # coordinates of Tycho-2 stars
    center = (tychostar['RA'], tychostar['DEC'])
    #
    #
    # coordinates of objects (galaxies)
    coord = (objgal['RA_DEEP'], objgal['DEC_DEEP'])
    #
    #
    # a 0.0 / 1.0 array (1.0: means the object is contaminated by a Tycho-2 star, so 0.0s are good)
    tychomask = (~veto(coord, center, radii)).astype('f4')
    objgal = rec.append_fields(objgal, ['TYCHOVETO'], data=[tychomask], dtypes=tychomask.dtype, usemask=False)
    return objgal


def broad_cut(g, r, z):
    """
    Given grz-magnitude, return boolean array for selecting objects in the design space.
    Note that there is no magnitude cut here.
    """
    ygr = g-r
    xrz = r-z
    return np.logical_and((ygr < 0.8), np.logical_or((xrz > (0.7*ygr+ 0.2)), (ygr < 0.2)))
    




def gen_init_mean_from_sample(Ndim, sample, K):
    """
    Ndim: Dimensionality
    sample [Nsample, Ndim]: Sample array
    K: Number of components
    """
    idxes = np.random.choice(range(sample.shape[0]), K, replace=False)
    init_mean = []
    for idx in idxes:
        init_mean.append(sample[idx])
    return np.asarray(init_mean)

def gen_uni_init_amps(K):
    """
    K: Number of components
    """
    return np.asarray([1/float(K)]*K)

def gen_diag_init_covar(Ndim, var, K):
    """
    Genereate diagonal covariance matrices given var [Ndim]
    Ndim: Dimensionality
    K: Number of components
    """
    return np.asarray([np.diag(var)]*K)
    
def gen_diag_data_covar(Nsample, var):
    """
    Generate covar matrix for each of Nsample data points
    given diagonal variances of Ndim
    """
    return np.asarray([np.diag(var)]*Nsample)



def cov_ellipse(cov, q=None, nsig=None, **kwargs):
    """
    Parameters
    ----------
    cov : (2, 2) array
        Covariance matrix.
    q : float, optional
        Confidence level, should be in (0, 1)
    nsig : int, optional
        Confidence level in unit of standard deviations. 
        E.g. 1 stands for 68.3% and 2 stands for 95.4%.

    Returns
    -------
    width, height, rotation :
         The lengths of two axises and the rotation angle in degree
    for the ellipse.
    """

    if q is not None:
        q = np.asarray(q)
    elif nsig is not None:
        q = 2 * norm.cdf(nsig) - 1
    else:
        raise ValueError('One of `q` and `nsig` should be specified.')
    r2 = chi2.ppf(q, 2)

    val, vec = np.linalg.eigh(cov)
    width, height = 2 * np.sqrt(val[:, None] * r2)
    rotation = np.degrees(np.arctan2(*vec[::-1, 0]))

    return width, height, rotation


def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

def gen_comb(ND, N_choose):
    """
    ND: Number of variables. 
    N_choose: Number of variables to choose
    
    -----
    Output: A list of all possible combinations to use without redundant entry.
    
    -----
    Overview:
    - Start with 0. Next you can choose a number from 1, ..., ND-1. 

    """
    arrays = [range(ND)] * N_choose

    e_list = []
    for e in cartesian(arrays):
        e_list.append(list(np.sort(e)))

    len_orig = len(e_list)
    new_len = len_orig

    # Eliminate list-level duplicate
    i=0
    while i < new_len:
        e_current = e_list[i]

        # Index of objects to be deleted
        idx_del_list = []

        for j in range(i+1, new_len):
            if (e_list[j] == e_current):
                idx_del_list.append(j)
        num_del = len(idx_del_list)
        for j in range(num_del-1, -1, -1):
            del e_list[idx_del_list[j]]

        # Update the length of e_list
        new_len -= num_del

        # Update the index 
        i+=1

    # Eliminate lists with element-level duplicates 
    i=0
    while i < new_len:
        e_current = e_list[i]

        # Index of objects to be deleted
        idx_del_list = []

        for j in range(i, new_len):
            if (len(set(e_list[j]))!=N_choose):
                idx_del_list.append(j)
        num_del = len(idx_del_list)
        for j in range(num_del-1, -1, -1):
            del e_list[idx_del_list[j]]

        # Update the length of e_list
        new_len -= num_del

        # Update the index 
        i+=1
        
    return e_list

def plot_1D_gauss(ax, xlims, amps, means, covs, var_num, vertical=True, MoG_color="blue"):
    NK = amps.size
    xmin, xmax = xlims[0], xlims[1]
    dx = (xmax-xmin)/1000.
    xgrid = np.arange(xmin, xmax+dx/2., dx)
    gauss = None
    for k in range(NK):
        if gauss is None:
            gauss = amps[k] * stats.norm.pdf(xgrid, loc=means[k][var_num], scale=np.sqrt(covs[k][var_num, var_num]))
            gauss_tmp = gauss
        else:
            gauss_tmp = amps[k] * stats.norm.pdf(xgrid, loc=means[k][var_num], scale=np.sqrt(covs[k][var_num, var_num]))
            gauss += gauss_tmp
        if vertical:
            ax.plot(xgrid, gauss_tmp, lw=1, alpha=0.5, c=MoG_color)            
        else:
            ax.plot(gauss_tmp, xgrid, lw=1, alpha=0.5, c=MoG_color)            
    if vertical:
        ax.plot(xgrid, gauss, lw=2, alpha=1., c=MoG_color)        
    else:
        ax.plot(gauss, xgrid, lw=2, alpha=1., c=MoG_color)
        
    return

def plot_cov_ellipse(ax, mus, covs, var_num1, var_num2, MoG_color="Blue"):
    N_ellip = len(mus)
    for i in range(N_ellip):
        cov = covs[i]
        cov = [[cov[var_num1, var_num1], cov[var_num1, var_num2]], [cov[var_num2, var_num1], cov[var_num2, var_num2]]]
        mu = mus[i]
        mu = [mu[var_num1], mu[var_num2]]
        for j in [1, 2]:
            width, height, theta = cov_ellipse(cov, q=None, nsig=j)
            e = Ellipse(xy=mu, width=width, height=height, angle=theta, lw=1.25)
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            e.set_alpha(1)
            e.set_facecolor("none")
            e.set_edgecolor(MoG_color)

    return

def gen_gauss_kernel_3D(N):
    """
    Create a gaussian kernel of size for use in kernel approximation.
    If ND = 3, sig = 1/3. 
    """
    assert N in [3, 5, 7, 9, 11]
    if N == 3:
        sig = 1/3.
    elif N==5:
        sig = 3/5.
    elif N==7:
        sig = 4/5.
    elif N==9:
        sig = 1.
    else:
        sig = 1.25
    
    shape = (N,) * 3
    a = np.zeros(shape)
    a[(N/2,) * 3] = 1
    gauss_kernel = np.zeros(shape)
    gaussian_filter(a, sigma=sig, order=0, output=gauss_kernel, mode="constant", cval=0.0, truncate=4.0)
    
    return gauss_kernel



def load_DR5_calibration():
    """
    Note g-magnitude cut. 

    The calibration files were prepared by choosing four sweep files from DR5 
    catalogs and trimming (g< 24).
    """
    A = np.sum(np.load("../data/DR5/calibration/DR5-calibration-sweeps-areas.npy"))
    data = np.load("../data/DR5/calibration/DR5-calibration-sweeps-glim24.npz")
    g = data["g"]    
    r = data["r"]
    z = data["z"]
    w1 = data["w1"]
    w2  = data["w2"]

    return g, r, z, w1, w2, A
