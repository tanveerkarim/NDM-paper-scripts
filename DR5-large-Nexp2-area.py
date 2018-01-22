import numpy as np
import matplotlib.pylab as plt
import astropy.io.fits as fits 
from xd_elg_utils import *


def hr2deg(hr):
    return hr * 15

def radec_angular_distance(ra1, dec1, ra2, dec2, indeg = True):
    """
    Assume that angles are provided in degrees to begin with.
    """
    if indeg:
        deg2rad = np.pi/180.        
        ra1, dec1, ra2, dec2 = ra1*deg2rad, dec1*deg2rad, ra2*deg2rad, dec2*deg2rad
    
    return np.arccos(np.sin(dec1)*np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(ra1-ra2)) / deg2rad


def wrap_ra(ra, new_break_point = 180):
    """
    Given ra [0, 360], wrap it such that the resulting objects have ra [-(360-new_break_point), new_break_point]
    """
    ibool = ra>new_break_point
    ra[ibool] = ra[ibool]-360.

    return ra
tycho_directory = "/Users/jaehyeon/Documents/Research/ELG_target_selection/data-repository/"







print "Load ccdtosky output file."
# Load in hpix file
data = np.load("/Users/jaehyeon/Documents/Research/DESI-angular-clustering/ccdtosky/outputs/DR5/decals_Nside11/output_arr_chunk0thru50331648.npy")
ra, dec = data["hpix_ra"], data["hpix_dec"]
# ra = wrap_ra(ra) # Hopefully this works with Tycho mask below!
Ng, Nr, Nz = data["g_Nexp_sum"], data["r_Nexp_sum"], data["z_Nexp_sum"]
N_pix = ra.size# 
iNexp_cut = (Ng >=2) & (Nr >=2) & (Nz >=2)

# # Apply further tycho constraint    
# print "Generate tycho mask." # (iTycho True is good.)
# iTycho = apply_tycho_radec(ra, dec, tycho_directory+"tycho2.fits", galtype="ELG") == 0
# print "\n"


ra_Nexp2 = ra[iNexp_cut]
dec_Nexp2 = dec[iNexp_cut]


fig, ax  = plt.subplots(1, figsize=(7, 7))

# idx = np.random.randint(low=0, high=ra.size-1, size=10000)
# ax1.scatter(ra[idx], dec[idx], c="black", edgecolor="none", s=2)
# ax1.axis("equal")

idx = np.random.randint(low=0, high=ra_Nexp2.size-1, size=10000)
ax.scatter(ra_Nexp2[idx], dec_Nexp2[idx], c="black", edgecolor="none", s=2)
ax.plot(np.array([-5, 5, 5, -5, -5])+30, [-5, -5, 5, 5, -5], c="red", lw = 2)
ax.plot(np.array([-5, 5, 5, -5, -5])+150, [-5, -5, 5, 5, -5], c="red", lw = 2)
ax.plot(np.array([-5, 5, 5, -5, -5])+190, [-5, -5, 5, 5, -5], c="red", lw = 2)
ax.plot(np.array([-5, 5, 5, -5, -5])+220, np.array([-5, -5, 5, 5, -5])+12, c="red", lw = 2)
ax.axis("equal")
plt.savefig("DR5-calibration-sweeps-samples.png", dpi=200, bbox_inches="tight")
# plt.show()
plt.close()


import pandas as pd
data = pd.read_table("../data-repository/DR5/calibration-large-area-Nexp2/legacysurvey_dr5_sweep_5.0.sha256sum", delimiter=" ")
sweep_fname = data["sweep-000m005-010p000.fits"]
sweep_radec = [str(x).split("-")[1] for x in sweep_fname]

ra_list = []
dec_list = []
for x in sweep_radec:
    if "m" in x:
        a, b = x.split("m")
        a = float(a)
        b = -float(b)
        ra_list.append(a)
        dec_list.append(b)
    elif "p" in x:
        a, b = x.split("p")
        a = float(a)
        b = float(b)
        ra_list.append(a)
        dec_list.append(b)        
        
ra_sweep = np.asarray(ra_list)
dec_sweep = np.asarray(dec_list)



ra_min_list = np.array([25, 145, 185, 215])
ra_max_list = np.array([35, 155, 195, 225])
dec_min_list = np.array([-5, -5, -5, -2])
dec_max_list = np.array([5, 5, 5, 17])

ra_c_list = (ra_max_list + ra_min_list)/2.
dec_c_list = (dec_max_list + dec_min_list)/2.


tol = 5
sweep_list = []
for i in range(4):
    print i
    idx = (ra_sweep < ra_c_list[i]+tol) & (ra_sweep > ra_c_list[i]-tol) &\
    (dec_sweep > dec_c_list[i]-tol) & (dec_sweep < dec_c_list[i]+tol)
    
    sweep_tmp = []
    for x in sweep_fname[idx]:
        sweep_tmp.append(x)
        print x
    sweep_list.append(sweep_tmp)
    print "\n"





# Load each file
data_directory = "../data-repository/DR5/calibration-large-area-Nexp2/"
fnames = ["sweep-030p000-040p005.fits", "sweep-150p000-160p005.fits", "sweep-190p000-200p005.fits", "sweep-220p010-230p015.fits"]




# Plot ra/dec of regions chosen
fig, ax_list = plt.subplots(2, 2, figsize = (12, 12))

for i in range(2):
    for j in range(2):
        data = load_fits_table(data_directory+fnames[2*i+j])
        ra = data["ra"]
        dec = data["dec"]
        idx = np.random.randint(0, ra.size, size=5000)
        ax_list[i, j].scatter(ra[idx], dec[idx], edgecolor="none", s=2)
        ax_list[i, j].axis("equal")
        ax_list[i, j].set_xlabel("RA", fontsize=15)
        ax_list[i, j].set_ylabel("DEC", fontsize=15)        
        ax_list[i, j].set_title(fnames[2*i+j], fontsize=15)        
plt.savefig("DR5-calibration-sweeps.png", dpi=200, bbox_inches="tight")
# plt.show()
plt.close()




# From each file extracting columns of interest and saving as seperate files.
# - For each file, impose minimum conditions. Positive invar, allmask.
# - Reload the file and impose Tycho mask.
# - Compute the area that is not masked through Monte Carlo
# - Save the as well as columns of interest. 

# Plot ra/dec of regions chosen
fig, ax_list = plt.subplots(2, 2, figsize = (12, 12))

area_list = []
ra_combined, dec_combined, gflux, rflux, zflux, w1flux, w2flux = None, None, None, None, None, None, None

for k, fn in enumerate(fnames):
    print fn
    print "load data and impose minimum condition."
    data = load_fits_table(data_directory+fn)
    givar, rivar, zivar, w1ivar, w2ivar = data["flux_ivar_g"], data["flux_ivar_r"], data["flux_ivar_z"], data["flux_ivar_w1"], data["flux_ivar_w2"]
    gall, rall, zall = data["allmask_g"], data["allmask_r"], data["allmask_z"]
    ibool = (givar > 0) & (rivar > 0) & (zivar > 0) & (w1ivar > 0) & (w2ivar > 0) & (gall==0) & (rall==0)& (zall==0)
    data = data[ibool]
    
    print "Load ra, dec and impose Tycho mask"
    ra, dec = data["ra"], data["dec"]
    iTycho = apply_tycho_radec(ra, dec, tycho_directory+"tycho2.fits", galtype="ELG") == 0
    data = data[iTycho]
    
    print "Compute area"
    ra, dec = data["ra"], data["dec"]    
    ra_min, ra_max = ra.min(), ra.max()
    dec_min, dec_max = dec.min(), dec.max()
    # Generate a 10M random points
    Nsample = int(1e4)
    ra_MC = np.random.uniform(ra_min, ra_max, Nsample)
    dec_MC = np.random.uniform(dec_min, dec_max, Nsample)
    # Match within 10 arcsecond?
    idx1, idx2 = crossmatch_cat1_to_cat2(ra_MC, dec_MC, ra, dec, tol=30./(deg2arcsec+1e-12))
    area = (idx1.size/float(ra_MC.size)) * (ra_max-ra_min) * (dec_max-dec_min) * np.cos(np.pi * (dec_max+dec_min)/2. / 180.)
    area_list.append(area)    
    print "Area of the field: %.2f" % area
    

    print "Plotting"
    idx = np.random.randint(0, ra.size, size=5000)
    i = k // 2
    j = k % 2
    ax_list[i, j].scatter(ra[idx], dec[idx], edgecolor="none", s=2)
    ax_list[i, j].axis("equal")
    ax_list[i, j].set_xlabel("RA", fontsize=15)
    ax_list[i, j].set_ylabel("DEC", fontsize=15)        
    ax_list[i, j].set_title(fn, fontsize=15)        
    
    print "Extracting columns of interest"
    g = data["flux_g"]/data["mw_transmission_g"]    
    r = data["flux_r"]/data["mw_transmission_r"]    
    z = data["flux_z"]/data["mw_transmission_z"]        
    w1 = data["flux_w1"]/data["mw_transmission_w1"]
    w2 = data["flux_w2"]/data["mw_transmission_w2"]
    if gflux is None:
        ra_combined, dec_combined, gflux, rflux, zflux, w1flux, w2flux = ra, dec, g, r, z, w1, w2
    else:
        ra_combined = np.concatenate((ra_combined, ra))
        dec_combined = np.concatenate((dec_combined, dec))
        gflux = np.concatenate((gflux, g))
        rflux = np.concatenate((rflux, r))
        zflux = np.concatenate((zflux, z))
        w1flux = np.concatenate((w1flux, w1))
        w2flux = np.concatenate((w2flux, w2))
    print "\n"
print "Done!"
    
plt.savefig("DR5-calibration-sweeps-after-trimming.png", dpi=200, bbox_inches="tight")
# plt.show()
plt.close()    



arr = np.recarray((ra_combined.size,), \
             dtype=[('ra', float), ('dec', float), ('g', float), ('r', float), ('z', float), ('w1', float), ('w2', float)])

arr["ra"] = ra_combined
arr["dec"] = dec_combined
arr["g"] = gflux
arr["r"] = rflux
arr["z"] = zflux
arr["w1"] = w1flux
arr["w2"] = w2flux   

np.save("DR5-calibration-sweeps.npy", arr)
np.save("DR5-calibration-sweeps-areas.npy", area_list)