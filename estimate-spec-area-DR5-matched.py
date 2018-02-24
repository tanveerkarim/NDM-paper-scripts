from utils import *


def load_tractor_DR5(fname, ibool=None):
    """
    Load select columns
    """
    tbl = load_fits_table(fname)
    if ibool is not None:
        tbl = tbl[ibool]

    ra, dec = load_radec(tbl)
    bid = tbl["brickid"]
    bp = tbl["brick_primary"]
    r_dev, r_exp = tbl["shapedev_r"], tbl["shapeexp_r"]
    gflux, rflux, zflux = tbl["flux_g"], tbl["flux_r"], tbl["flux_z"]
    gflux_raw, rflux_raw, zflux_raw = gflux/tbl["mw_transmission_g"], rflux/tbl["mw_transmission_r"],zflux/tbl["mw_transmission_z"]
    givar, rivar, zivar = tbl["flux_ivar_g"], tbl["flux_ivar_r"], tbl["flux_ivar_z"]
    g_allmask, r_allmask, z_allmask = tbl["allmask_g"], tbl["allmask_r"], tbl["allmask_z"]
    objtype = tbl["type"]
    tycho = tbl["TYCHOVETO"]    
    D2matched = tbl["DEEP2_matched"]
    
    return bid, objtype, tycho, bp, ra, dec, gflux_raw, rflux_raw, zflux_raw, gflux, rflux, zflux, givar, rivar, zivar, r_dev, r_exp, g_allmask, r_allmask, z_allmask, D2matched



dir_derived = "../data/derived/"
dir_figures = "../figures/"

lw=3
lw2=2
gmag_max = 25.

areas = []


print("Only consider objects that were matched in DEEP2 when estimating the area.")


figure, ax_list = plt.subplots(2, 3, figsize=(25,13))
for i, fnum in enumerate([2, 3, 4]):
    # DR5 data
    bid, objtype, tycho, bp, ra, dec, gflux_raw, rflux_raw, zflux_raw, gflux, rflux, zflux, givar, rivar, zivar, r_dev, r_exp,\
     g_allmask, r_allmask, z_allmask, D2matched = load_tractor_DR5(dir_derived+"DR5-matched-to-DEEP2-f%d-glim24p25.fits"%fnum)
    ibool = bp & (g_allmask==0) & (r_allmask==0) & (z_allmask==0) & (givar>0) & (rivar>0) & (zivar>0) & (tycho==0) & (gflux > mag2flux(gmag_max)) & (D2matched == 1)

    bid, objtype, tycho, bp, ra, dec, gflux_raw, rflux_raw, zflux_raw, gflux, rflux, zflux, givar, rivar, zivar, r_dev, r_exp,\
     g_allmask, r_allmask, z_allmask, D2matched = load_tractor_DR5(dir_derived+"DR5-matched-to-DEEP2-f%d-glim24p25.fits"%fnum)

    ra = ra[ibool]
    dec = dec[ibool]
    
    # Generate RA/DEC grid
    ra_range = ra.max()-ra.min()
    dec_range = -(dec.min()-dec.max())
    dec_median = np.median(dec * np.pi / float(180.))
    area = (ra.max()-ra.min()) * (dec.max()-dec.min()) * np.cos(dec_median * np.pi / float(180))
    NS = int(area * 2.5e6)

    xv = np.random.rand(NS) * ra_range + ra.min()
    yv = np.random.rand(NS) * dec_range + dec.min()
    

    
    # Spherematch with DEEP2
    # Match randoms to the pcat catalog. Make a cut in distance. 
    idx, d2d = match_cat1_to_cat2(xv, yv, ra, dec)
    imatched = d2d < 1/175.
    
    
    # Plot the matched and unmatched        
    # Matched
    ax_list[0, i].scatter(xv[imatched], yv[imatched], color="black", label="Matched", s=.25, edgecolor="none")
    ax_list[0, i].axis("equal")
    ax_list[0, i].set_xlabel("ra", fontsize=20)
    ax_list[0, i].set_ylabel("dec", fontsize=20)    
    ax_list[0, i].set_title("Field %d"%(i+2), fontsize=20)
    ax_list[0, i].legend(loc="upper right", fontsize=20)
    # Unmatched
    ax_list[1, i].scatter(xv[~imatched], yv[~imatched], color="black", label="Unmatched", s=.25, edgecolor="none")
    ax_list[1, i].axis("equal")
    ax_list[1, i].set_xlabel("ra", fontsize=20)
    ax_list[1, i].set_ylabel("dec", fontsize=20)        
    ax_list[1, i].set_title("Field %d"%(i+2), fontsize=20)    
    ax_list[1, i].legend(loc="upper right", fontsize=20)

    areas.append(area * (imatched.sum()/float(xv.size)))     # Note the cosine factor.

# plt.show()
plt.savefig(dir_figures+"estimate-area-monte-carlo-DR5-matched.png", dpi=400, bbox_inches="tight")
plt.close()

# Save area
print(areas, sum(areas))
np.save(dir_derived+"spec-area", areas)

