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
    gflux_raw, rflux_raw, zflux_raw = tbl["flux_g"], tbl["flux_r"], tbl["flux_z"]
    gflux, rflux, zflux = gflux_raw/tbl["mw_transmission_g"], rflux_raw/tbl["mw_transmission_r"],zflux_raw/tbl["mw_transmission_z"]
    givar, rivar, zivar = tbl["flux_ivar_g"], tbl["flux_ivar_r"], tbl["flux_ivar_z"]
    g_allmask, r_allmask, z_allmask = tbl["allmask_g"], tbl["allmask_r"], tbl["allmask_z"]
    objtype = tbl["type"]
    tycho = tbl["TYCHOVETO"]    
    
    return bid, objtype, tycho, bp, ra, dec, gflux_raw, rflux_raw, zflux_raw, gflux, rflux, zflux, givar, rivar, zivar, r_dev, r_exp, g_allmask, r_allmask, z_allmask


dir_derived = "../data/derived/"
dir_figure = "../figures/"
print "Import estimated areas"
areas = np.load(dir_derived+"spec-area.npy")    


field_colors = ["black", "red", "blue"]
ft_size = 15
ft_size2 = 20

grz_names = ["g", "r", "z"]

# BRI magnitude distributions
plt.close()
fig, ax_list = plt.subplots(1, 3, figsize = (25, 7))
mag_bins = np.arange(19, 26, 0.1)

glim = 24.25
for j, fnum in enumerate([2, 3, 4]):
    fname = dir_derived+"DR5-Tractor-D2f%d.fits" % fnum
    bid, objtype, tycho, bp, ra, dec, gflux_raw, rflux_raw, zflux_raw, gflux, rflux,\
    zflux, givar, rivar, zivar, r_dev, r_exp, g_allmask, r_allmask, z_allmask = load_tractor_DR5(fname)
    ibool = bp & (g_allmask==0) & (r_allmask==0) & (z_allmask==0) & (givar>0) & (rivar>0) & (zivar>0) & (tycho==0) & (gflux > mag2flux(glim))
    bid, objtype, tycho, bp, ra, dec, gflux_raw, rflux_raw, zflux_raw, gflux, rflux, zflux, givar, rivar, zivar, r_dev, r_exp, g_allmask, r_allmask, z_allmask = load_tractor_DR5(dir_derived + "DR5-Tractor-D2f%d.fits"%fnum, ibool=ibool)
    g, r, z = flux2mag(gflux), flux2mag(rflux), flux2mag(zflux)
    
    for i, mag in enumerate([g, r, z]):
        ax_list[i].hist(mag, bins=mag_bins, color=field_colors[j], histtype="step", alpha=0.9, lw=2.5, label="F%d" % fnum, weights=np.ones(ra.size)/areas[j])
        ax_list[i].set_xlim([19, 26])
        ax_list[i].legend(loc="upper right", fontsize=ft_size)
        ax_list[i].set_xlabel(grz_names[i], fontsize=ft_size2)
        ax_list[i].set_ylabel("dNd(0.1mag)", fontsize=ft_size2)

plt.savefig(dir_figure + "DR5-grz-hist.png", dpi=200, bbox_inches="tight")
# plt.show()
plt.close()
