from utils import *

def load_DEEP2(fname, ibool=None):
    tbl = load_fits_table(fname)
    if ibool is not None:
        tbl = tbl[ibool]

    ra, dec = tbl["RA_DEEP"], tbl["DEC_DEEP"]
    tycho = tbl["TYCHOVETO"]
    B, R, I = tbl["BESTB"], tbl["BESTR"], tbl["BESTI"]
    cn = tbl["cn"]
    w = tbl["TARG_WEIGHT"]
    redz = tbl["RED_Z"]
    oii = tbl["OII_3727"] * 1e17
    zquality = tbl["ZQUALITY"]
    return ra, dec, tycho, B, R, I, cn, w, redz, oii, zquality

dir_derived = "../data/derived/"
dir_figure = "../figures/"
print("Import estimated areas")
areas = np.load(dir_derived+"spec-area.npy")    



# Based on the combined DEEP2 photometric and redshift catalogs, the target probability weighted properties of ELG in the three Fields should be comparable. Compare the following quantities: 
# redshift distribution, OII distribution, BRI-magnitude-color distribution, color-color distribution

field_colors = ["black", "red", "blue"]
ft_size = 15
ft_size2 = 20

# Fig 1: Redshift and OII comparison
plt.close()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 7))
redz_bins = np.arange(0, 2, 0.025)
oii_bins = np.arange(-2, 40, 0.5)
for i, fnum in enumerate([2, 3, 4]):
    fname = dir_derived+"deep2-f%d-photo-redz-oii.fits" % fnum
    ra, dec, tycho, B, R, I, cn, w, redz, oii, zquality = load_DEEP2(fname)
    ibool = (tycho==0) & np.logical_or.reduce((cn==0, cn==1))
    ra, dec, tycho, B, R, I, cn, w, redz, oii, zquality = load_DEEP2(fname, ibool)
    
    # Redshift
    ax1.hist(redz, bins=redz_bins, color=field_colors[i], histtype="step", alpha=0.9, lw=2.5, label="F%d" % fnum, weights=w/areas[i])
    ax1.set_xlim([0.5, 1.7])
    ax1.legend(loc="upper right", fontsize=ft_size)
    ax1.set_xlabel("Redshift z", fontsize=ft_size2)
    ax1.set_ylabel("dNd(0.025z)", fontsize=ft_size2)

    # OII
    ax2.hist(oii, bins=oii_bins, color=field_colors[i],  histtype="step", lw=2, label="F%d" % fnum, weights=w/areas[i])
    ax2.set_xlim([-2, 40])
    ax2.legend(loc="upper right", fontsize=ft_size)
    ax2.set_xlabel("OII * 1e17", fontsize=ft_size2)
    ax2.set_ylabel("dNd(0.5OII)", fontsize=ft_size2)
    
plt.savefig(dir_figure + "DEEP2-redz-oii-hist.png", dpi=200, bbox_inches="tight")
# plt.show()
plt.close()

BRI_names = ["B", "R", "I"]

# BRI magnitude distributions
plt.close()
fig, ax_list = plt.subplots(1, 3, figsize = (25, 7))
mag_bins = np.arange(19, 28, 0.1)
for j, fnum in enumerate([2, 3, 4]):
    fname = dir_derived+"deep2-f%d-photo-redz-oii.fits" % fnum
    ra, dec, tycho, B, R, I, cn, w, redz, oii, zquality = load_DEEP2(fname)
    ibool = (tycho==0) & np.logical_or.reduce((cn==0, cn==1))
    ra, dec, tycho, B, R, I, cn, w, redz, oii, zquality = load_DEEP2(fname, ibool)
    
    for i, mag in enumerate([B, R, I]):
        ax_list[i].hist(mag, bins=mag_bins, color=field_colors[j], histtype="step", alpha=0.9, lw=2.5, label="F%d" % fnum, weights=w/areas[j])
        ax_list[i].set_xlim([19, 28])
        ax_list[i].legend(loc="upper right", fontsize=ft_size)
        ax_list[i].set_xlabel(BRI_names[i], fontsize=ft_size2)
        ax_list[i].set_ylabel("dNd(0.1mag)", fontsize=ft_size2)

    
plt.savefig(dir_figure + "DEEP2-BRI-hist.png", dpi=200, bbox_inches="tight")
# plt.show()
plt.close()
