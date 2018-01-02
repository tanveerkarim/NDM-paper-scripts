from utils import *

dir_derived = "../data/derived/"
dir_figure = "../figures/"
print "Import estimated areas"
areas = np.load(dir_derived+"spec-area.npy")    

def load_tractor_DR5_matched_to_DEEP2_full(ibool=None):
    """
    Load select columns. From all fields.
    """
    tbl1 = load_fits_table(dir_derived + "DR5-matched-to-DEEP2-f2-glim24p25.fits")
    tbl2 = load_fits_table(dir_derived + "DR5-matched-to-DEEP2-f3-glim24p25.fits")    
    tbl3 = load_fits_table(dir_derived + "DR5-matched-to-DEEP2-f4-glim24p25.fits")

    tbl1_size = tbl1.size
    tbl2_size = tbl2.size
    tbl3_size = tbl3.size    
    field = np.ones(tbl1_size+tbl2_size+tbl3_size, dtype=int)
    field[:tbl1_size] = 2 # Ad hoc solution    
    field[tbl1_size:tbl1_size+tbl2_size] = 3 # Ad hoc solution
    field[tbl1_size+tbl2_size:] = 4 
    tbl = np.hstack([tbl1, tbl2, tbl3])
    if ibool is not None:
        tbl = tbl[ibool]
        field = field[ibool]

    ra, dec = load_radec(tbl)
    bid = tbl["brickid"]
    bp = tbl["brick_primary"]
    r_dev, r_exp = tbl["shapedev_r"], tbl["shapeexp_r"]
    gflux_raw, rflux_raw, zflux_raw = tbl["flux_g"], tbl["flux_r"], tbl["flux_z"]
    w1_raw, w2_raw = tbl["flux_w1"], tbl["flux_w2"]
    w1_flux, w2_flux = w1_raw/tbl["mw_transmission_w1"], w2_raw/tbl["mw_transmission_w2"]
    gflux, rflux, zflux = gflux_raw/tbl["mw_transmission_g"], rflux_raw/tbl["mw_transmission_r"],zflux_raw/tbl["mw_transmission_z"]
    mw_g, mw_r, mw_z = tbl["mw_transmission_g"], tbl["mw_transmission_r"], tbl["mw_transmission_z"]    
    givar, rivar, zivar = tbl["flux_ivar_g"], tbl["flux_ivar_r"], tbl["flux_ivar_z"]
    w1ivar, w2ivar = tbl["flux_ivar_w1"], tbl["flux_ivar_w2"]
    g_allmask, r_allmask, z_allmask = tbl["allmask_g"], tbl["allmask_r"], tbl["allmask_z"]
    objtype = tbl["type"]
    tycho = tbl["TYCHOVETO"]
    B, R, I = tbl["BESTB"], tbl["BESTR"], tbl["BESTI"]
    cn = tbl["cn"].astype(int)
    w = tbl["TARG_WEIGHT"]
    # Proper weights for NonELG and color selected but unobserved classes. 
    w[cn==5] = 0
    w[cn==4] = 1
    red_z, z_err, z_quality = tbl["RED_Z"], tbl["Z_ERR"], tbl["ZQUALITY"]
    oii, oii_err = tbl["OII_3727"]*1e17, tbl["OII_3727_ERR"]*1e17
    D2matched = tbl["DEEP2_matched"]
    BRI_cut = tbl["BRI_cut"].astype(int).astype(bool)
#     rex_expr, rex_expr_ivar = tbl["rex_shapeExp_r"], tbl["rex_shapeExp_r_ivar"]

    # error
    gf_err = np.sqrt(1./givar)/mw_g
    rf_err = np.sqrt(1./rivar)/mw_r
    zf_err = np.sqrt(1./zivar)/mw_z




    # Computing w1 and w2 err
    w1_err, w2_err = np.sqrt(1./w1ivar)/tbl["mw_transmission_w1"], np.sqrt(1./w2ivar)/tbl["mw_transmission_w2"]
        
    return bid, objtype, tycho, bp, ra, dec, gflux_raw, rflux_raw, zflux_raw, gflux, rflux, zflux, givar,\
        rivar, zivar, gf_err, rf_err, zf_err, mw_g, mw_r, mw_z, r_dev, r_exp, g_allmask, r_allmask, z_allmask, B, R, I, BRI_cut, cn,\
        w, red_z, z_err, z_quality, oii, oii_err, D2matched, field, w1_flux, w2_flux, w1_err, w2_err

bid, objtype, tycho, bp, ra, dec, gflux_raw, rflux_raw, zflux_raw, gflux, rflux, zflux, givar,\
rivar, zivar, gf_err, rf_err, zf_err, mw_g, mw_r, mw_z, r_dev, r_exp, g_allmask, r_allmask, z_allmask, B, R, I, BRI_cut, cn,\
w, red_z, z_err, z_quality, oii, oii_err, D2matched, field, w1_flux, w2_flux, w1_err, w2_err\
= load_tractor_DR5_matched_to_DEEP2_full()



field_colors = ["black", "red", "blue"]
ft_size = 15
ft_size2 = 20

grz_names = ["g", "r", "z"]

# grz magnitude
g, r, z = flux2mag(gflux), flux2mag(rflux), flux2mag(zflux)
# gr = g - r
# rz = r - z

# g-z vs. g-r color-color distribution
mu_g = flux2asinh_mag(gflux, band="g")
mu_r = flux2asinh_mag(rflux, band="r")
mu_z = flux2asinh_mag(zflux, band="z")

mu_gz = mu_g - mu_z
mu_gr = mu_g - mu_r



# Color distribution: matched vs. unmatched
plt.close()
colors = ["red", "black"]
labels = ["Unmatched", "Matched"]

fig, ax_list = plt.subplots(1, 3, figsize=(25, 7))
for i in [1, 0]: # Matched and unamtched
    for fnum in [2, 3, 4]:
        ibool = (D2matched==i) & (field==fnum)
        ax_list[fnum-2].scatter(mu_gz[ibool], mu_gr[ibool], s=25, marker="s", c=colors[i], label="F%d-%s" % (fnum, labels[i]))
        ax_list[fnum-2].set_xlim([-3, 5])
        ax_list[fnum-2].set_ylim([-1, 2]) 
        ax_list[fnum-2].grid(ls="--")
        ax_list[fnum-2].set_xlabel("asinh $g-z$", fontsize=ft_size2)
        ax_list[fnum-2].set_ylabel("asinh $g-r$", fontsize=ft_size2) 
        ax_list[fnum-2].legend(loc="upper left", fontsize=ft_size*1.5)
plt.suptitle("DR5 matched vs unmatched to DEEP2", fontsize=ft_size * 2)
plt.savefig(dir_figure + "DR5-matched-vs-unmatched-grz-color-by-field.png", dpi=200, bbox_inches="tight")
# plt.show()
plt.close()

# Mag distribution: matched vs. unmatched
plt.close()
colors = ["red", "black"]
labels = ["Unmatched", "Matched"]

mag_bins = np.arange(19, 26, 0.1)

fig, ax_list = plt.subplots(3, 3, figsize=(25, 25))
for i, mag in enumerate([g, r, z]):
    for l in [1, 0]: # Matched and unamtched
        for fnum in [2, 3, 4]:
            ibool = (D2matched==l) & (field==fnum)
            ax_list[i, fnum-2].hist(mag[ibool], bins=mag_bins, color=colors[l], histtype="step", alpha=1, lw=2.5, label=labels[l], normed=True)
            ax_list[i, fnum-2].set_xlim([19, 26])
            ax_list[i, fnum-2].set_xlabel(grz_names[i], fontsize=ft_size2)
            ax_list[i, fnum-2].set_ylabel("dNd(0.1mag)", fontsize=ft_size2)            
            ax_list[i, fnum-2].legend(loc="upper left", fontsize=ft_size*1.5)
plt.suptitle("DR5 matched vs unmatched to DEEP2", fontsize=ft_size * 2)
plt.savefig(dir_figure + "DR5-matched-vs-unmatched-grz-hist-normed-by-field.png", dpi=200, bbox_inches="tight")
# plt.show()
plt.close()