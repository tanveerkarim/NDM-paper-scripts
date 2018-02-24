from utils import *

dir_derived = "../data/derived/"
dir_figure = "../figures/"
print("Import estimated areas")
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
    cn = tbl["cn"]
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

ibool = D2matched==1

# Load only the matched set.
bid, objtype, tycho, bp, ra, dec, gflux_raw, rflux_raw, zflux_raw, gflux, rflux, zflux, givar,\
rivar, zivar, gf_err, rf_err, zf_err, mw_g, mw_r, mw_z, r_dev, r_exp, g_allmask, r_allmask, z_allmask, B, R, I, BRI_cut, cn,\
w, red_z, z_err, z_quality, oii, oii_err, D2matched, field, w1_flux, w2_flux, w1_err, w2_err\
= load_tractor_DR5_matched_to_DEEP2_full(ibool)


field_colors = ["black", "red", "blue"]
ft_size = 15
ft_size2 = 20

grz_names = ["g", "r", "z"]

# grz magnitude
g, r, z = flux2mag(gflux), flux2mag(rflux), flux2mag(zflux)

# g-z vs. g-r color-color distribution
mu_g = flux2asinh_mag(gflux, band="g")
mu_r = flux2asinh_mag(rflux, band="r")
mu_z = flux2asinh_mag(zflux, band="z")

mu_gz = mu_g - mu_z
mu_gr = mu_g - mu_r






print("grz magnitude distributions - raw")
plt.close()
fig, ax_list = plt.subplots(1, 3, figsize = (25, 7))
mag_bins = np.arange(19, 26, 0.1)

glim = 24.25
for j, fnum in enumerate([2, 3, 4]):    
    ifield = field == fnum
    for i, mag in enumerate([g, r, z]):
        #Added lines to fix NaN issue; certain values of mag are NaN
        weights = np.ones(ifield.sum())/areas[j]
        weights = weights[~np.isnan(mag)]
        mag = mag[~np.isnan(mag)]
        #
        ax_list[i].hist(mag[ifield], bins=mag_bins, color=field_colors[j], histtype="step", alpha=1, lw=2.5, label="F%d" % fnum, weights=weights)
        ax_list[i].set_xlim([19, 26])
        ax_list[i].legend(loc="upper right", fontsize=ft_size)
        ax_list[i].set_xlabel(grz_names[i], fontsize=ft_size2)
        ax_list[i].set_ylabel("dNd(0.1mag)", fontsize=ft_size2)

plt.savefig(dir_figure + "Intersection-grz-hist-raw.png", dpi=200, bbox_inches="tight")
# plt.show()
plt.close()


print("grz magnitude distributions - weighted")
plt.close()
fig, ax_list = plt.subplots(1, 3, figsize = (25, 7))
mag_bins = np.arange(19, 26, 0.1)

glim = 24.25
for j, fnum in enumerate([2, 3, 4]):    
    ifield = field == fnum
    for i, mag in enumerate([g, r, z]):
        ax_list[i].hist(mag[ifield], bins=mag_bins, color=field_colors[j], histtype="step", alpha=1, lw=2.5, label="F%d" % fnum, weights=w[ifield]/areas[j])
        ax_list[i].set_xlim([19, 26])
        ax_list[i].legend(loc="upper right", fontsize=ft_size)
        ax_list[i].set_xlabel(grz_names[i], fontsize=ft_size2)
        ax_list[i].set_ylabel("dNd(0.1mag)", fontsize=ft_size2)

plt.savefig(dir_figure + "Intersection-grz-hist-weighted.png", dpi=200, bbox_inches="tight")
# plt.show()
plt.close()



print("redz-oii weighted")
plt.close()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 7))
redz_bins = np.arange(0, 2, 0.025)
oii_bins = np.arange(-2, 40, 0.5)
for i, fnum in enumerate([2, 3, 4]):    
    ifield = (field == fnum) & np.logical_or(cn==0, cn==1) # Only ELGs 
    # Redshift
    ax1.hist(red_z[ifield], bins=redz_bins, color=field_colors[i], histtype="step", alpha=0.9, lw=2.5, label="F%d" % fnum, weights=w[ifield]/areas[i])
    ax1.set_xlim([0.5, 1.7])
    ax1.legend(loc="upper right", fontsize=ft_size)
    ax1.set_xlabel("Redshift z", fontsize=ft_size2)
    ax1.set_ylabel("dNd(0.025z)", fontsize=ft_size2)

    # OII
    ax2.hist(oii[ifield], bins=oii_bins, color=field_colors[i],  histtype="step", lw=2, label="F%d" % fnum, weights=w[ifield]/areas[i])
    ax2.set_xlim([-2, 40])
    ax2.legend(loc="upper right", fontsize=ft_size)
    ax2.set_xlabel("OII * 1e17", fontsize=ft_size2)
    ax2.set_ylabel("dNd(0.5OII)", fontsize=ft_size2)
    
plt.savefig(dir_figure + "Intersection-redz-oii-hist-ELG.png", dpi=200, bbox_inches="tight")
# plt.show()
plt.close()



print("Color distribution by class and field")
plt.close()
fig, ax_list = plt.subplots(len(cnames), 3, figsize = (25, 52))
for i, name in enumerate(cnames):
    for j, fnum in enumerate([2, 3, 4]):
        ibool = (g > 21) & (g < 24) & (field==fnum) & (cn == i)
        ax_list[i, j].scatter(mu_gz[ibool], mu_gr[ibool], s=25, marker="s", c=colors[i], label="F%d-%s" % (fnum, cnames[i]))
#         ax_list[i, j].axis("equal")
        ax_list[i, j].set_xlim([-3, 5])
        ax_list[i, j].set_ylim([-1, 2]) 
        ax_list[i, j].grid(ls="--")
        ax_list[i, j].set_xlabel("asinh $g-z$", fontsize=ft_size2)
        ax_list[i, j].set_ylabel("asinh $g-r$", fontsize=ft_size2) 
        ax_list[i, j].legend(loc="upper left", fontsize=ft_size*1.5)

plt.savefig(dir_figure + "Intersection-grz-color-by-class-field.png", dpi=200, bbox_inches="tight")
# plt.show() 
plt.close()


print("Color distribution by class only and all together")
plt.close()
fig, ax_list = plt.subplots(3, 2, figsize = (16, 25))
for i, name in enumerate(cnames):
    ax_row = i // 2
    ax_col = i % 2    
    if i < 5:
        ibool = (g > 21) & (g < 24) &  (cn == i)
        ax_list[ax_row, ax_col].scatter(mu_gz[ibool], mu_gr[ibool], s=25, marker="s", c=colors[i], label="%s" % cnames[i])
        ax_list[ax_row, ax_col].set_xlim([-3, 5])
        ax_list[ax_row, ax_col].set_ylim([-1, 2]) 
        ax_list[ax_row, ax_col].grid(ls="--")
        ax_list[ax_row, ax_col].set_xlabel("asinh $g-z$", fontsize=ft_size2)
        ax_list[ax_row, ax_col].set_ylabel("asinh $g-r$", fontsize=ft_size2) 
        ax_list[ax_row, ax_col].legend(loc="upper left", fontsize=ft_size*1.5)
    else: # Last panel is used to plot everything all together except the unobserved.
        for j in [4, 3, 2, 1, 0]:
            ibool = (g > 21) & (g < 24) &  (cn == j)
            ax_list[ax_row, ax_col].scatter(mu_gz[ibool], mu_gr[ibool], s=25, marker="s", c=colors[j], label="%s" % cnames[j])
            ax_list[ax_row, ax_col].set_xlim([-3, 5])
            ax_list[ax_row, ax_col].set_ylim([-1, 2]) 
            ax_list[ax_row, ax_col].grid(ls="--")
            ax_list[ax_row, ax_col].set_xlabel("asinh $g-z$", fontsize=ft_size2)
            ax_list[ax_row, ax_col].set_ylabel("asinh $g-r$", fontsize=ft_size2) 
            ax_list[ax_row, ax_col].legend(loc="lower right", fontsize=ft_size*1.)
plt.savefig(dir_figure + "Intersection-grz-color-by-class.png", dpi=200, bbox_inches="tight")
# plt.show() 
plt.close()


fontsize2 = 20
print("Color distribution by class only and all together")
plt.close()
for i, name in enumerate(cnames):
    print(i)
    fig, ax = plt.subplots(1, figsize = (5, 5))       
    if i < 5:
        ibool = (g > 21) & (g < 24) &  (cn == i)
        ax.scatter(mu_gz[ibool], mu_gr[ibool], s=25, marker="s", c=colors[i], label="%s" % cnames[i])
        ax.set_xlim([-3, 5])
        ax.set_ylim([-1, 2]) 
        ax.grid(ls="--")
        ax.set_xlabel("$\mu_g-\mu_z$", fontsize=ft_size2 * 1.5)
        ax.set_ylabel("$\mu_g-\mu_r$", fontsize=ft_size2 * 1.5) 
        ax.legend(loc="upper left", fontsize=ft_size*1.)
        plt.savefig(dir_figure + "Intersection-grz-color-%d.png" % i, dpi=400, bbox_inches="tight")
#         plt.show() 
        plt.close()
        
    else: # Last panel is used to plot everything all together except the unobserved.
        fig, ax = plt.subplots(1, figsize = (5, 5))            
        for j in [4, 3, 2, 1, 0]:
            ibool = (g > 21) & (g < 24) &  (cn == j)
            ax.scatter(mu_gz[ibool], mu_gr[ibool], s=25, marker="s", c=colors[j], label="%s" % cnames[j])
            ax.set_xlim([-3, 5])
            ax.set_ylim([-1, 2]) 
            ax.grid(ls="--")
            ax.set_xlabel("$\mu_g-\mu_z$", fontsize=ft_size2 * 1.5)
            ax.set_ylabel("$\mu_g-\mu_r$", fontsize=ft_size2 * 1.5) 
        ax.legend(loc="upper left", fontsize=ft_size*1.)
        plt.savefig(dir_figure + "Intersection-grz-color-all.png", dpi=400, bbox_inches="tight")
#         plt.show() 
        plt.close()
