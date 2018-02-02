from NDM_models import *

#---- Calibration data
gflux_cal, _, _, _, _, A = load_DR5_calibration()
g_cal = flux2mag(gflux_cal)



model = DESI_NDM()
field = model.field
for fnum in range(2,5):

    #---- Load DEEP2 data
    weights = model.w[field==fnum]
    g_DEEP2 = model.gmag[field==fnum]
    area = model.areas[0]


    # Plot 
    fig, ax = plt.subplots(1, figsize=(5, 4))
    dm =0.05
    mag_bins = np.arange(19, 24+0.05, dm)
    ax.hist(g_cal, bins=mag_bins, color="black", histtype="step", label="DR5 Cal.", weights=np.ones(g_cal.size)/A, lw=2)
    ax.hist(g_DEEP2, bins=mag_bins, color="red", histtype="step", label="DEEP2 F%d"%fnum, weights=weights/area, lw=2)
    ax.set_xlim([19, 24])
    ax.set_xlabel("g", fontsize=20)
    ax.set_ylabel("dN/d(%.2fmag)" % dm, fontsize=20)
    ax.legend(loc="upper left", fontsize=20)
    plt.savefig("../figures/g-hist-comparison-F%d.png" % fnum, dpi=400, bbox_inches="tight")
    # plt.show()
    plt.close()