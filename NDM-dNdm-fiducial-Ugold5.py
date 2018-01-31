from NDM_models import *

model = DESI_NDM()
model.cell_select = np.load("./DR5-NDM1/NDM1-cell_select.npy")

model2 = DESI_NDM()
model2.cell_select = np.load("./DR5-NDM2/NDM2-cell_select.npy")


# Load calibration data
g, r, z, _, _, A= load_DR5_calibration()

# Apply selections to both
iselected = model.apply_selection(g, r, z)
iselected2 = model2.apply_selection(g, r, z)

dm = 0.05
mag_min = 19.5
mag_max = 24.5
magbins = np.arange(mag_min, mag_max, dm)

# Find magnitudes and plot
plt.close()
fig, ax = plt.subplots(1, figsize=(8, 6))
#--- Fiducial
gmag, rmag, zmag = flux2mag(g[iselected]), flux2mag(r[iselected]), flux2mag(z[iselected])
# g hist
ax.hist(gmag, bins=magbins, color="green", lw=2, histtype="stepfilled", alpha=0.4,weights=np.ones(gmag.size)/A, edgecolor="none", label="$g$")
# r hist
ax.hist(rmag, bins=magbins, color="red", lw=2, histtype="stepfilled", alpha=0.4,weights=np.ones(gmag.size)/A, edgecolor="none", label="$r$")
# z hist
ax.hist(zmag, bins=magbins, color="purple", lw=2, histtype="stepfilled", alpha=0.4,weights=np.ones(gmag.size)/A, edgecolor="none", label="$z$")

#--- Ugold5
gmag, rmag, zmag = flux2mag(g[iselected2]), flux2mag(r[iselected2]), flux2mag(z[iselected2])
# g hist
ax.hist(gmag, bins=magbins, color="green", lw=2, histtype="step", weights=np.ones(gmag.size)/A)
# r hist
ax.hist(rmag, bins=magbins, color="red", lw=2, histtype="step", weights=np.ones(gmag.size)/A)
# z hist
ax.hist(zmag, bins=magbins, color="purple", lw=2, histtype="step", weights=np.ones(gmag.size)/A)

#--- Depth limits
ax.axvline(x=23.8, c="green", lw=2, ls="--")
ax.axvline(x=23.4, c="red", lw=2, ls="--")
ax.axvline(x=22.4, c="purple", lw=2, ls="--")


ax.set_xlabel("Magnitude", fontsize=20)
ax.set_ylabel("dNd(0.5mag)", fontsize=20)
ax.set_xlim([mag_min, mag_max])

plt.legend(loc="upper left", fontsize=20)
plt.savefig("../figures/NDM-dNdm-fiducial-Ugold5-calibration.png", bbox_inches="tight", dpi=400)
# plt.show()
plt.close()
