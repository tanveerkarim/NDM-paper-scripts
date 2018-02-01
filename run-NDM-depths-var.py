from NDM_models import *
#--- fiducial
model = DESI_NDM()
model.cell_select = np.load("./DR5-NDM1/NDM1-cell_select.npy")

def delta(a, b):
    if a == b:
        return 1
    else: 
        return 0

print "#---- NDM depths variation from the fiducial"
print "Generate intrinsic samples."
# Generate intrinsic samples.
area_MC = 1000
model.set_area_MC(area_MC)
model.load_calibration_data()
model.load_dNdm_models()
model.load_MoG_models()
model.gen_sample_intrinsic_mag() # This generates err seeds as well.

# Mag depth changes
dm = 0.05
mag_bins = np.arange(-1, 1+dm/2., dm)

# Efficiency save
# axis 0: grz
# axis 1: dm
# axis 2: Efficiency contribution of each class
eff_arr = np.zeros((3, mag_bins.size, 6), dtype=float)


labels = ["g", "r", "z"]
for i in range(3): # 0, 1, 2: g, r, z
    print "/---- %s" % labels[i]
    print "Depths change/efficiency"    
    for j, m in enumerate(mag_bins):
        # Projected decam
        model.set_err_lims(23.8+m*delta(i, 0), 23.4+m*delta(i, 1), 22.4+m*delta(i, 2), 8) 
        
        eff_arr_tmp = model.check_eff_depth_var(gaussian_smoothing=True, sig_smoothing_window=[5, 5, 5], \
        dNdm_mag_reg=True, fake_density_fraction = 0.01, regen_intrinsic=False)

                # Total efficiency.
        eff_arr[i, j, :] = eff_arr_tmp
    np.save("./DR5-NDM4/eff_arr.npy", eff_arr)


# Make the plot.
colors = ["g","r","purple"]
labels = ["g", "r", "z"]
fig, ax = plt.subplots(1, figsize=(7, 7))
for i in range(3):
    ax.plot(mag_bins, eff_arr[i][:, -1], lw=2, c=colors[i], label=labels[i])
ax.set_xlim([-1, 1])
ax.axvline(x=0, lw=2, ls="--", c="black")
ax.axhline(y=0.637, lw=2, ls="--", c="black")
ax.set_xlabel("5-sigma depth change", fontsize=20)
ax.set_ylabel("Efficiency", fontsize=20)
plt.legend(loc="lower right", fontsize=20)
plt.savefig("../figures/NDM-depths-var.png", dpi=400, bbox_inches="tight")
# plt.show()
plt.close()