from NDM_models import *

model = DESI_NDM()
model.cell_select = np.load("./DR5-NDM1/NDM1-cell_select.npy")

model2 = DESI_NDM()
model2.cell_select = np.load("./DR5-NDM2/NDM2-cell_select.npy")


xmin = 0.55
xmax = 1.65

ymin = 0
ymax = 220.
dx = 0.05
redz_bins = np.arange(xmin, xmax, dx)

# Generate np=1 line
X, Y = np1_line(dz = dx)

for fnum in range(2, 5):
    # Selecting only objects in the field.
    ifield = (model.field == fnum)
    area_sample = model.areas[fnum-2]
    gflux = model.gflux[ifield] 
    rflux = model.rflux[ifield]
    zflux = model.zflux[ifield]
    var_x = model.var_x[ifield]
    var_y = model.var_y[ifield]
    gmag = model.gmag[ifield]
    oii = model.oii[ifield]
    redz = model.red_z[ifield]
    w = model.w[ifield]
    cn = model.cn[ifield]

    # Apply the selection.
    iselected = model.apply_selection(gflux, rflux, zflux) # Fiducial
    iselected2 = model2.apply_selection(gflux, rflux, zflux) # U_Gold

    plt.close()
    # Plot redshift histograms if requested
    fig, ax = plt.subplots(1, figsize=(5, 5))
    # -- Fiducial
    # Plot dNdredz of Gold and Silver classes
    ibool = np.logical_or((cn == 0), (cn==1)) * (oii > 8) & iselected
    ax.hist(redz[ibool], bins = redz_bins, histtype="step", \
            weights=w[ibool]/area_sample, color="black", lw=2, label="Fid.")
    # Contribution of NoOII
    ibool = (cn==2) & iselected
    NoOII_contribution = np.sum(w[ibool])/area_sample * 0.6 # 0.6 is the expected fraction.
    ax.plot([0.65, 0.75], [NoOII_contribution/(0.1/dx)]*2, lw=4, c=colors[2], label="NoOII") 
    # Contribution of NoZ
    ibool = (cn==3) & iselected
    NoZ_contribution = np.sum(w[ibool])/area_sample * 0.25 # 0.25 is the expected fraction.
    ax.plot([1.4, 1.6], [NoZ_contribution/(.2/dx)]*2, lw=3, c=colors[3], label="NoZ") 

    
    # -- U_Gold = 5
    # Plot dNdredz of Gold and Silver classes
    ibool = np.logical_or((cn == 0), (cn==1)) * (oii > 8) & iselected2
    ax.hist(redz[ibool], bins = redz_bins+0.005, histtype="step", weights=w[ibool]/area_sample, color="orange", lw=2, label="Var.")
    # Contribution of NoOII
    ibool = (cn==2) & iselected2
    NoOII_contribution = np.sum(w[ibool])/area_sample * 0.6 # 0.6 is the expected fraction.
    ax.plot([0.65, 0.75], [NoOII_contribution/(0.1/dx)]*2, lw=4, c=colors[2], ls="--") 
    # Contribution of NoZ
    ibool = (cn==3) & iselected2
    NoZ_contribution = np.sum(w[ibool])/area_sample * 0.25 # 0.25 is the expected fraction.
    ax.plot([1.4, 1.6], [NoZ_contribution/(.2/dx)]*2, lw=3, c=colors[3], ls="--") 
    
    # -- np=1 line
    ax.plot(X, Y, lw=2, c="blue", ls="--", label="$nP$=1")

    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_xlabel("Redshift z", fontsize=20)
    ax.set_ylabel("dN/d(0.05z)", fontsize=20)
    plt.legend(loc="upper right", fontsize=18)
    plt.savefig("../figures/NDM-dNdz-fiducial-Ugold5-DEEP2F%d.png" % fnum, dpi=400, bbox_inches="tight")
#     plt.show()
    plt.close()
        