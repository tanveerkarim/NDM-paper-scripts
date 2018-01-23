from NDM_models import *

save_dir = "./DR5-NDM-Obiwan-samples/"

model = DESI_NDM()
model.set_area_MC(100)
model.load_dNdm_models()
model.load_MoG_models()

model.gen_sample_intrinsic_mag(intrinsic_only=True)

# grz-fluxes and redz and OII and weights
for i in range(2):
    gflux = model.gflux0[i]
    rflux = model.rflux0[i]
    zflux = model.zflux0[i]
    oii = model.oii0[i]
    weights = model.iw0[i]
    
    # Save the samples
    np.savez(save_dir+"NDM-obiwan-sample-%s.npz" % cnames[i], gflux = gflux, rflux = rflux, zflux = zflux, OII = oii, weights = weights)
    # To load use: data = np.load("NDM-obiwan-sample.npz")

    
    data = np.load(save_dir+"NDM-obiwan-sample-%s.npz"  % cnames[i])

    gflux = data["gflux"]
    rflux = data["rflux"]
    zflux = data["zflux"]
    oii = data["OII"]
    weights = data["weights"]

    # Parameterization
    mu_g = flux2asinh_mag(gflux, band="g")
    mu_r = flux2asinh_mag(rflux, band="r")
    mu_z = flux2asinh_mag(zflux, band="z")

    mu_gz = mu_g - mu_z
    mu_gr = mu_g - mu_r

    ibool = (gflux>0) & (rflux >0) & (zflux>0)
    gmag = flux2mag(gflux[ibool])
    rmag = flux2mag(rflux[ibool])
    zmag = flux2mag(zflux[ibool])

    gr = gmag-rmag
    rz = rmag-zmag



    # Asinh color scatter
    plt.close()
    fig, ax = plt.subplots(1, figsize=(7, 7))
    ax.scatter(mu_gz, mu_gr, s=1, edgecolor="none", alpha=1, c="black")
    ax.set_xlabel("asinh g-z")
    ax.set_ylabel("asinh g-r")
    ax.axis("equal")
    ax.set_xlim([-2, 5])
    ax.set_ylim([-2, 5])
    # plt.show()
    plt.savefig(save_dir+"asinh-colors-%s.png" % cnames[i], dpi=200, bbox_inches="tight")
    plt.close()

    # mag color scatter
    fig = plt.figure(figsize=(7, 7))
    plt.scatter(rz, gr, s=1, edgecolor="none", alpha=1., c="black")
    plt.xlabel("r-z")
    plt.ylabel("g-r")
    plt.axis("equal")
    plt.xlim([-4, 4])
    # plt.show()
    plt.savefig(save_dir+"mag-colors-%s.png" % cnames[i], dpi=200, bbox_inches="tight")
    plt.close()



    # grz histogram
    bins = np.arange(18, 26, 0.1)
    fig = plt.figure(figsize=(7, 7))
    plt.hist(gmag, bins=bins, histtype="step", color="green", lw=2, label="gmag")
    plt.hist(rmag, bins=bins, histtype="step", color="red", lw=2, label="rmag")
    plt.hist(zmag, bins=bins, histtype="step", color="purple", lw=2, label="zmag")
    plt.legend(loc="upper left", fontsize=15)
    plt.savefig(save_dir+"mag-hist-%s.png" % cnames[i], dpi=200, bbox_inches="tight")
    # plt.show()
    plt.close()

    fig = plt.figure(figsize=(7, 7))
    plt.hist(gmag, bins=bins, histtype="step", color="green", lw=2, label="gmag", weights=weights[ibool])
    plt.hist(rmag, bins=bins, histtype="step", color="red", lw=2, label="rmag", weights=weights[ibool])
    plt.hist(zmag, bins=bins, histtype="step", color="purple", lw=2, label="zmag", weights=weights[ibool])
    plt.legend(loc="upper left", fontsize=15)
    plt.savefig(save_dir+"mag-hist-weighted-%s.png" % cnames[i], dpi=200, bbox_inches="tight")
    # plt.show()
    plt.close()



    # OII histogram
    fig = plt.figure(figsize=(7, 7))
    bins = np.arange(-10, 100, 1)
    plt.hist(oii, bins=bins, histtype="step", color="black", lw=2, label="OII")
    plt.xlabel("Flux in 1e-17")
    plt.legend(loc="upper right", fontsize=15)
    plt.savefig(save_dir+"OII-%s.png" % cnames[i], dpi=200, bbox_inches="tight")
    plt.close()

    fig = plt.figure(figsize=(7, 7))
    plt.hist(oii, bins=bins, histtype="step", color="black", lw=2, label="OII", weights=weights)
    plt.xlabel("Flux in 1e-17")
    plt.legend(loc="upper right", fontsize=15)
    plt.savefig(save_dir+"OII-weighted-%s.png" % cnames[i], dpi=200, bbox_inches="tight")
    plt.close()

