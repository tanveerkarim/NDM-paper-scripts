from NDM_models import *
model = DESI_NDM()

plt.close()
#--- Fit dNdm
model.fit_dNdm_broken_pow() # Fit dNdm for all classes
model.load_dNdm_models()
model.plot_dNdm_all(savefig=True, fits=True)

#--- Fit colors
for i in range(5):
    print("Fitting MoG to class %s" % cnames[i])
    for k in range(1, 4):
        print("Component number: %d"% k)
        model.fit_MoG(cn=i, K=k)
    if i==4: # For the last class fit more
        for k in range(4, 7):
            print("Component number: %d"% k            )
            model.fit_MoG(cn=i, K=k)
    print("\n")

# For each fitted models, make fit plots.
# 1) Generate MC samples from dNdm
# 2) Generate MC samples from MoG
# 3) Make error convolution
# 4) Make plots
area_MC = 10
model.set_area_MC(area_MC)
model.set_err_lims(25., 24.5, 23.2, 8) # For the training data set

for k in range(1, 4):
    print("Component number: %d"% k)
    model.load_MoG_models(K_list=[k]*5) # Load the desired models
    for i in range(5):
        print("Validation plot for %s" % cnames[i])
        model.gen_sample_intrinsic_mag() # Draw intrinsic samples
        model.gen_err_conv_sample() # Perform error convolution
        # Make plots
        model.plot_colors(cn=i, K=k, savefig=True, show=False, plot_ext=True, A_ext = area_MC,\
                         iw = model.iw[i], gflux = model.gflux_obs[i], rflux = model.rflux_obs[i], zflux = model.zflux_obs[i],
                         oii = model.oii_obs[i]) 
        print("\n")

i==4 # For the last
for k in range(4, 7):
    print("Component number: %d"% k   )
    print("Validation plot for %s" % cnames[i])
    model.load_MoG_models(K_list=[1, 1, 1, 1, k]) # Load the desired models
    model.gen_sample_intrinsic_mag() # Draw intrinsic samples
    model.gen_err_conv_sample() # Perform error convolution
    # Make plots
    model.plot_colors(cn=i, K=k, savefig=True, show=False, plot_ext=True, A_ext = area_MC,\
                     iw = model.iw[i], gflux = model.gflux_obs[i], rflux = model.rflux_obs[i], zflux = model.zflux_obs[i],
                     oii = model.oii_obs[i]) 
    print("\n")
