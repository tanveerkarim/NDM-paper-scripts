from NDM_models import *

fake_density_fraction = 0.01
num_batches = 10

print "\#---- NDM2: Fiducial and U_Gold = 5"
save_dir = "./DR5-NDM2/"
model_name = "NDM2"
model = DESI_NDM()
model.set_num_desired(2400)
model.U_Gold = 5
model.load_calibration_data()
# Projected decam
model.set_err_lims(23.8, 23.4, 22.4, 8) 
model.load_dNdm_models()
model.load_MoG_models()
bin_centers, summary_arr = model.gen_selection_volume_ext_cal(num_batches=num_batches, batch_size=1000, \
                                                              gaussian_smoothing=True, dNdm_mag_reg=True, \
                                                             fake_density_fraction=fake_density_fraction)
# Save the selection
np.save(save_dir + model_name + "-cell_select.npy", model.cell_select)

# Save the marginal efficiency summaries
np.savez(save_dir + model_name + "-marginal_eff.npz", centers = bin_centers, summary = summary_arr)

# Validate on DEEP2 data
model.validate_on_DEEP2()

# Make marginal efficiency plot
model.make_marginal_eff_plot(save_dir=save_dir)

for j in range(3):
    model.gen_select_boundary_slices(slice_dir=j, prefix=model_name, save_dir=save_dir, output_sparse=True)




print "\n\n\n\n\n#---- NDM3: Fiducial and U_NoZ = 1"
save_dir = "./DR5-NDM3/"
model_name = "NDM3"
model = DESI_NDM()
model.set_num_desired(2400)
model.U_NoZ = 1
model.load_calibration_data()
# Projected decam
model.set_err_lims(23.8, 23.4, 22.4, 8) 
model.load_dNdm_models()
model.load_MoG_models()
bin_centers, summary_arr = model.gen_selection_volume_ext_cal(num_batches=num_batches, batch_size=1000, \
                                                              gaussian_smoothing=True, dNdm_mag_reg=True, \
                                                             fake_density_fraction=fake_density_fraction)
# Save the selection
np.save(save_dir + model_name + "-cell_select.npy", model.cell_select)

# Save the marginal efficiency summaries
np.savez(save_dir + model_name + "-marginal_eff.npz", centers = bin_centers, summary = summary_arr)

# Validate on DEEP2 data
model.validate_on_DEEP2()

# Make marginal efficiency plot
model.make_marginal_eff_plot(save_dir=save_dir)

for j in range(3):
    model.gen_select_boundary_slices(slice_dir=j, prefix=model_name, save_dir=save_dir, output_sparse=True)


