from NDM_models import *

#---- Load fiducial boundary
model = DESI_NDM()
model.cell_select = np.load("./DR5-NDM1/NDM1-cell_select.npy")
for j in range(3):
    model.gen_select_boundary_slices(slice_dir=j, prefix="fiducial", save_dir="../figures/", output_sparse=True, increment=25, \
                                plot_calibration_sample=True, pt_size_ext=1)