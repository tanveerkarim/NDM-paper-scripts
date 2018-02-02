from NDM_models import *
model = DESI_NDM()

# ---- Fiducial
save_dir = "./DR5-NDM1/"
model_name = "NDM1"
data = np.load(save_dir + model_name + "-marginal_eff.npz")
bin_centers = data["centers"]
summary = data["summary"]
model.bin_centers = bin_centers
model.summary_arr = summary
model.make_marginal_eff_plot(save_dir=save_dir)

# ----- U_Gold
save_dir = "./DR5-NDM2/"
model_name = "NDM2"
data = np.load(save_dir + model_name + "-marginal_eff.npz")
bin_centers = data["centers"]
summary = data["summary"]
model.bin_centers = bin_centers
model.summary_arr = summary
model.make_marginal_eff_plot(save_dir=save_dir)
