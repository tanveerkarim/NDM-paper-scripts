from utils import *

# Data directory
dir_derived = "../data/derived/"

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
    # Ad hoc solution       
    field[:tbl1_size] = 2 
    field[tbl1_size:tbl1_size+tbl2_size] = 3 
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
    cn = tbl["cn"].astype(int)
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


class DESI_NDM(object):
    def __init__(self):
        """
        Import the intersection training data. Set global parameters to default values.
        """

        # mag_max, mag_min: Magnitue range considered for intrinsic density modeling and MC sampling
        self.mag_max = 24.25 
        self.mag_min = 17.

        # Areas of Field 2, 3, and 4
        self.areas = np.load(dir_derived+"spec-area.npy")        

        # Model variables
        self.gflux, self.gf_err, self.rflux, self.rf_err, self.zflux, self.zf_err, self.red_z, self.z_err, self.oii, self.oii_err, self.w, self.field,\
        self.ra, self.dec, self.w1_flux, self.w2_flux, self.w1_err, self.w2_err, self.cn,\
        self.iELG_DESI, self.iNoZ, self.iNoOII, self.iNonELG = self.import_data_DEEP2_full()

        return

    def import_data_DEEP2_full(self):
        """Return DEEP2-DR5 data."""
        bid, objtype, tycho, bp, ra, dec, gflux_raw, rflux_raw, zflux_raw, gflux, rflux, zflux, givar,\
        rivar, zivar, gf_err, rf_err, zf_err, mw_g, mw_r, mw_z, r_dev, r_exp, g_allmask, r_allmask, z_allmask, B, R, I, BRI_cut, cn,\
        w, red_z, z_err, z_quality, oii, oii_err, D2matched, field, w1_flux, w2_flux, w1_err, w2_err\
            = load_tractor_DR5_matched_to_DEEP2_full()

        ifcut = (gflux > mag2flux(self.mag_max)) & (gflux < mag2flux(self.mag_min))
        ibool = (D2matched==1) & ifcut
        nobjs_cut = ifcut.sum() # Number of objects that pass the magnitude cuts.
        nobjs_matched = ibool.sum() # Number of objects that pass the magnitude cuts and matched in DEEP2.

        frac_unmatched = (nobjs_cut-nobjs_matched)/float(nobjs_cut)

        print "Fraction of unmatched objects with g [%.2f, %.2f]: %.2f %%" % (self.mag_min, self.mag_max, 100 * frac_unmatched)
        print "We up-weight the matched set by this fraction before fitting the intrinsic densities."

        bid, objtype, tycho, bp, ra, dec, gflux_raw, rflux_raw, zflux_raw, gflux, rflux, zflux, givar,\
        rivar, zivar, gf_err, rf_err, zf_err, mw_g, mw_r, mw_z, r_dev, r_exp, g_allmask, r_allmask, z_allmask, B, R, I, BRI_cut, cn,\
        w, red_z, z_err, z_quality, oii, oii_err, D2matched, field, w1_flux, w2_flux, w1_err, w2_err\
            = load_tractor_DR5_matched_to_DEEP2_full(ibool = ibool)

        # Some convenient indicators
        iELG_DESI = np.logical_or((cn==0), (cn==1)) & (oii > 8)
        iNoZ = (cn==3)
        iNoOII = (cn==2)
        iNonELG = (cn==4)

        # Correct the weights.
        w /= float(1.0-frac_unmatched)

        return gflux, gf_err, rflux, rf_err, zflux, zf_err, red_z, z_err, oii, oii_err, w, field, \
        ra, dec, w1_flux, w2_flux, w1_err, w2_err, cn, iELG_DESI, iNoZ, iNoOII, iNonELG



    # def plot_dNdm_all(self):
    #     """
    #     Plot dNdm for all four classes (ALL ELG, DESI ELG, NoZ, NonELG) in 3 by 4 plot.

    #     rows: g, r, z
    #     columns: classes
    #     graphs: fields
    #     """
    #     colors = ["black", "red", "blue"]
    #     fig, ax_list = plt.subplots(3, 4, figsize = (28, 20))
    #     class_names = ["All ELGs", "DESI ELGs", "NoZ", "NonELG"]
    #     mag_names = [r"$g$", r"$r$", r"$z$"]
    #     mag_bins = np.arange(20, 25., 0.1)
    #     for j, iclass in enumerate([self.iELG, self.iELG & (self.oii>8), self.iNoZ, self.iNonELG]):
    #         for i, flux in enumerate([self.gflux, self.rflux, self.zflux]):
    #             for fnum in [2, 3, 4]:
    #                 ibool = iclass & (self.field==fnum)
    #                 A = self.areas[fnum-2]
    #                 ax_list[i][j].hist(flux2mag(flux[ibool]), bins=mag_bins, weights = self.w[ibool]/float(A),\
    #                     label = "F%d" % fnum, color = colors[fnum-2], histtype="step", lw = 2.5)
    #             title_str = "%s. %s" % (class_names[j], mag_names[i])
    #             ax_list[i][j].set_title(title_str, fontsize=25)
    #             ax_list[i][j].set_xlabel(mag_names[i], fontsize=20)
    #             ax_list[i][j].set_ylabel("dNdm", fontsize=20)
    #     plt.suptitle("dNdm with g [%.2f, %.2f]. Only matched. Weighted density" %(self.mag_min, self.mag_max), fontsize=30)
    #     plt.savefig("dNdm-grz-all-DR5-matched-wdensity.png", dpi = 400, bbox_inches="tight")
    #     plt.close()





    # def fit_MoG(self, NK_list, model_tag="", cv_tag="", cache=False):
    #     """
    #     Fit MoGs to data. Note that here we only consider fitting to 3 or 5 dimensions.

    #     If cache = True, then search to see if there are models already fit and if available use them.
    #     """
    #     cache_success = False
    #     if cache:
    #         for i in range(3):  
    #             model_fname = "./MODELS-%s-%s-%s.npy" % (self.category[i], model_tag, cv_tag)
    #             if os.path.isfile(model_fname):
    #                 self.MODELS[i] = np.load(model_fname).item()
    #                 cache_success = True
    #                 print "Cached result will be used for MODELS-%s-%s-%s." % (self.category[i], model_tag, cv_tag)

    #     if not cache_success: # If cached result was not requested or was searched for but not found.
    #         # Dimension of model
    #         ND = 3
    #         # Number of variables up to which MoG is being proposed
    #         ND_fit = 3
    #         for i, ibool in enumerate([self.iNonELG, self.iNoZ]):
    #             print "Fitting MoGs to %s" % self.category[i]
    #             ifit = ibool & self.iTrain
    #             Ydata = np.array([self.var_x[ifit], self.var_y[ifit], self.var_z[ifit]]).T
    #             Ycovar = self.gen_covar(ifit, ND=3)
    #             weight = self.w[ifit]
    #             self.MODELS[i] = fit_GMM(Ydata, Ycovar, ND, ND_fit, NK_list=NK_list, Niter=5, fname_suffix="%s-%s-%s" % (self.category[i], model_tag, cv_tag), MaxDIM=True, weight=weight)

    #         i = 2
    #         # Dimension of model
    #         ND = 5
    #         # Number of variables up to which MoG is being proposed
    #         ND_fit = 5
    #         print "Fitting MoGs to %s" % self.category[i]        
    #         ifit = self.iELG & self.iTrain
    #         Ydata = np.array([self.var_x[ifit], self.var_y[ifit], self.var_z[ifit], self.var_w[ifit], self.red_z[ifit]]).T
    #         Ycovar = self.gen_covar(ifit, ND=5)
    #         weight = self.w[ifit]
    #         self.MODELS[i] = fit_GMM(Ydata, Ycovar, ND, ND_fit, NK_list=NK_list, Niter=5, fname_suffix="%s-%s-%s" % (self.category[i], model_tag, cv_tag), MaxDIM=True, weight=weight)

    #     return




    # def visualize_fit(self, model_tag="", cv_tag="", cum_contour=False):
    #     """
    #     Make corr plots of the various classes with fits overlayed.
    #     If cum_contour is True, then instead of plotting individual component gaussians,
    #     plt the cumulative gaussian fit.
    #     """

    #     print "Corr plot - var_xyz - Separately"
    #     num_cat = 1
    #     num_vars = 3

    #     lims = [self.lim_x, self.lim_y, self.lim_z]
    #     binws = [self.dx, self.dy, self.dz]
    #     var_names = [self.var_x_name, self.var_y_name, self.var_z_name]
    #     lines = [self.var_x_lines, self.var_y_lines, self.var_z_lines]

    #     for i, ibool in enumerate([self.iNonELG, self.iNoZ]):
    #         print "Plotting %s" % self.category[i]

    #         # Take the real data points.
    #         variables = []
    #         weights = []                
    #         iplot = np.copy(ibool) & self.iTrain
    #         variables.append([self.var_x[iplot], self.var_y[iplot], self.var_z[iplot]])
    #         weights.append(self.w[iplot]/self.area_train)

    #         MODELS = self.MODELS[i] # Take the model for the category.
    #         # Plotting the fits
    #         for j, var_num_tuple in enumerate(MODELS.keys()): # For each selection of variables
    #             if len(var_num_tuple) < 3: # Only plot the last models.
    #                 pass
    #             else:
    #                 # Models corresponding to the tuples
    #                 ms = MODELS[var_num_tuple]
    #                 for K in ms.keys(): # For each component number tried
    #                     print "K: %d" % K
    #                     # Fits
    #                     m = ms[K]
    #                     amps_fit  = m["amps"]
    #                     means_fit  = m["means"]
    #                     covs_fit = m["covs"]        
            
    #                     fig, ax_list = plt.subplots(num_vars, num_vars, figsize=(25, 25))
    #                     # Corr plots without annotation
    #                     ax_dict = make_corr_plots(ax_list, num_cat, num_vars, variables, lims, binws,\
    #                                               var_names, weights, lines=lines, category_names=[self.category[i]],\
    #                                               pt_sizes=None, colors=None, ft_size_legend = 15, lw_dot=2, hist_normed=True,\
    #                                               plot_MoG_general=True, var_num_tuple=var_num_tuple, amps_general=amps_fit,\
    #                                               means_general=means_fit, covs_general=covs_fit, color_general="red", cum_contour=cum_contour)
    #                     plt.tight_layout()
    #                     if cum_contour:
    #                         plt.savefig("%s-%s-data-%s-fit-K%d-cum-contour.png" % (model_tag, cv_tag, self.category[i], K), dpi=200, bbox_inches="tight")
    #                     else:
    #                         plt.savefig("%s-%s-data-%s-fit-K%d.png" % (model_tag, cv_tag, self.category[i], K), dpi=200, bbox_inches="tight")
    #                     # plt.show()
    #                     plt.close()



    #     print "Corr plot - var_xyz, var_w, red_z - ELG only"
    #     num_cat = 1
    #     num_vars = 5
    #     lims = [self.lim_x, self.lim_y, self.lim_z, self.lim_w, self.lim_redz]
    #     binws = [self.dx, self.dy, self.dz, self.dw, self.dred_z]
    #     var_names = [self.var_x_name, self.var_y_name, self.var_z_name, self.var_w_name, self.red_z_name]
    #     lines = [self.var_x_lines, self.var_y_lines, self.var_z_lines, self.var_w_lines, self.redz_lines]

    #     iplot = np.copy(self.iELG) & self.iTrain
    #     i = 2 # For category
    #     variables = [[self.var_x[iplot], self.var_y[iplot], self.var_z[iplot], self.var_w[iplot], self.red_z[iplot]]]
    #     weights = [self.w[iplot]/self.area_train]

    #     MODELS = self.MODELS[i] # Take the model for the category.
    #     # Plotting the fits
    #     for j, var_num_tuple in enumerate(MODELS.keys()): # For each selection of variables
    #         if len(var_num_tuple) < 5: # Only plot the last models.
    #             pass
    #         else:
    #             # Models corresponding to the tuples
    #             ms = MODELS[var_num_tuple]
    #             for K in ms.keys(): # For each component number tried                        
    #                 print "K: %d" % K                
    #                 # Fits
    #                 m = ms[K]
    #                 amps_fit  = m["amps"]
    #                 means_fit  = m["means"]
    #                 covs_fit = m["covs"]        
        
    #                 fig, ax_list = plt.subplots(num_vars, num_vars, figsize=(35, 35))
    #                 # Corr plots without annotation
    #                 ax_dict = make_corr_plots(ax_list, num_cat, num_vars, variables, lims, binws,\
    #                                           var_names, weights, lines=lines, category_names=[self.category[i]],\
    #                                           pt_sizes=None, colors=None, ft_size_legend = 15, lw_dot=2, hist_normed=True,\
    #                                           plot_MoG_general=True, var_num_tuple=var_num_tuple, amps_general=amps_fit,\
    #                                           means_general=means_fit, covs_general=covs_fit, color_general="red", cum_contour=cum_contour)
    #                 plt.tight_layout()
    #                 if cum_contour:
    #                     plt.savefig("%s-%s-data-%s-fit-K%d-cum_contour.png" % (model_tag, cv_tag, self.category[i], K), dpi=200, bbox_inches="tight")
    #                 else:
    #                     plt.savefig("%s-%s-data-%s-fit-K%d.png" % (model_tag, cv_tag, self.category[i], K), dpi=200, bbox_inches="tight")
    #                 # plt.show()
    #                 plt.close()

    #     return



    # def gen_covar(self, ifit, ND=5):
    #     """
    #     Covariance matrix in the original grz-oii-redz space is diagonal.
    #     """
    #     Nsample = np.sum(ifit)
    #     Covar = np.zeros((Nsample, ND, ND))        

    #     var_err_list = self.zf_err[ifit], self.rf_err[ifit], self.gf_err[ifit], self.oii_err[ifit], np.zeros(np.sum(ifit))

    #     for i in range(Nsample):
    #         tmp = []
    #         for j in range(ND):
    #             tmp.append(var_err_list[j][i]**2) # var = err^2
    #         Covar[i] = np.diag(tmp)
        
    #     return Covar






# class model3(parent_model):
#     """
#     parametrization: asinh mag g-r (x), asinh mag g-z (y), asinh mag g-oii (z), redz, 
#     and gmag(which is practically asinh mag g)

#     Note that all models were trained using g [22, 24] sample.
#     """
#     def __init__(self, sub_sample_num):
#         parent_model.__init__(self, sub_sample_num, tag="_model3")      

#         # Re-parametrizing variables
#         self.var_x, self.var_y, self.var_z, self.gmag =\
#             self.var_reparam(self.gflux, self.rflux, self.zflux, self.oii) 

#         # Plot variables
#         # var limits
#         self.lim_x = [-1.25, 5.] # g-z
#         self.lim_y = [-.75, 2.75] # r-z
#         self.lim_z = [-0, 6] # g-oii
#         self.lim_gmag = [21.5, 24.0]

#         # bin widths
#         self.dx = 0.05
#         self.dy = 0.025
#         self.dz = 0.05
#         self.dgmag = 0.025

#         # var names
#         self.var_x_name = r"$\mu_g - \mu_z$"        
#         self.var_y_name = r"$\mu_g - \mu_r$"  
#         self.var_z_name = r"$\mu_g - \mu_{OII}$"
#         self.red_z_name = r"$\eta$"
#         self.gmag_name  = r"$g$"

#         # var lines
#         self.var_x_lines = np.arange(-0.0, 4.5, 1.)
#         self.var_y_lines = np.arange(-0.0, 2.5, 1.)
#         self.var_z_lines = np.arange(-0.0, 6, 1.)
#         self.redz_lines = [0.6, 1.1, 1.6] # Redz
#         self.gmag_lines = [21, 22, 23, 24]

#         # Fit parameters for pow/broken pow law
#         self.MODELS_pow = [None, None, None]
#         self.MODELS_broken_pow = [None, None, None]
#         self.MODELS_mag_pow = [None, None, None] # magnitude space power law.
#         self.use_dNdm = True # If true, use magnitude space number density distribution.
#         # self.use_broken_dNdf = True # If true, use broken pow law.


#         # Number of components chosen for each category based on the training sample.
#         self.K_best = self.gen_K_best()

#         # ----- MC Sample Variables ----- # 
#         self.area_MC = self.area_train #

#         # FoM value options.
#         # "flat": Assign 1.0 to all objects that meet DESI criteria
#         # "NoOII": Assign 1.0 to all objects with z>0.6.
#         self.FoM_option = "flat"

#         # Flux range to draw the sample from. Slightly larger than the range we are interested.
#         self.fmin_MC = mag2flux(24.5) # Note that around 23.8, the power law starts to break down.
#         self.fmax_MC = mag2flux(19.5)
#         self.fcut = mag2flux(24.) # After noise addition, we make a cut at 24.
#         # Original sample.
#         # 0: NonELG, 1: NoZ, 2: ELG
#         self.NSAMPLE = [None, None, None]
#         self.gflux0 = [None, None, None] # 0 for original
#         self.rflux0 = [None, None, None] # 0 for original
#         self.zflux0 = [None, None, None] # 0 for original
#         self.oii0 = [None, None, None] # Although only ELG class has oii and redz, for consistency, we have three elements lists.
#         self.redz0 = [None, None, None]
#         # Default noise levels
#         self.glim_err = 23.8
#         self.rlim_err = 23.4
#         self.zlim_err = 22.4
#         self.oii_lim_err = 8 # 7 sigma
#         # Noise seed. err_seed ~ N(0, 1). This can be transformed by scaling appropriately.
#         self.g_err_seed = [None, None, None] # Error seed.
#         self.r_err_seed = [None, None, None] # Error seed.
#         self.z_err_seed = [None, None, None] # Error seed.
#         self.oii_err_seed = [None, None, None] # Error seed.
#         # Noise convolved values
#         self.gflux_obs = [None, None, None] # obs for observed
#         self.rflux_obs = [None, None, None] # obs for observed
#         self.zflux_obs = [None, None, None] # obs for observed
#         self.oii_obs = [None, None, None] # Although only ELG class has oii and redz, for consistency, we have three elements lists.
#         # Importance weight: Used when importance sampling is asked for
#         self.iw = [None, None, None]
#         self.iw0 = [None, None, None] # 0 denotes intrinsic sample

#         # Mag Power law from which to generate importance samples.
#         self.alpha_q = [9, 20, 20]
#         self.A_q = [1, 1, 1] # This information is not needed.
#         # These are not really used anymore.

#         # For MoG
#         self.sigma_proposal = 1.5 # sigma factor for the proposal        

#         # FoM per sample. Note that FoM depends on the observed property such as OII.
#         self.FoM_obs = [None, None, None]

#         # Observed final distributions
#         self.var_x_obs = [None, None, None] # g-z
#         self.var_y_obs = [None, None, None] # g-r
#         self.var_z_obs = [None, None, None] # g-oii
#         self.redz_obs = [None, None, None]        
#         self.gmag_obs = [None, None, None]

#         # Cell number
#         self.cell_number_obs = [None, None, None]

#         # Selection grid limits and number of bins 
#         # var_x, var_y, gmag. Width (0.01, 0.01, 0.01)
#         self.var_x_limits = [-.25, 3.5] # g-z
#         self.var_y_limits = [-0.6, 1.5] # g-r
#         self.gmag_limits = [19.5, 24.]
#         self.num_bins = [375, 210, 450]

#         # Number of pixels width to be used during Gaussian smoothing.
#         self.sigma_smoothing = [5., 5., 5.]
#         self.sigma_smoothing_limit = 5

#         # Cell_number in selection
#         self.cell_select = None

#         # Desired nubmer of objects
#         self.num_desired = 2400

#         # Regularization number when computing utility
#         self.frac_regular = 0.05

#         # Fraction of NoZ objects that we expect to be good
#         self.f_NoZ = 0.25

#         # FoM values for individual NoZ and NonELG objects.
#         self.FoM_NoZ = 0.25
#         self.FoM_NonELG = 0.0

#         # Exernal calibration data historgram
#         self.MD_hist_N_cal_flat = None        




#     def var_reparam(self, gflux, rflux, zflux, oii = None):
#         """
#         Given the input variables return the model3 parametrization as noted above.
#         """
#         mu_g = flux2asinh_mag(gflux, band = "g")
#         mu_r = flux2asinh_mag(rflux, band = "r")
#         mu_z = flux2asinh_mag(zflux, band = "z")
#         if oii is not None:
#             mu_oii = flux2asinh_mag(oii, band = "oii")
#             return mu_g - mu_z, mu_g - mu_r, mu_g - mu_oii, flux2mag(gflux)
#         else:
#             return mu_g - mu_z, mu_g - mu_r, flux2mag(gflux)

#     def set_area_MC(self, val):
#         self.area_MC = val
#         return

#     def gen_K_best(self):
#         """
#         Return best K number chosen by eye.

#         [K_NonELG, K_NoZ, K_ELG]
#         """
#         K_best = None
#         if self.sub_sample_num == 0: # Full
#             K_best = [4, 2, 5]
#         elif self.sub_sample_num == 1: #F3
#             K_best = [5, 2, 6]
#         elif self.sub_sample_num == 2: #F4
#             K_best = [6, 2, 5]            
#         elif self.sub_sample_num == 3: #CV1
#             K_best = [5, 2, 5]        
#         elif self.sub_sample_num == 4: #2
#             K_best = [6, 2, 5]            
#         elif self.sub_sample_num == 5: #3
#             K_best = [5, 2, 4]            
#         elif self.sub_sample_num == 6: #4
#             K_best = [6, 2, 4]            
#         elif self.sub_sample_num == 7: #5
#             K_best = [7, 2, 5]            
#         elif self.sub_sample_num == 8: #mag1
#             K_best = [7, 2, 4]            
#         elif self.sub_sample_num == 9: #mag2
#             K_best = [6, 2, 4]            
#         elif self.sub_sample_num == 10: #mag3
#             K_best = [4, 2, 7]
#         elif self.sub_sample_num == 11: #F2
#             K_best = [5, 1, 3]

#         return K_best


#     def plot_data(self, model_tag="", cv_tag="", guide=False):
#         """
#         Model 3
#         Use self model/plot variables to plot the data given an external figure ax_list.
#         Save the resulting image using title_str (does not include extension)

#         If guide True, then plot visual guide.
#         """

#         print "Corr plot - var_xyz - all classes together"
#         lims = [self.lim_x, self.lim_y, self.lim_gmag]
#         binws = [self.dx, self.dy, self.dgmag]
#         var_names = [self.var_x_name, self.var_y_name, self.gmag_name]
#         lines = [self.var_x_lines, self.var_y_lines, self.gmag_lines]
#         num_cat = 3
#         num_vars = 3

#         variables = []
#         weights = []            
#         for ibool in [self.iNonELG, self.iNoZ, self.iELG]:
#             iplot = np.copy(ibool) & self.iTrain
#             variables.append([self.var_x[iplot], self.var_y[iplot], self.gmag[iplot]])
#             weights.append(self.w[iplot]/self.area_train)
#         fig, ax_list = plt.subplots(num_vars, num_vars, figsize=(35, 35))
#         ax_dict = make_corr_plots(ax_list, num_cat, num_vars, variables, lims, binws, var_names, weights, lines=lines, category_names=self.category, pt_sizes=None, colors=self.colors, ft_size_legend = 15, lw_dot=2, guide=guide)
#         plt.savefig("%s-%s-data-all.png" % (model_tag, cv_tag), dpi=200, bbox_inches="tight")
#         plt.close()        



#         print "Corr plot - var_xyz - separately"
#         lims = [self.lim_x, self.lim_y, self.lim_gmag]
#         binws = [self.dx, self.dy, self.dgmag]
#         var_names = [self.var_x_name, self.var_y_name, self.gmag_name]
#         lines = [self.var_x_lines, self.var_y_lines, self.gmag_lines]
#         num_cat = 1
#         num_vars = 3


#         for i, ibool in enumerate([self.iNonELG, self.iNoZ, self.iELG]):
#             print "Plotting %s" % self.category[i]                
#             variables = []
#             weights = []                
#             iplot = np.copy(ibool) & self.iTrain
#             variables.append([self.var_x[iplot], self.var_y[iplot], self.gmag[iplot]])
#             weights.append(self.w[iplot]/self.area_train)

#             fig, ax_list = plt.subplots(num_vars, num_vars, figsize=(35, 35))
#             ax_dict = make_corr_plots(ax_list, num_cat, num_vars, variables, lims, binws, var_names, weights=weights, lines=lines, category_names=[self.category[i]], pt_sizes=None, colors=None, ft_size_legend = 15, lw_dot=2, guide=guide)

#             plt.savefig("%s-%s-data-%s.png" % (model_tag, cv_tag, self.category[i]), dpi=200, bbox_inches="tight")
#             plt.close()


#         print "Corr plot - var_xyz, red_z, gmag - ELG only"
#         num_cat = 1
#         num_vars = 5
#         lims = [self.lim_x, self.lim_y, self.lim_z, self.lim_redz, self.lim_gmag]
#         binws = [self.dx, self.dy, self.dz, self.dred_z, self.dgmag]
#         var_names = [self.var_x_name, self.var_y_name, self.var_z_name, self.red_z_name, self.gmag_name]
#         lines = [self.var_x_lines, self.var_y_lines, self.var_z_lines, self.redz_lines, self.gmag_lines]

#         iplot = np.copy(self.iELG) & self.iTrain
#         i = 2 # For category
#         variables = [[self.var_x[iplot], self.var_y[iplot], self.var_z[iplot], self.red_z[iplot], self.gmag[iplot]]]
#         weights = [self.w[iplot]/self.area_train]

#         fig, ax_list = plt.subplots(num_vars, num_vars, figsize=(50, 50))
#         ax_dict = make_corr_plots(ax_list, num_cat, num_vars, variables, lims, binws, var_names, weights=weights, lines=lines, category_names=[self.category[i]], pt_sizes=None, colors=None, ft_size_legend = 15, lw_dot=2, guide=guide)

#         plt.savefig("%s-%s-data-ELG-redz-oii.png" % (model_tag, cv_tag), dpi=200, bbox_inches="tight")
#         plt.close()


#     def fit_MoG(self, NK_list, model_tag="", cv_tag="", cache=False, Niter=5):
#         """
#         Fit MoGs to data. Note that here we only consider fitting to 2 or 4 dimensions.

#         If cache = True, then search to see if there are models already fit and if available use them.
#         """
#         cache_success = False
#         if cache:
#             for i in range(3):
#                 model_fname = "./MODELS-%s-%s-%s.npy" % (self.category[i], model_tag, cv_tag)
#                 if os.path.isfile(model_fname):
#                     self.MODELS[i] = np.load(model_fname).item()
#                     cache_success = True
#                     print "Cached result will be used for MODELS-%s-%s-%s." % (self.category[i], model_tag, cv_tag)

#         if not cache_success: # If cached result was not requested or was searched for but not found.
#             # For NonELG and NoZ
#             ND = 2 # Dimension of model
#             ND_fit = 2 # Number of variables up to which MoG is being proposed
#             for i, ibool in enumerate([self.iNonELG, self.iNoZ]):
#                 print "Fitting MoGs to %s" % self.category[i]
#                 ifit = ibool & self.iTrain
#                 Ydata = np.array([self.var_x[ifit], self.var_y[ifit]]).T
#                 Ycovar = self.gen_covar(ifit, ND=ND)
#                 weight = self.w[ifit]
#                 self.MODELS[i] = fit_GMM(Ydata, Ycovar, ND, ND_fit, NK_list=NK_list, Niter=Niter, fname_suffix="%s-%s-%s" % (self.category[i], model_tag, cv_tag), MaxDIM=True, weight=weight)

#             # For ELG
#             i = 2
#             ND = 4 # Dimension of model
#             ND_fit = 4 # Number of variables up to which MoG is being proposed
#             print "Fitting MoGs to %s" % self.category[i]
#             ifit = self.iELG & self.iTrain
#             Ydata = np.array([self.var_x[ifit], self.var_y[ifit], self.var_z[ifit], self.red_z[ifit]]).T
#             Ycovar = self.gen_covar(ifit, ND=ND)
#             weight = self.w[ifit]
#             self.MODELS[i] = fit_GMM(Ydata, Ycovar, ND, ND_fit, NK_list=NK_list, Niter=Niter, fname_suffix="%s-%s-%s" % (self.category[i], model_tag, cv_tag), MaxDIM=True, weight=weight)

#         return



#     def fit_dNdf(self, model_tag="", cv_tag="", cache=False, Niter=5, bw=0.025):
#         """
#         Model 3
#         Fit mag pow laws
#         """
#         cache_success = False
#         if cache:
#             for i in range(3):
#                 model_fname = "./MODELS-%s-%s-%s-pow.npy" % (self.category[i], model_tag, cv_tag)
#                 if os.path.isfile(model_fname):
#                     self.MODELS_pow[i] = np.load(model_fname)
#                     cache_success = True
#                     print "Cached result will be used for MODELS-%s-%s-%s-pow." % (self.category[i], model_tag, cv_tag)
#         if not cache_success:
#             for i, ibool in enumerate([self.iNonELG, self.iNoZ, self.iELG]):
#                 print "Fitting power law for %s" % self.category[i]
#                 ifit = self.iTrain & ibool
#                 flux = self.gflux[ifit]
#                 weight = self.w[ifit]
#                 self.MODELS_pow[i] = dNdf_fit(flux, weight, bw, mag2flux(self.mag_max), mag2flux(self.mag_min_model), self.area_train, niter = Niter)
#                 np.save("MODELS-%s-%s-%s-pow.npy" % (self.category[i], model_tag, cv_tag), self.MODELS_pow[i])

#         return 


#     def fit_dNdm(self, model_tag="", cv_tag="", cache=False, Niter=5, bw=0.05):
#         """
#         Model 3
#         Fit mag pow laws
#         """
#         cache_success = False
#         if cache:
#             for i in range(3):
#                 model_fname = "./MODELS-%s-%s-%s-mag-pow.npy" % (self.category[i], model_tag, cv_tag)
#                 if os.path.isfile(model_fname):
#                     self.MODELS_mag_pow[i] = np.load(model_fname)
#                     cache_success = True
#                     print "Cached result will be used for MODELS-%s-%s-%s-mag-pow." % (self.category[i], model_tag, cv_tag)
#         if not cache_success:
#             for i, ibool in enumerate([self.iNonELG, self.iNoZ, self.iELG]):
#                 print "Fitting power law for %s" % self.category[i]
#                 ifit = self.iTrain & ibool
#                 flux = self.gflux[ifit]
#                 weight = self.w[ifit]
#                 self.MODELS_mag_pow[i] = dNdm_fit(flux2mag(flux), weight, bw, self.mag_min_model, self.mag_max, self.area_train, niter = Niter)                
#                 np.save("MODELS-%s-%s-%s-mag-pow.npy" % (self.category[i], model_tag, cv_tag), self.MODELS_mag_pow[i])

#         return None



#     def fit_dNdm_broken_pow(self, model_tag="", cv_tag="", cache=False, Niter=5, bw=0.05):
#         """
#         Model 3
#         Fit mag pow laws
#         """
#         cache_success = False
#         if cache:
#             for i in range(3):
#                 model_fname = "./MODELS-%s-%s-%s-mag-broken-pow.npy" % (self.category[i], model_tag, cv_tag)
#                 if os.path.isfile(model_fname):
#                     self.MODELS_mag_pow[i] = np.load(model_fname)
#                     cache_success = True
#                     print "Cached result will be used for MODELS-%s-%s-%s-mag-broken-pow." % (self.category[i], model_tag, cv_tag)
#         if not cache_success:
#             for i, ibool in enumerate([self.iNonELG, self.iNoZ, self.iELG]):
#                 print "Fitting broken power law for %s" % self.category[i]
#                 ifit = self.iTrain & ibool
#                 flux = self.gflux[ifit]
#                 weight = self.w[ifit]
#                 if i == 0:
#                     mag_max = 24.
#                     mag_min = 17.
#                 else:
#                     mag_max = 24.25                 
#                     mag_min = 22.

#                 self.MODELS_mag_pow[i] = dNdm_fit_broken_pow(flux2mag(flux), weight, bw, mag_min, mag_max, self.area_train, niter = Niter)                
#                 np.save("MODELS-%s-%s-%s-mag-broken-pow.npy" % (self.category[i], model_tag, cv_tag), self.MODELS_mag_pow[i])

#         return None




#     def fit_dNdf_broken_pow(self, model_tag="", cv_tag="", cache=False, Niter=5, bw=1e-2):
#         """
#         Model 3
#         Fit mag pow laws
#         """
#         cache_success = False
#         if cache:
#             for i in range(3):
#                 model_fname = "./MODELS-%s-%s-%s-broken-pow.npy" % (self.category[i], model_tag, cv_tag)
#                 if os.path.isfile(model_fname):
#                     self.MODELS_broken_pow[i] = np.load(model_fname)
#                     cache_success = True
#                     print "Cached result will be used for MODELS-%s-%s-%s-broken-pow." % (self.category[i], model_tag, cv_tag)
#         if not cache_success:
#             for i, ibool in enumerate([self.iNonELG, self.iNoZ, self.iELG]):
#                 print "Fitting broken power law for %s" % self.category[i]
#                 ifit = self.iTrain & ibool
#                 flux = self.gflux[ifit]
#                 weight = self.w[ifit]
#                 self.MODELS_broken_pow[i] = dNdf_fit_broken_pow(flux, weight, bw, mag2flux(self.mag_max), mag2flux(self.mag_min_model), self.area_train, niter = Niter)
#                 np.save("MODELS-%s-%s-%s-broken-pow.npy" % (self.category[i], model_tag, cv_tag), self.MODELS_broken_pow[i])

#         return 



#     def gen_sample_intrinsic(self, K_selected=None):
#         """
#         model 3
#         Given MoG x dNdf parameters specified by [amps, means, covs] corresponding to K_selected[i] components
#         and either MODELS_pow or MODELS_broken_pow, return a sample proportional to area.

#         Currently broken power law version is not supported.

#         Importance sampling is always used.
#         """
#         if K_selected is None:
#             K_selected = self.K_best

#         # NonELG, NoZ and ELG
#         for i in range(3):
#             # MoG model
#             MODELS = self.MODELS[i]
#             MODELS = MODELS[MODELS.keys()[0]][K_selected[i]] # We only want the model with K components
#             amps, means, covs = MODELS["amps"], MODELS["means"], MODELS["covs"]

#             if False:# self.use_broken_dNdf:
#                 # Compute the number of sample to draw.
#                 NSAMPLE = int(round(integrate_broken_pow_law(self.MODELS_broken_pow[i], self.fmin_MC, self.fmax_MC, df=1e-3) * self.area_MC))#

#                 # Sample flux
#                 gflux = gen_pow_law_sample(self.fmin_MC, NSAMPLE, self.alpha_q[i], exact=True, fmax=self.fmax_MC, importance_sampling=False)
#                 r_tilde = broken_pow_law(self.MODELS_broken_pow[i], gflux)/pow_law([self.alpha_q[i], self.phi_q[i]], gflux)
#                 self.iw0[i] = (r_tilde/r_tilde.sum()) 

#             else:
#                 # Pow law model
#                 alpha, A = self.MODELS_pow[i]

#                 # Compute the number of sample to draw.
#                 NSAMPLE = int(round(integrate_pow_law(alpha, A, self.fmin_MC, self.fmax_MC) * self.area_MC))#

#                 # Sample flux
#                 gflux = gen_pow_law_sample(self.fmin_MC, NSAMPLE, self.alpha_q[i], exact=True, fmax=self.fmax_MC, importance_sampling=False)
#                 r_tilde = pow_law([alpha, A], gflux)/pow_law([self.alpha_q[i], self.phi_q[i]], gflux)
#                 self.iw0[i] = (r_tilde/r_tilde.sum()) 


#             print "%s sample number: %d" % (self.category[i], NSAMPLE)


#             # Generate Nsample from MoG.
#             MoG_sample, iw = sample_MoG(amps, means, covs, NSAMPLE, importance_sampling=True, factor_importance = self.sigma_proposal)
#             self.iw0[i] *= iw            

#             # For all categories
#             mu_g = flux2asinh_mag(gflux, band = "g")
#             mu_gz, mu_gr = MoG_sample[:,0], MoG_sample[:,1]
#             mu_z = mu_g - mu_gz
#             mu_r = mu_g - mu_gr

#             zflux = asinh_mag2flux(mu_z, band = "z")
#             rflux = asinh_mag2flux(mu_r, band = "r")
            
#             # Saving
#             self.gflux0[i] = gflux
#             self.rflux0[i] = rflux
#             self.zflux0[i] = zflux
#             self.NSAMPLE[i] = NSAMPLE

#             # Gen err seed and save
#             # Also, collect unormalized importance weight factors, multiply and normalize.
#             self.g_err_seed[i], iw = gen_err_seed(self.NSAMPLE[i], sigma=self.sigma_proposal, return_iw_factor=True)
#             # print "g_err_seed importance weights. First 10", iw[]
#             self.iw0[i] *= iw
#             self.r_err_seed[i], iw = gen_err_seed(self.NSAMPLE[i], sigma=self.sigma_proposal, return_iw_factor=True)
#             self.iw0[i] *= iw        
#             self.z_err_seed[i], iw = gen_err_seed(self.NSAMPLE[i], sigma=self.sigma_proposal, return_iw_factor=True)
#             self.iw0[i] *= iw
#             if i==2: #ELG 
#                 mu_goii, redz = MoG_sample[:,2], MoG_sample[:,3]
#                 mu_oii = mu_g - mu_goii
#                 oii = asinh_mag2flux(mu_oii, band = "oii")

#                 # oii error seed
#                 self.oii_err_seed[i], iw = gen_err_seed(self.NSAMPLE[i], sigma=self.sigma_proposal, return_iw_factor=True)
#                 self.iw0[i] *= iw
#                 # Saving
#                 self.redz0[i] = redz
#                 self.oii0[i] = oii
#             self.iw0[i] = (self.iw0[i]/self.iw0[i].sum()) * self.NSAMPLE[i] # Normalization and multiply by the number of samples generated.

#         return



#     def gen_sample_intrinsic_mag(self, K_selected=None):
#         """
#         model 3
#         Given MoG x dNdm parameters specified by [amps, means, covs] corresponding to K_selected[i] components
#         and either MODELS_mag_pow. Broken power law version is used. 

#         Importance sampling is always used.
#         """
#         if K_selected is None:
#             K_selected = self.K_best

#         # NonELG, NoZ and ELG
#         for i in range(3):
#             # MoG model
#             MODELS = self.MODELS[i]
#             MODELS = MODELS[MODELS.keys()[0]][K_selected[i]] # We only want the model with K components
#             amps, means, covs = MODELS["amps"], MODELS["means"], MODELS["covs"]

#             # Compute the number of sample to draw.
#             NSAMPLE = int(integrate_mag_broken_pow_law(self.MODELS_mag_pow[i], flux2mag(self.fmax_MC), flux2mag(self.fmin_MC), area = self.area_MC))
#             # Sample flux ***************************
#             gmag = gen_mag_pow_law_samples([self.A_q[i], self.alpha_q[i]], flux2mag(self.fmax_MC), flux2mag(self.fmin_MC), NSAMPLE)
#             # assert False            
#             r_tilde = mag_broken_pow_law(self.MODELS_mag_pow[i], gmag)/mag_pow_law([self.A_q[i], self.alpha_q[i]], gmag)
#             self.iw0[i] = (r_tilde/r_tilde.sum()) 

#             print "%s sample number: %d" % (self.category[i], NSAMPLE)
#             gflux = mag2flux(gmag)
#             # assert False

#             # Generate Nsample from MoG.
#             MoG_sample, iw = sample_MoG(amps, means, covs, NSAMPLE, importance_sampling=True, factor_importance = self.sigma_proposal)
#             self.iw0[i] *= iw            

#             # For all categories
#             mu_g = flux2asinh_mag(gflux, band = "g")
#             mu_gz, mu_gr = MoG_sample[:,0], MoG_sample[:,1]
#             mu_z = mu_g - mu_gz
#             mu_r = mu_g - mu_gr

#             zflux = asinh_mag2flux(mu_z, band = "z")
#             rflux = asinh_mag2flux(mu_r, band = "r")
            
#             # Saving
#             self.gflux0[i] = gflux
#             self.rflux0[i] = rflux
#             self.zflux0[i] = zflux
#             self.NSAMPLE[i] = NSAMPLE

#             # Gen err seed and save
#             # Also, collect unormalized importance weight factors, multiply and normalize.
#             self.g_err_seed[i], iw = gen_err_seed(self.NSAMPLE[i], sigma=self.sigma_proposal, return_iw_factor=True)
#             # print "g_err_seed importance weights. First 10", iw[]
#             self.iw0[i] *= iw
#             self.r_err_seed[i], iw = gen_err_seed(self.NSAMPLE[i], sigma=self.sigma_proposal, return_iw_factor=True)
#             self.iw0[i] *= iw        
#             self.z_err_seed[i], iw = gen_err_seed(self.NSAMPLE[i], sigma=self.sigma_proposal, return_iw_factor=True)
#             self.iw0[i] *= iw
#             if i==2: #ELG 
#                 mu_goii, redz = MoG_sample[:,2], MoG_sample[:,3]
#                 mu_oii = mu_g - mu_goii
#                 oii = asinh_mag2flux(mu_oii, band = "oii")

#                 # oii error seed
#                 self.oii_err_seed[i], iw = gen_err_seed(self.NSAMPLE[i], sigma=self.sigma_proposal, return_iw_factor=True)
#                 self.iw0[i] *= iw
#                 # Saving
#                 self.redz0[i] = redz
#                 self.oii0[i] = oii
#             self.iw0[i] = (self.iw0[i]/self.iw0[i].sum()) * self.NSAMPLE[i] # Normalization and multiply by the number of samples generated.

#         return        


#     def set_err_lims(self, glim, rlim, zlim, oii_lim):
#         """
#         Set the error characteristics.
#         """
#         self.glim_err = glim
#         self.rlim_err = rlim 
#         self.zlim_err = zlim
#         self.oii_lim_err = oii_lim

#         return



#     def gen_err_conv_sample(self):
#         """
#         model3
#         Given the error properties glim_err, rlim_err, zlim_err, oii_lim_err, add noise to the intrinsic density
#         sample and compute the parametrization.
#         """
#         print "Convolving error and re-parametrizing"        
#         # NonELG, NoZ and ELG
#         for i in range(3):
#             print "%s" % self.category[i]
#             self.gflux_obs[i] = self.gflux0[i] + self.g_err_seed[i] * mag2flux(self.glim_err)/5.
#             self.rflux_obs[i] = self.rflux0[i] + self.r_err_seed[i] * mag2flux(self.rlim_err)/5.
#             self.zflux_obs[i] = self.zflux0[i] + self.z_err_seed[i] * mag2flux(self.zlim_err)/5.

#             # Make flux cut
#             ifcut = self.gflux_obs[i] > self.fcut
#             self.gflux_obs[i] = self.gflux_obs[i][ifcut]
#             self.rflux_obs[i] = self.rflux_obs[i][ifcut]
#             self.zflux_obs[i] = self.zflux_obs[i][ifcut]

#             # Compute model parametrization
#             mu_g = flux2asinh_mag(self.gflux_obs[i], band="g")
#             mu_r = flux2asinh_mag(self.rflux_obs[i], band="r")
#             mu_z = flux2asinh_mag(self.zflux_obs[i], band="z")
#             self.var_x_obs[i] = mu_g - mu_z
#             self.var_y_obs[i] = mu_g - mu_r
#             self.gmag_obs[i] = flux2mag(self.gflux_obs[i])

#             # Updating the importance weight with the cut
#             self.iw[i] = self.iw0[i][ifcut]

#             # Number of samples after the cut.
#             Nsample = self.gmag_obs[i].size

#             # More parametrization to compute for ELGs. Also, compute FoM.
#             if i==2:
#                 # oii parameerization
#                 self.oii_obs[i] = self.oii0[i] + self.oii_err_seed[i] * (self.oii_lim_err/7.) # 
#                 self.oii_obs[i] = self.oii_obs[i][ifcut]
#                 mu_oii = flux2asinh_mag(self.oii_obs[i], band="oii")
#                 self.var_z_obs[i] = mu_g - mu_oii

#                 # Redshift has no uncertainty
#                 self.redz_obs[i] = self.redz0[i][ifcut]

#                 # Gen FoM 
#                 self.FoM_obs[i] = self.gen_FoM(i, Nsample, self.oii_obs[i], self.redz_obs[i])
#             else:
#                 # Gen FoM 
#                 self.FoM_obs[i] = self.gen_FoM(i, Nsample)

#         return


#     def visualize_fit(self, model_tag="", cv_tag="", cat=0, K=1, cum_contour=False, MC=False):
#         """
#         Model 3

#         Make corr plots of a choosen classes with fits overlayed.
#         cat. 0: NonELG, 1: NoZ, 2: ELG

#         Note that number of components for MoG should be specified by the user.

#         If cum_contour is True, then instead of plotting individual component gaussians,
#         plot the cumulative gaussian fit. Also, plot the power law function corresponding
#         to the magnitude dimension.

#         If MC is True, then over-plot the MC sample as well.
#         """
#         ibool_list = [self.iNonELG, self.iNoZ, self.iELG]

#         # Take the real data points.
#         ibool = ibool_list[cat]

#         if cat in [0, 1]: # NonELG or NoZ 
#             if MC:
#                 num_cat = 2 # Training data + MC sample
#             else:
#                 num_cat = 1
#             num_vars = 3

#             lims = [self.lim_x, self.lim_y, self.lim_gmag]
#             binws = [self.dx, self.dy, self.dgmag]
#             var_names = [self.var_x_name, self.var_y_name, self.gmag_name]
#             lines = [self.var_x_lines, self.var_y_lines, self.gmag_lines]

#             # Data variable
#             variables = []
#             weights = []
#             labels = []
#             colors = []
#             alphas = []
#             # MC variable. Note that var_x and var_x_obs have different data structure.
#             if MC:
#                 variables.append([self.var_x_obs[cat], self.var_y_obs[cat], self.gmag_obs[cat]])
#                 weights.append(self.cw_obs[cat]/self.area_MC)
#                 labels.append("MC")
#                 colors.append("red")
#                 alphas.append(0.4)

#             iplot = np.copy(ibool) & self.iTrain
#             variables.append([self.var_x[iplot], self.var_y[iplot], self.gmag[iplot]])
#             weights.append(self.w[iplot]/self.area_train)
#             labels.append(self.category[cat])
#             colors.append("black")
#             alphas.append(1)

#             # Take the model for the category.
#             MODELS = self.MODELS[cat] 
#             var_num_tuple = MODELS.keys()[0]
#             m = MODELS[var_num_tuple][K] # Plot only the case requested.
#             amps_fit  = m["amps"]
#             means_fit  = m["means"]
#             covs_fit = m["covs"]
#             MODELS_pow = self.MODELS_pow[cat] # Power law for magnitude.

#             # Plotting the fits
#             fig, ax_list = plt.subplots(num_vars, num_vars, figsize=(25, 25))
#             # Corr plots without annotation
#             ax_dict = make_corr_plots(ax_list, num_cat, num_vars, variables, lims, binws,\
#                                       var_names, weights, lines=lines, category_names=labels,\
#                                       pt_sizes=None, colors=colors, ft_size_legend = 15, lw_dot=2, hist_normed=True,\
#                                       plot_MoG_general=True, var_num_tuple=var_num_tuple, amps_general=amps_fit, alphas=alphas,\
#                                       means_general=means_fit, covs_general=covs_fit, color_general="blue", cum_contour=cum_contour,\
#                                       plot_pow=True, pow_model=MODELS_pow, pow_var_num=2)
#             plt.tight_layout()
#             if cum_contour:
#                 plt.savefig("%s-%s-data-%s-fit-K%d-cum-contour.png" % (model_tag, cv_tag, self.category[cat], K), dpi=200, bbox_inches="tight")
#             else:
#                 plt.savefig("%s-%s-data-%s-fit-K%d.png" % (model_tag, cv_tag, self.category[cat], K), dpi=200, bbox_inches="tight")
#             # plt.show()
#             plt.close()


#         else:
#             if MC:
#                 num_cat = 2 # Training data + MC sample
#             else:
#                 num_cat = 1
#             num_vars = 5

#             lims = [self.lim_x, self.lim_y, self.lim_z, self.lim_redz, self.lim_gmag]
#             binws = [self.dx, self.dy, self.dz, self.dred_z, self.dgmag]
#             var_names = [self.var_x_name, self.var_y_name, self.var_z_name, self.red_z_name, self.gmag_name]
#             lines = [self.var_x_lines, self.var_y_lines, self.var_z_lines, self.redz_lines, self.gmag_lines]


#             variables = []
#             weights = []
#             labels = []
#             colors = []
#             alphas = []

#             # MC variable. Note that var_x and var_x_obs have different data structure.
#             if MC:
#                 variables.append([self.var_x_obs[cat], self.var_y_obs[cat], self.var_z_obs[cat], self.redz_obs[cat], self.gmag_obs[cat]])
#                 weights.append(self.cw_obs[cat]/self.area_MC)
#                 labels.append("MC")
#                 colors.append("red")
#                 alphas.append(0.4)

#             iplot = np.copy(ibool) & self.iTrain
#             variables.append([self.var_x[iplot], self.var_y[iplot], self.var_z[iplot], self.red_z[iplot], self.gmag[iplot]])
#             weights.append(self.w[iplot]/self.area_train)
#             labels.append(self.category[cat])
#             colors.append("black")
#             alphas.append(1)


#             # Take the model for the category.
#             MODELS = self.MODELS[cat] 
#             var_num_tuple = MODELS.keys()[0]
#             m = MODELS[var_num_tuple][K] # Plot only the case requested.
#             amps_fit  = m["amps"]
#             means_fit  = m["means"]
#             covs_fit = m["covs"]

#             MODELS_pow = self.MODELS_pow[cat] # Power law for magnitude.

#             # Plotting the fits
#             fig, ax_list = plt.subplots(num_vars, num_vars, figsize=(35, 35))
#             # Corr plots without annotation
#             ax_dict = make_corr_plots(ax_list, num_cat, num_vars, variables, lims, binws,\
#                                       var_names, weights, lines=lines, category_names=labels,\
#                                       pt_sizes=None, colors=colors, ft_size_legend = 15, lw_dot=2, hist_normed=True,\
#                                       plot_MoG_general=True, var_num_tuple=var_num_tuple, amps_general=amps_fit, alphas=alphas,\
#                                       means_general=means_fit, covs_general=covs_fit, color_general="blue", cum_contour=cum_contour,\
#                                       plot_pow=True, pow_model=MODELS_pow, pow_var_num=4)
#             plt.tight_layout()
#             if cum_contour:
#                 plt.savefig("%s-%s-data-%s-fit-K%d-cum-contour.png" % (model_tag, cv_tag, self.category[cat], K), dpi=200, bbox_inches="tight")
#             else:
#                 plt.savefig("%s-%s-data-%s-fit-K%d.png" % (model_tag, cv_tag, self.category[cat], K), dpi=200, bbox_inches="tight")
#             # plt.show()
#             plt.close()


#         return


#     def gen_FoM(self, cat, Nsample, oii=None, redz=None):
#         """
#         Model3 

#         Give the category number
#         0: NonELG
#         1: NoZ
#         2: ELG
#         compute the appropriate FoM corresponding to each sample.
#         """
#         if cat == 0:
#             return np.ones(Nsample, dtype=float) * self.FoM_NonELG
#         elif cat == 1:
#             return np.ones(Nsample, dtype=float) * self.FoM_NoZ # Some arbitrary number. 25% success rate.
#         elif cat == 2:
#             if (oii is None) or (redz is None):
#                 "You must provide oii AND redz"
#                 assert False
#             else:
#                 if self.FoM_option == "flat":# Flat option
#                     ibool = (oii>8) & (redz > 0.6) # For objects that lie within this criteria
#                     FoM = np.zeros(Nsample, dtype=float)
#                     FoM[ibool] = 1.0
#                 elif self.FoM_option == "NoOII": # NoOII means objects without OII values are also included.
#                     ibool = (redz > 0.6) # For objects that lie within this criteria
#                     FoM = np.zeros(Nsample, dtype=float)
#                     FoM[ibool] = 1.0
#                 elif self.FoM_option == "Linear_redz": # FoM linearly scale with redshift
#                     ibool = (oii>8) & (redz > 0.6) & (redz <1.6) # For objects that lie within this criteria
#                     FoM = np.zeros(Nsample, dtype=float)
#                     FoM[ibool] = 1 + (redz[ibool]-0.6) * 5. # This means redz = 1.6 has FoM of 2.
#                 elif self.FoM_option == "Quadratic_redz": # FoM linearly scale with redshift
#                     ibool = (oii>8) & (redz > 0.6) & (redz <1.6) # For objects that lie within this criteria
#                     FoM = np.zeros(Nsample, dtype=float)
#                     FoM[ibool] = 1 + 10 * (redz[ibool]-0.6) ** 2 # This means redz = 1.6 has FoM of 2.                    

#                 return FoM


#     def gen_covar(self, ifit, ND=4):
#         """
#         Covariance matrix corresponding to the new parametrization.
#         Original parameterization: zf, rf, oii, redz, gf
#         New parameterization is given by the model3.
#         """
#         Nsample = np.sum(ifit)
#         Covar = np.zeros((Nsample, ND, ND))

#         zflux, rflux, gflux, oii, red_z = self.zflux[ifit], self.rflux[ifit], self.gflux[ifit], self.oii[ifit], self.red_z[ifit]
#         var_err_list = [self.zf_err[ifit], self.rf_err[ifit], self.oii_err[ifit], np.zeros(np.sum(Nsample)), self.gf_err[ifit]]

#         # constant factors
#         const = 0.542868
#         # Softening factors for asinh mags
#         b_g = 1.042 * 0.0284297
#         b_r = 1.042 * 0.0421544
#         b_z = 1.042 * 0.122832
#         b_oii= 1.042 * 0.574175        

#         for i in range(Nsample):
#             if ND == 2:
#                 # Construct the original space covariance matrix in 3 x 3 subspace.
#                 tmp = []
#                 for j in [0, 1, 4]:
#                     tmp.append(var_err_list[j][i]**2) # var = err^2
#                 Cx = np.diag(tmp)

#                 g, r, z = gflux[i], rflux[i], zflux[i]
#                 M00, M01, M02 = const/np.sqrt(b_z**2+z**2/4.), 0, -const/(g*np.sqrt(b_g**2+g**2/4.))
#                 M10, M11, M12 = 0, const/np.sqrt(b_r**2+r**2/4.), -const/(g*np.sqrt(b_g**2+g**2/4.))
#                 M = np.array([[M00, M01, M02],
#                             [M10, M11, M12]])
                
#                 Covar[i] = np.dot(np.dot(M, Cx), M.T)
#             elif ND == 4:
#                 # Construct the original space covariance matrix in 5 x 5 subspace.
#                 tmp = []
#                 for j in range(5):
#                     tmp.append(var_err_list[j][i]**2) # var = err^2
#                 Cx = np.diag(tmp)

#                 # Construct the affine transformation matrix.
#                 g, r, z, o = gflux[i], rflux[i], zflux[i], oii[i]
#                 M00, M01, M02, M03, M04 = const/np.sqrt(b_z**2+z**2/4.), 0, 0, 0, -const/(g*np.sqrt(b_g**2+g**2/4.))
#                 M10, M11, M12, M13, M14 = 0, const/np.sqrt(b_r**2+r**2/4.), 0, 0, -const/(g*np.sqrt(b_g**2+g**2/4.))
#                 M20, M21, M22, M23, M24 = 0, 0, const/np.sqrt(b_oii**2+o**2/4.), 0, -const/(g*np.sqrt(b_g**2+g**2/4.))
#                 M30, M31, M32, M33, M34 = 0, 0, 0, 1, 0
                
#                 M = np.array([[M00, M01, M02, M03, M04],
#                                     [M10, M11, M12, M13, M14],
#                                     [M20, M21, M22, M23, M24],
#                                     [M30, M31, M32, M33, M34]])
#                 Covar[i] = np.dot(np.dot(M, Cx), M.T)
#             else: 
#                 print "The input number of variables need to be either 2 or 4."
#                 assert False

#         return Covar


#     def validate_on_DEEP2(self, fnum, model_tag="", cv_tag="", plot_validation=False):
#         """
#         Model 3

#         Given the field number, apply the selection to the DEEP2 training data set.
#         The error corresponding to the field is automatically determined form the data set.
        
#         If plot validation is True, then plot 
#         1) var_x - var_y - gmag corr plot
#             - corr_plots: All selected objects in red and others in black
#             - Hist: Plot prediction, selected, and total original. 
#             The hieght is set at minimum(max(dNdvar_original), 1.5 * max(dNdvar_selected, dNdvar_predicted))
#         2) OII and redshift plots
#             - Redshift plot: Plot all ELGs, all DESI ELGs, all selected DESI, plot predicted. ELGs NP=1 line
#             - OII plot: Plot all OII, plot dotted line at OII=8, plot OII of selected. Plot OII of predicted.

#         Return the following set of numbers
#         0: eff
#         1: eff_pred
#         2: Ntotal
#         3: Ntotal_weighted
#         4: Ntotal_pred
#         5: N_ELG_DESI
#         6: N_ELG_DESI_weighted
#         7: N_ELG_pred
#         8: N_ELG_NonDESI
#         9: N_ELG_NonDESI_weighted
#         10:N_ELG_NonDESI_pred 
#         11: N_NoZ
#         12: N_NoZ_weighted
#         13: N_NoZ_pred
#         14: N_NonELG
#         15: N_NonELG_weighted
#         16: N_NonELG_pred
#         17: N_leftover
#         18:N_leftover_weighted
#         """
#         # Selecting only objects in the field.
#         ifield = (self.field == fnum)
#         area_sample = self.areas[fnum-2]
#         gflux = self.gflux[ifield] 
#         rflux = self.rflux[ifield]
#         zflux = self.zflux[ifield]
#         var_x = self.var_x[ifield]
#         var_y = self.var_y[ifield]
#         gmag = self.gmag[ifield]
#         oii = self.oii[ifield]
#         redz = self.red_z[ifield]
#         w = self.w[ifield]
#         iELG = self.iELG[ifield]
#         iNonELG = self.iNonELG[ifield]
#         iNoZ = self.iNoZ[ifield]
#         ra, dec = self.ra[ifield], self.dec[ifield]

#         # Compute the error characteristic of the field. Median.
#         glim_err = median_mag_depth(self.gf_err[ifield])
#         rlim_err = median_mag_depth(self.rf_err[ifield])
#         zlim_err = median_mag_depth(self.zf_err[ifield])
#         oii_lim_err = 8
#         self.set_err_lims(glim_err, rlim_err, zlim_err, oii_lim_err) 

#         # Convolve error to the intrinsic sample.
#         self.gen_err_conv_sample()

#         # Create the selection.
#         eff_pred, Ntotal_pred, Ngood_pred, N_NonELG_pred, N_NoZ_pred, N_ELG_DESI_pred, N_ELG_NonDESI_pred = self.gen_selection_volume_scipy(gaussian_smoothing=True)

#         # Used for debugging
#         # print self.cell_select.size
#         # print self.cell_select
#         # print eff_pred, Ntotal_pred, Ngood_pred, N_NonELG_pred, N_NoZ_pred, N_ELG_DESI_pred, N_ELG_NonDESI_pred

#         # Apply the selection.
#         iselected = self.apply_selection(gflux, rflux, zflux)

#         # ----- Selection validation ----- #
#         # Compute Ntotal and eff
#         Ntotal = np.sum(iselected)/area_sample
#         Ntotal_weighted = np.sum(w[iselected])/area_sample



#         # Boolean vectors
#         iELG_DESI = (oii>8) & (redz>0.6) & (redz<1.6) & iELG
#         iselected_ELG_DESI = iselected & iELG_DESI
#         N_ELG_DESI = np.sum(iselected_ELG_DESI)/area_sample
#         N_ELG_DESI_weighted = np.sum(w[iselected_ELG_DESI])/area_sample

#         iselected_ELG_NonDESI = iselected & ((oii<8) & (redz>0.6) & (redz<1.6)) & iELG
#         N_ELG_NonDESI = np.sum(iselected_ELG_NonDESI)/area_sample
#         N_ELG_NonDESI_weighted = np.sum(w[iselected_ELG_NonDESI])/area_sample

#         iselected_NonELG = iselected & iNonELG
#         N_NonELG = np.sum(iselected_NonELG)/area_sample
#         N_NonELG_weighted = np.sum(w[iselected_NonELG])/area_sample

#         iselected_NoZ = iselected & iNoZ
#         N_NoZ = np.sum(iselected_NoZ)/area_sample
#         N_NoZ_weighted = np.sum(w[iselected_NoZ])/area_sample

#         # Left over?
#         iselected_leftover = np.logical_and.reduce((~iselected_ELG_DESI, ~iselected_ELG_NonDESI, ~iselected_NonELG, ~iselected_NoZ, iselected))
#         N_leftover = np.sum(iselected_leftover)/area_sample
#         N_leftover_weighted = np.sum(w[iselected_leftover])/area_sample

#         # Efficiency
#         eff = (N_ELG_DESI_weighted+self.f_NoZ*N_NoZ_weighted)/float(Ntotal_weighted)

#         print "Raw/Weigthed/Predicted number of selection"
#         print "----------"
#         print "DESI ELGs: %.1f, %.1f, %.1f" % (N_ELG_DESI, N_ELG_DESI_weighted, N_ELG_DESI_pred)
#         print "NonDESI ELGs: %.1f, %.1f, %.1f" % (N_ELG_NonDESI, N_ELG_NonDESI_weighted, N_ELG_NonDESI_pred)
#         print "NoZ: %.1f, %.1f, %.1f" % (N_NoZ, N_NoZ_weighted, N_NoZ_pred)
#         print "NonELG: %.1f, %.1f, %.1f" % (N_NonELG, N_NonELG_weighted, N_NonELG_pred)
#         print "Poorly characterized objects (not included in density modeling, no prediction): %.1f, %.1f, NA" % (N_leftover, N_leftover_weighted)
#         print "----------"
#         print "Total based on individual parts: NA, %.1f, NA" % ((N_NonELG_weighted + N_NoZ_weighted+ N_ELG_DESI_weighted+ N_ELG_NonDESI_weighted+N_leftover_weighted))        
#         print "Total number: %.1f, %.1f, %.1f" % (Ntotal, Ntotal_weighted, Ntotal_pred)
#         print "----------"
#         print "Efficiency, weighted vs. prediction (DESI/Ntotal): %.3f, %.3f" % (eff, eff_pred)

#         if plot_validation:
#             print "Corr plot - var_xyz"

#             # 1) var_x - var_y - gmag corr plot
#             #     - corr_plots: All selected objects in red and others in black
#             #     - Hist: Plot prediction, selected, and total original. 
#             #     The hieght is set at minimum(max(dNdvar_original), 1.5 * max(dNdvar_selected, dNdvar_predicted))

#             lims = [self.lim_x, self.lim_y, self.lim_gmag]
#             binws = [self.dx, self.dy, self.dgmag]
#             var_names = [self.var_x_name, self.var_y_name, self.gmag_name]
#             lines = [self.var_x_lines, self.var_y_lines, self.gmag_lines]
#             num_cat = 2
#             num_vars = 3
#             colors = ["black", "red"]
#             categories = ["Total", "Selected"]
#             pt_sizes = [5, 20]

#             variables = []
#             weights = []
#             hist_types = ["step", "stepfilled"]

#             for ibool in [np.ones(var_x.size, dtype=bool), iselected]:
#                 variables.append([var_x[ibool], var_y[ibool], gmag[ibool]])
#                 weights.append(w[ibool]/area_sample)

#             fig, ax_list = plt.subplots(num_vars, num_vars, figsize=(35, 35))
#             ax_dict = make_corr_plots(ax_list, num_cat, num_vars, variables, lims, binws, var_names, weights, lines=lines,\
#             category_names=categories, pt_sizes=pt_sizes, colors=colors, ft_size_legend = 15, lw_dot=2, hist_types = hist_types)
#             plt.savefig("%s-%s-DEEP2-F%d-validation-plot-corr.png" % (model_tag, cv_tag, fnum), dpi=200, bbox_inches="tight")
#             plt.close()



#             # 2) OII and redshift plots
#             #     - Redshift plot: Plot all ELGs, all DESI ELGs, all selected DESI, plot predicted. ELGs NP=1 line
#             #     - OII plot: Plot all OII, plot dotted line at OII=8, plot OII of selected. Plot OII of predicted.

#             # np=1 Line
#             dz = 0.025
#             doii = 0.5
#             x, y = np1_line(dz=dz)        
#             redz_bins = np.arange(0.5, 1.7, dz)
#             oii_bins = np.arange(0, 50, doii)

#             histtypes = ["step", "stepfilled", "step"]
#             colors = ["black", "black", "red"]
#             alphas = [1, 0.5, 1.]
#             labels = ["All", "All DESI", "Selected DESI"]

#             # All ELGs, DESI ELGs, selected DESI ELGs
#             fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10)) 
#             for i, ibool in enumerate([iELG, iELG_DESI, iselected_ELG_DESI]):
#                 wobjs = np.sum(w[ibool])
#                 # Redshfit
#                 ax1.hist(redz[ibool], bins=redz_bins, weights=w[ibool]/float(area_sample),\
#                  label="%s: %d" % (labels[i], wobjs/float(area_sample)),\
#                  histtype=histtypes[i], lw=2.5, color=colors[i], alpha=alphas[i])

#                 # OII
#                 ax2.hist(oii[ibool], bins=oii_bins, weights=w[ibool]/float(area_sample),\
#                  label="%s: %d" % (labels[i], wobjs/float(area_sample)),\
#                  histtype=histtypes[i], lw=2.5, color=colors[i], alpha=alphas[i])                

#             ax1.plot(x, y, c="green", lw=3, ls="--")            
#             ax1.legend(loc="upper right", fontsize=20)
#             ax1.set_ylim([0, 400])
#             ax1.set_xlim([0.5, 1.7])

#             ax2.legend(loc="upper right", fontsize=20)
#             ax2.set_ylim([0, 200])
#             ax2.set_xlim([0, 40])  

#             # Save
#             plt.suptitle("dNdz and dNdOII. Density per sq. deg.", fontsize=25)
#             plt.savefig("%s-%s-DEEP2-F%d-validation-plot-redz-OII.png" % (model_tag, cv_tag, fnum), dpi=200, bbox_inches="tight")
#             plt.close()

#         return eff, eff_pred, Ntotal, Ntotal_weighted, Ntotal_pred, N_ELG_DESI, N_ELG_DESI_weighted, N_ELG_DESI_pred,\
#          N_ELG_NonDESI, N_ELG_NonDESI_weighted, N_ELG_NonDESI_pred, N_NoZ, N_NoZ_weighted, N_NoZ_pred,\
#          N_NonELG, N_NonELG_weighted, N_NonELG_pred, N_leftover, N_leftover_weighted, ra[iselected], dec[iselected]


#     def gen_selection_volume_scipy(self, gaussian_smoothing=True, selection_ext=None, Ndesired_var=None):
#         """
#         Model 3

#         Given the generated sample (intrinsic val + noise), generate a selection volume,
#         using kernel approximation to the number density. That is, when tallying up the 
#         number of objects in each cell, use a gaussian kernel centered at the cell where
#         the particle happens to fall.

#         This version is different from the vanila version with kernel option in that
#         the cross correlation or convolution is done using scipy convolution function.

#         Note we don't alter the generated sample in any way.
        
#         Strategy:
#             - Construct a multi-dimensional histogram.
#             - Perform FFT convolution with a gaussian kernel. Use padding.
#             - Given the resulting convolved MD histogram, we can flatten it and order them according to
#             utility. Currently, the utility is FoM divided by Ntot.
#             - We can either predict the number density to define a cell of selected cells 
#             or remember the order of the entire (or half of) the cells. Then when selection is applied
#             we can include objects up to the number we want by adjust the utility threshold.
#             This would require remember the number density of all the objects.

#         If gaussian_smoothing, then the filtering is applied to the MD histograms.
#             If smoothing is asked for, the selection region is computed based on the smoothed array
#             But the evaluation is still done on MC array.        

#         If selection_ext is not None, then to the generated data is applied external selction specfieid by the cell numbers
#         and the resulting selection statistics is reported.

#         If Ndesired_var is not None, then sets all the other flags to False. Takes in an array of desired number densities and 
#         outputs an MD array that summarize the predictions corresponding to input desired numbers.
#         """

#         if Ndesired_var is not None:
#             selection_ext = None

#         # Create MD histogarm of each type of objects. 
#         # 0: NonELG, 1: NoZ, 2: ELG
#         MD_hist_N_NonELG, MD_hist_N_NoZ, MD_hist_N_ELG_DESI, MD_hist_N_ELG_NonDESI = None, None, None, None
#         MD_hist_N_FoM = None # Tally of FoM corresponding to all objects in the category.
#         MD_hist_N_good = None # Tally of only good objects. For example, DESI ELGs.
#         MD_hist_N_total = None # Tally of all objects.

#         print "Start of computing selection region."
#         print "Constructing histograms."
#         start = time.time()
#         # NonELG
#         i = 0
#         samples = np.array([self.var_x_obs[i], self.var_y_obs[i], self.gmag_obs[i]]).T
#         MD_hist_N_NonELG, edges = np.histogramdd(samples, bins=self.num_bins, range=[self.var_x_limits, self.var_y_limits, self.gmag_limits], weights=self.iw[i])
#         FoM_tmp, _ = np.histogramdd(samples, bins=self.num_bins, range=[self.var_x_limits, self.var_y_limits, self.gmag_limits], weights=self.FoM_obs[i]*self.iw[i])
#         MD_hist_N_FoM = FoM_tmp
#         MD_hist_N_total = np.copy(MD_hist_N_NonELG)

#         # NoZ
#         i=1
#         samples = np.array([self.var_x_obs[i], self.var_y_obs[i], self.gmag_obs[i]]).T
#         MD_hist_N_NoZ, _ = np.histogramdd(samples, bins=self.num_bins, range=[self.var_x_limits, self.var_y_limits, self.gmag_limits], weights=self.iw[i])
#         FoM_tmp, _ = np.histogramdd(samples, bins=self.num_bins, range=[self.var_x_limits, self.var_y_limits, self.gmag_limits], weights=self.FoM_obs[i]*self.iw[i])
#         MD_hist_N_FoM += FoM_tmp
#         MD_hist_N_good = self.f_NoZ * MD_hist_N_NoZ
#         MD_hist_N_total += MD_hist_N_NoZ

#         # ELG (DESI and NonDESI)
#         i=2
#         samples = np.array([self.var_x_obs[i], self.var_y_obs[i], self.gmag_obs[i]]).T
#         w_DESI = (self.redz_obs[i]>0.6) & (self.redz_obs[i]<1.6) & (self.oii_obs[i]>8) # Only objects in the correct redshift and OII ranges.
#         w_NonDESI = (self.redz_obs[i]>0.6) & (self.redz_obs[i]<1.6) & (self.oii_obs[i]<8) # Only objects in the correct redshift and OII ranges.
#         MD_hist_N_ELG_DESI, _ = np.histogramdd(samples, bins=self.num_bins, range=[self.var_x_limits, self.var_y_limits, self.gmag_limits], weights=w_DESI*self.iw[i])
#         MD_hist_N_ELG_NonDESI, _ = np.histogramdd(samples, bins=self.num_bins, range=[self.var_x_limits, self.var_y_limits, self.gmag_limits], weights=w_NonDESI*self.iw[i])
#         FoM_tmp, _ = np.histogramdd(samples, bins=self.num_bins, range=[self.var_x_limits, self.var_y_limits, self.gmag_limits], weights=self.FoM_obs[i]*self.iw[i])
#         MD_hist_N_FoM += FoM_tmp
#         MD_hist_N_good += MD_hist_N_ELG_DESI
#         MD_hist_N_total += MD_hist_N_ELG_DESI 
#         MD_hist_N_total += MD_hist_N_ELG_NonDESI
#         print "Time taken: %.2f seconds" % (time.time() - start)

#         if gaussian_smoothing:
#             # Note that we only need to smooth the quantities used for making decisiosns.
#             print "Applying gaussian smoothing."
#             start = time.time()
#             # Applying Gaussian filtering
#             # MD_hist_N_NonELG_decision = np.zeros_like(MD_hist_N_NonELG)
#             # MD_hist_N_NoZ_decision = np.zeros_like(MD_hist_N_NoZ)
#             # MD_hist_N_ELG_DESI_decision = np.zeros_like(MD_hist_N_ELG_DESI)
#             # MD_hist_N_ELG_NonDESI_decision = np.zeros_like(MD_hist_N_ELG_NonDESI)
#             MD_hist_N_FoM_decision = np.zeros_like(MD_hist_N_FoM) # Tally of FoM corresponding to all objects in the category.
#             # MD_hist_N_good_decision = np.zeros_like(MD_hist_N_good) # Tally of only good objects. For example, DESI ELGs.
#             MD_hist_N_total_decision = np.zeros_like(MD_hist_N_total) # Tally of all objects.            
#             # gaussian_filter(MD_hist_N_NonELG, self.sigma_smoothing, order=0, output=MD_hist_N_NonELG, mode='constant', cval=0.0, truncate=self.sigma_smoothing_limit)
#             # gaussian_filter(MD_hist_N_NoZ, self.sigma_smoothing, order=0, output=MD_hist_N_NoZ, mode='constant', cval=0.0, truncate=self.sigma_smoothing_limit)
#             # gaussian_filter(MD_hist_N_ELG_DESI, self.sigma_smoothing, order=0, output=MD_hist_N_ELG_DESI, mode='constant', cval=0.0, truncate=self.sigma_smoothing_limit)
#             # gaussian_filter(MD_hist_N_ELG_NonDESI, self.sigma_smoothing, order=0, output=MD_hist_N_ELG_NonDESI, mode='constant', cval=0.0, truncate=self.sigma_smoothing_limit)
#             gaussian_filter(MD_hist_N_FoM, self.sigma_smoothing, order=0, output=MD_hist_N_FoM_decision, mode='constant', cval=0.0, truncate=self.sigma_smoothing_limit)
#             # gaussian_filter(MD_hist_N_good, self.sigma_smoothing, order=0, output=MD_hist_N_good, mode='constant', cval=0.0, truncate=self.sigma_smoothing_limit)
#             gaussian_filter(MD_hist_N_total, self.sigma_smoothing, order=0, output=MD_hist_N_total_decision, mode='constant', cval=0.0, truncate=self.sigma_smoothing_limit)
#             print "Time taken: %.2f seconds" % (time.time() - start)        
#         else:
#             # MD_hist_N_NonELG_decision = MD_hist_N_NonELG
#             # MD_hist_N_NoZ_decision = MD_hist_N_NoZ
#             # MD_hist_N_ELG_DESI_decision = MD_hist_N_ELG_DESI
#             # MD_hist_N_ELG_NonDESI_decision = MD_hist_N_ELG_NonDESI
#             MD_hist_N_FoM_decision = MD_hist_N_FoM # Tally of FoM corresponding to all objects in the category.
#             # MD_hist_N_good_decision = MD_hist_N_good # Tally of only good objects. For example, DESI ELGs.
#             MD_hist_N_total_decision = MD_hist_N_total # Tally of all objects.            


#         print "Computing magnitude dependent regularization."
#         start = time.time()
#         MD_hist_N_regular = np.zeros_like(MD_hist_N_total)
#         # dNdm - broken pow law version
#         for e in self.MODELS_mag_pow: 
#             m_min, m_max = self.gmag_limits[0], self.gmag_limits[1]
#             m_Nbins = self.num_bins[2]
#             m = np.linspace(m_min, m_max, m_Nbins, endpoint=False)
#             dm = (m_max-m_min)/m_Nbins

#             for i, m_tmp in enumerate(m):
#                 MD_hist_N_regular[:, :, i] += self.frac_regular * integrate_mag_broken_pow_law(e, m_tmp, m_tmp+dm, area=self.area_MC) / np.multiply.reduce((self.num_bins[:2]))
#         # dNdf pow law version
#         # for e in self.MODELS_pow: 
#         #     alpha, A = e
#         #     m_min, m_max = self.gmag_limits[0], self.gmag_limits[1]
#         #     m_Nbins = self.num_bins[2]
#         #     m = np.linspace(m_min, m_max, m_Nbins, endpoint=False)
#         #     dm = (m_max-m_min)/m_Nbins
#         #     dNdm = integrate_pow_law(alpha, A, mag2flux(m+dm), mag2flux(m)) * self.area_MC/ np.multiply.reduce((self.num_bins[:2]))
#         #     for i, n in enumerate(dNdm):
#         #         MD_hist_N_regular[:, :, i] += n * self.frac_regular

#         print "Time taken: %.2f seconds" % (time.time() - start)        

#         print "Computing utility and sorting."
#         start = time.time()        
#         # Compute utility
#         MD_hist_N_total_decision += MD_hist_N_regular
#         MD_hist_N_total += MD_hist_N_regular
#         utility = MD_hist_N_FoM_decision/MD_hist_N_total_decision 

#         # # Fraction of cells filled
#         # frac_filled = np.sum(utility>0)/float(utility.size) * 100
#         # print "Fraction of cells filled: %.1f precent" % frac_filled

#         # Flatten utility array
#         utility_flat = utility.flatten()

#         # Order cells according to utility
#         # This corresponds to cell number of descending order sorted array.
#         idx_sort = (-utility_flat).argsort()
#         print "Time taken: %.2f seconds" % (time.time() - start)        


#         print "Flattening the MD histograms."
#         start = time.time()        
#         # Flatten other arrays.
#         MD_hist_N_NonELG_flat = MD_hist_N_NonELG.flatten()
#         MD_hist_N_NoZ_flat = MD_hist_N_NoZ.flatten()
#         MD_hist_N_ELG_DESI_flat = MD_hist_N_ELG_DESI.flatten()
#         MD_hist_N_ELG_NonDESI_flat = MD_hist_N_ELG_NonDESI.flatten()
#         MD_hist_N_FoM_flat = MD_hist_N_FoM.flatten()
#         MD_hist_N_good_flat = MD_hist_N_good.flatten()
#         # Decisions are based on *decision* arrays
#         MD_hist_N_total_flat = MD_hist_N_total.flatten()
#         MD_hist_N_total_flat_decision = MD_hist_N_total_decision.flatten()                
#         print "Time taken: %.2f seconds" % (time.time() - start)        

#         # If external selection result is asked for, then perform the selection now before sorting the flattened array.
#         if selection_ext is not None:
#             print "Applying the external selection."
#             start = time.time()                    
#             Ntotal_ext = np.sum(MD_hist_N_total_flat[selection_ext])/float(self.area_MC)
#             Ngood_ext = np.sum(MD_hist_N_good_flat[selection_ext])/float(self.area_MC)
#             N_NonELG_ext = np.sum(MD_hist_N_NonELG_flat[selection_ext])/float(self.area_MC)
#             N_NoZ_ext = np.sum(MD_hist_N_NoZ_flat[selection_ext])/float(self.area_MC)
#             N_ELG_DESI_ext = np.sum(MD_hist_N_ELG_DESI_flat[selection_ext])/float(self.area_MC)
#             N_ELG_NonDESI_ext = np.sum(MD_hist_N_ELG_NonDESI_flat[selection_ext])/float(self.area_MC)
#             eff_ext = (Ngood_ext/float(Ntotal_ext))
#             print "Time taken: %.2f seconds" % (time.time() - start)        


#         # Sort flattened arrays according to utility.
#         print "Sorting the flattened arrays."
#         start = time.time()                            
#         MD_hist_N_NonELG_flat = MD_hist_N_NonELG_flat[idx_sort]
#         MD_hist_N_NoZ_flat = MD_hist_N_NoZ_flat[idx_sort]
#         MD_hist_N_ELG_DESI_flat = MD_hist_N_ELG_DESI_flat[idx_sort]
#         MD_hist_N_ELG_NonDESI_flat = MD_hist_N_ELG_NonDESI_flat[idx_sort]
#         MD_hist_N_FoM_flat = MD_hist_N_FoM_flat[idx_sort]
#         MD_hist_N_good_flat = MD_hist_N_good_flat[idx_sort]
#         MD_hist_N_total_flat = MD_hist_N_total_flat[idx_sort]
#         MD_hist_N_total_flat_decision = MD_hist_N_total_flat_decision[idx_sort]        
#         print "Time taken: %.2f seconds" % (time.time() - start)                                       

#         # Starting from the keep including cells until the desired number is eached.        
#         if Ndesired_var is not None:
#             # Place holder for answer
#             summary_array = np.zeros((Ndesired_var.size, 7))

#             for i, n in enumerate(Ndesired_var):
#                 Ntotal = 0
#                 counter = 0
#                 for ntot in MD_hist_N_total_flat_decision:
#                     if Ntotal > (n * self.area_MC): 
#                         break            
#                     Ntotal += ntot
#                     counter +=1

#                 # Predicted numbers in the selection.
#                 Ntotal = np.sum(MD_hist_N_total_flat[:counter])/float(self.area_MC)
#                 Ngood = np.sum(MD_hist_N_good_flat[:counter])/float(self.area_MC)
#                 N_NonELG = np.sum(MD_hist_N_NonELG_flat[:counter])/float(self.area_MC)
#                 N_NoZ = np.sum(MD_hist_N_NoZ_flat[:counter])/float(self.area_MC)
#                 N_ELG_DESI = np.sum(MD_hist_N_ELG_DESI_flat[:counter])/float(self.area_MC)
#                 N_ELG_NonDESI = np.sum(MD_hist_N_ELG_NonDESI_flat[:counter])/float(self.area_MC)
#                 eff = (Ngood/float(Ntotal))

#                 summary_array[i, :] = np.array([eff, Ntotal, Ngood, N_NonELG, N_NoZ, N_ELG_DESI, N_ELG_NonDESI])

#             return summary_array
#         else: 
#             Ntotal = 0
#             counter = 0
#             for ntot in MD_hist_N_total_flat_decision:
#                 if Ntotal > (self.num_desired * self.area_MC): 
#                     break            
#                 Ntotal += ntot
#                 counter +=1

#             # Predicted numbers in the selection.
#             Ntotal = np.sum(MD_hist_N_total_flat[:counter])/float(self.area_MC)
#             Ngood = np.sum(MD_hist_N_good_flat[:counter])/float(self.area_MC)
#             N_NonELG = np.sum(MD_hist_N_NonELG_flat[:counter])/float(self.area_MC)
#             N_NoZ = np.sum(MD_hist_N_NoZ_flat[:counter])/float(self.area_MC)
#             N_ELG_DESI = np.sum(MD_hist_N_ELG_DESI_flat[:counter])/float(self.area_MC)
#             N_ELG_NonDESI = np.sum(MD_hist_N_ELG_NonDESI_flat[:counter])/float(self.area_MC)
#             eff = (Ngood/float(Ntotal))    
                

#             # Save the selection
#             self.cell_select = np.sort(idx_sort[:counter])

#             # Return the answer
#             if selection_ext is None:
#                 return eff, Ntotal, Ngood, N_NonELG, N_NoZ, N_ELG_DESI, N_ELG_NonDESI                
#             else: 
#                 return eff, Ntotal, Ngood, N_NonELG, N_NoZ, N_ELG_DESI, N_ELG_NonDESI,\
#                 eff_ext, Ntotal_ext, Ngood_ext, N_NonELG_ext, N_NoZ_ext, N_ELG_DESI_ext, N_ELG_NonDESI_ext




#     def gen_selection_volume_ext_cal(self, gaussian_smoothing=True, Ndesired_var=None, fname_cal=""):
#         """
#         Model 3

#         Given the generated sample (intrinsic val + noise), generate a selection volume,
#         using kernel approximation to the number density. That is, when tallying up the 
#         number of objects in each cell, use a gaussian kernel centered at the cell where
#         the particle happens to fall.

#         This version is different from the vanila version with kernel option in that
#         the cross correlation or convolution is done using scipy convolution function.

#         This version is different from *_scipy version as external dataset is used to
#         calibrate the resulting number density of selection.

#         - Construct histogram of generate samples
#         - Construct histogram of external calibration dataset (In future versions, this should be done externally.)
#         - Smooth the MC sample histograms.
#         - Add in the regularization. 
#         - Compute the utility and sort "All" histograms.
#         - Get the last utility threshold from external calibration to get Ndensity 2400.
#         - Compute the predicted number density and precision. (Get the break down if possible.)
#         - Evaluate the selection on DEEP2 F234 data seperately.
        
#         If gaussian_smoothing, then the filtering is applied to the MD histograms.
#             If smoothing is asked for, the selection region is computed based on the smoothed array
#             But the evaluation is still done on MC array.        

#         If Ndesired_var is not None, then takes in an array of desired number densities and 
#         outputs precision and number density based on prediction as well as that given by
#         DEEP2 F234 datasets.
#         """

#         # Create MD histogarm of each type of objects. 
#         # 0: NonELG, 1: NoZ, 2: ELG
#         MD_hist_N_NonELG, MD_hist_N_NoZ, MD_hist_N_ELG_DESI, MD_hist_N_ELG_NonDESI = None, None, None, None
#         MD_hist_N_FoM = None # Tally of FoM corresponding to all objects in the category.
#         MD_hist_N_good = None # Tally of only good objects. For example, DESI ELGs.
#         MD_hist_N_total = None # Tally of all objects.

#         print "Start of computing selection region."
#         print "Constructing histograms."
#         start = time.time()
#         # NonELG
#         i = 0
#         samples = np.array([self.var_x_obs[i], self.var_y_obs[i], self.gmag_obs[i]]).T
#         MD_hist_N_NonELG, edges = np.histogramdd(samples, bins=self.num_bins, range=[self.var_x_limits, self.var_y_limits, self.gmag_limits], weights=self.iw[i])
#         FoM_tmp, _ = np.histogramdd(samples, bins=self.num_bins, range=[self.var_x_limits, self.var_y_limits, self.gmag_limits], weights=self.FoM_obs[i]*self.iw[i])
#         MD_hist_N_FoM = FoM_tmp
#         MD_hist_N_total = np.copy(MD_hist_N_NonELG)

#         # NoZ
#         i=1
#         samples = np.array([self.var_x_obs[i], self.var_y_obs[i], self.gmag_obs[i]]).T
#         MD_hist_N_NoZ, _ = np.histogramdd(samples, bins=self.num_bins, range=[self.var_x_limits, self.var_y_limits, self.gmag_limits], weights=self.iw[i])
#         FoM_tmp, _ = np.histogramdd(samples, bins=self.num_bins, range=[self.var_x_limits, self.var_y_limits, self.gmag_limits], weights=self.FoM_obs[i]*self.iw[i])
#         MD_hist_N_FoM += FoM_tmp
#         MD_hist_N_good = self.f_NoZ * MD_hist_N_NoZ
#         MD_hist_N_total += MD_hist_N_NoZ

#         # ELG (DESI and NonDESI)
#         i=2
#         samples = np.array([self.var_x_obs[i], self.var_y_obs[i], self.gmag_obs[i]]).T
#         w_DESI = (self.redz_obs[i]>0.6) & (self.redz_obs[i]<1.6) & (self.oii_obs[i]>8) # Only objects in the correct redshift and OII ranges.
#         w_NonDESI = (self.redz_obs[i]>0.6) & (self.redz_obs[i]<1.6) & (self.oii_obs[i]<8) # Only objects in the correct redshift and OII ranges.
#         MD_hist_N_ELG_DESI, _ = np.histogramdd(samples, bins=self.num_bins, range=[self.var_x_limits, self.var_y_limits, self.gmag_limits], weights=w_DESI*self.iw[i])
#         MD_hist_N_ELG_NonDESI, _ = np.histogramdd(samples, bins=self.num_bins, range=[self.var_x_limits, self.var_y_limits, self.gmag_limits], weights=w_NonDESI*self.iw[i])
#         FoM_tmp, _ = np.histogramdd(samples, bins=self.num_bins, range=[self.var_x_limits, self.var_y_limits, self.gmag_limits], weights=self.FoM_obs[i]*self.iw[i])
#         MD_hist_N_FoM += FoM_tmp
#         MD_hist_N_good += MD_hist_N_ELG_DESI
#         MD_hist_N_total += MD_hist_N_ELG_DESI 
#         MD_hist_N_total += MD_hist_N_ELG_NonDESI
#         print "Time taken: %.2f seconds" % (time.time() - start)


#         if gaussian_smoothing:
#             # Note that we only need to smooth the quantities used for making decisiosns.
#             print "Applying gaussian smoothing."
#             start = time.time()
#             # Applying Gaussian filtering
#             # MD_hist_N_NonELG_decision = np.zeros_like(MD_hist_N_NonELG)
#             # MD_hist_N_NoZ_decision = np.zeros_like(MD_hist_N_NoZ)
#             # MD_hist_N_ELG_DESI_decision = np.zeros_like(MD_hist_N_ELG_DESI)
#             # MD_hist_N_ELG_NonDESI_decision = np.zeros_like(MD_hist_N_ELG_NonDESI)
#             MD_hist_N_FoM_decision = np.zeros_like(MD_hist_N_FoM) # Tally of FoM corresponding to all objects in the category.
#             # MD_hist_N_good_decision = np.zeros_like(MD_hist_N_good) # Tally of only good objects. For example, DESI ELGs.
#             MD_hist_N_total_decision = np.zeros_like(MD_hist_N_total) # Tally of all objects.            
#             # gaussian_filter(MD_hist_N_NonELG, self.sigma_smoothing, order=0, output=MD_hist_N_NonELG, mode='constant', cval=0.0, truncate=self.sigma_smoothing_limit)
#             # gaussian_filter(MD_hist_N_NoZ, self.sigma_smoothing, order=0, output=MD_hist_N_NoZ, mode='constant', cval=0.0, truncate=self.sigma_smoothing_limit)
#             # gaussian_filter(MD_hist_N_ELG_DESI, self.sigma_smoothing, order=0, output=MD_hist_N_ELG_DESI, mode='constant', cval=0.0, truncate=self.sigma_smoothing_limit)
#             # gaussian_filter(MD_hist_N_ELG_NonDESI, self.sigma_smoothing, order=0, output=MD_hist_N_ELG_NonDESI, mode='constant', cval=0.0, truncate=self.sigma_smoothing_limit)
#             gaussian_filter(MD_hist_N_FoM, self.sigma_smoothing, order=0, output=MD_hist_N_FoM_decision, mode='constant', cval=0.0, truncate=self.sigma_smoothing_limit)
#             # gaussian_filter(MD_hist_N_good, self.sigma_smoothing, order=0, output=MD_hist_N_good, mode='constant', cval=0.0, truncate=self.sigma_smoothing_limit)
#             gaussian_filter(MD_hist_N_total, self.sigma_smoothing, order=0, output=MD_hist_N_total_decision, mode='constant', cval=0.0, truncate=self.sigma_smoothing_limit)
#             print "Time taken: %.2f seconds" % (time.time() - start)        
#         else:
#             # MD_hist_N_NonELG_decision = MD_hist_N_NonELG
#             # MD_hist_N_NoZ_decision = MD_hist_N_NoZ
#             # MD_hist_N_ELG_DESI_decision = MD_hist_N_ELG_DESI
#             # MD_hist_N_ELG_NonDESI_decision = MD_hist_N_ELG_NonDESI
#             MD_hist_N_FoM_decision = MD_hist_N_FoM # Tally of FoM corresponding to all objects in the category.
#             # MD_hist_N_good_decision = MD_hist_N_good # Tally of only good objects. For example, DESI ELGs.
#             MD_hist_N_total_decision = MD_hist_N_total # Tally of all objects.            

#         print "Computing magnitude dependent regularization."
#         start = time.time()
#         MD_hist_N_regular = np.zeros_like(MD_hist_N_total)
#         # dNdm - broken pow law version
#         for e in self.MODELS_mag_pow: 
#             m_min, m_max = self.gmag_limits[0], self.gmag_limits[1]
#             m_Nbins = self.num_bins[2]
#             m = np.linspace(m_min, m_max, m_Nbins, endpoint=False)
#             dm = (m_max-m_min)/m_Nbins

#             for i, m_tmp in enumerate(m):
#                 MD_hist_N_regular[:, :, i] += self.frac_regular * integrate_mag_broken_pow_law(e, m_tmp, m_tmp+dm, area=self.area_MC) / np.multiply.reduce((self.num_bins[:2]))
#         # dNdf pow law version
#         # for e in self.MODELS_pow: 
#         #     alpha, A = e
#         #     m_min, m_max = self.gmag_limits[0], self.gmag_limits[1]
#         #     m_Nbins = self.num_bins[2]
#         #     m = np.linspace(m_min, m_max, m_Nbins, endpoint=False)
#         #     dm = (m_max-m_min)/m_Nbins
#         #     dNdm = integrate_pow_law(alpha, A, mag2flux(m+dm), mag2flux(m)) * self.area_MC/ np.multiply.reduce((self.num_bins[:2]))
#         #     for i, n in enumerate(dNdm):
#         #         MD_hist_N_regular[:, :, i] += n * self.frac_regular

#         print "Time taken: %.2f seconds" % (time.time() - start)        

#         print "Computing utility and sorting."
#         start = time.time()        
#         # Compute utility
#         MD_hist_N_total_decision += MD_hist_N_regular
#         MD_hist_N_total += MD_hist_N_regular
#         utility = MD_hist_N_FoM_decision/MD_hist_N_total_decision 

#         # Flatten utility array
#         utility_flat = utility.flatten()

#         # Order cells according to utility
#         # This corresponds to cell number of descending order sorted array.
#         idx_sort = (-utility_flat).argsort()
#         print "Time taken: %.2f seconds" % (time.time() - start)        


#         print "Flattening the MD histograms."
#         start = time.time()        
#         # Flatten other arrays.
#         MD_hist_N_NonELG_flat = MD_hist_N_NonELG.flatten()
#         MD_hist_N_NoZ_flat = MD_hist_N_NoZ.flatten()
#         MD_hist_N_ELG_DESI_flat = MD_hist_N_ELG_DESI.flatten()
#         MD_hist_N_ELG_NonDESI_flat = MD_hist_N_ELG_NonDESI.flatten()
#         MD_hist_N_FoM_flat = MD_hist_N_FoM.flatten()
#         MD_hist_N_good_flat = MD_hist_N_good.flatten()        
#         # Decisions are based on *decision* arrays
#         MD_hist_N_total_flat = MD_hist_N_total.flatten()
#         MD_hist_N_total_flat_decision = MD_hist_N_total_decision.flatten()                
#         print "Time taken: %.2f seconds" % (time.time() - start)        


#         # Sort flattened arrays according to utility.
#         print "Sorting the flattened arrays."
#         start = time.time()                            
#         MD_hist_N_NonELG_flat = MD_hist_N_NonELG_flat[idx_sort]
#         MD_hist_N_NoZ_flat = MD_hist_N_NoZ_flat[idx_sort]
#         MD_hist_N_ELG_DESI_flat = MD_hist_N_ELG_DESI_flat[idx_sort]
#         MD_hist_N_ELG_NonDESI_flat = MD_hist_N_ELG_NonDESI_flat[idx_sort]
#         MD_hist_N_FoM_flat = MD_hist_N_FoM_flat[idx_sort]
#         MD_hist_N_good_flat = MD_hist_N_good_flat[idx_sort]
#         MD_hist_N_total_flat = MD_hist_N_total_flat[idx_sort]
#         MD_hist_N_total_flat_decision = MD_hist_N_total_flat_decision[idx_sort]
#         # Calibration data histogram.
#         MD_hist_N_cal_flat = self.MD_hist_N_cal_flat[idx_sort]

#         print "Time taken: %.2f seconds" % (time.time() - start)                                       

#         # Starting from the keep including cells until the desired number is eached.        
#         if Ndesired_var is not None:
#             # Place holder for answer
#             summary_array = np.zeros((Ndesired_var.size, 7))

#             for i, n in enumerate(Ndesired_var):
#                 # print "Predicting boundary for Ndensity = %d" % n
#                 Ntotal = 0
#                 counter = 0
#                 for ncell in MD_hist_N_cal_flat:
#                     if Ntotal > n: 
#                         break            
#                     Ntotal += ncell
#                     counter +=1

#                 # Predicted numbers in the selection.
#                 Ntotal = np.sum(MD_hist_N_total_flat[:counter])/float(self.area_MC)
#                 Ngood = np.sum(MD_hist_N_good_flat[:counter])/float(self.area_MC)
#                 N_NonELG = np.sum(MD_hist_N_NonELG_flat[:counter])/float(self.area_MC)
#                 N_NoZ = np.sum(MD_hist_N_NoZ_flat[:counter])/float(self.area_MC)
#                 N_ELG_DESI = np.sum(MD_hist_N_ELG_DESI_flat[:counter])/float(self.area_MC)
#                 N_ELG_NonDESI = np.sum(MD_hist_N_ELG_NonDESI_flat[:counter])/float(self.area_MC)
#                 eff = (Ngood/float(Ntotal))

#                 summary_array[i, :] = np.array([eff, Ntotal, Ngood, N_NonELG, N_NoZ, N_ELG_DESI, N_ELG_NonDESI])

#             return summary_array
#         else: 
#             Ntotal_pred = 0
#             counter = 0
#             for ncell in MD_hist_N_cal_flat:
#                 if Ntotal_pred > self.num_desired:  # MD_hist_N_cal_flat is already normalized.
#                     break            
#                 Ntotal_pred += ncell
#                 counter +=1

#             # Save the selection
#             self.cell_select = np.sort(idx_sort[:counter])

#             # Predicted numbers in the selection.
#             Ntotal_pred = np.sum(MD_hist_N_total_flat[:counter])/float(self.area_MC)
#             Ngood_pred = np.sum(MD_hist_N_good_flat[:counter])/float(self.area_MC)
#             N_NonELG_pred = np.sum(MD_hist_N_NonELG_flat[:counter])/float(self.area_MC)
#             N_NoZ_pred = np.sum(MD_hist_N_NoZ_flat[:counter])/float(self.area_MC)
#             N_ELG_DESI_pred = np.sum(MD_hist_N_ELG_DESI_flat[:counter])/float(self.area_MC)
#             N_ELG_NonDESI_pred = np.sum(MD_hist_N_ELG_NonDESI_flat[:counter])/float(self.area_MC)
#             eff_pred = (Ngood_pred/float(Ntotal_pred))    

#             # ----- Validation on DEEP2 F234 ----- #
#             for fnum in range(2, 5):
#                 # Selecting only objects in the field.
#                 ifield = (self.field == fnum)
#                 area_sample = self.areas[fnum-2]
#                 gflux = self.gflux[ifield] 
#                 rflux = self.rflux[ifield]
#                 zflux = self.zflux[ifield]
#                 var_x = self.var_x[ifield]
#                 var_y = self.var_y[ifield]
#                 gmag = self.gmag[ifield]
#                 oii = self.oii[ifield]
#                 redz = self.red_z[ifield]
#                 w = self.w[ifield]
#                 iELG = self.iELG[ifield]
#                 iNonELG = self.iNonELG[ifield]
#                 iNoZ = self.iNoZ[ifield]
#                 # ra, dec = self.ra[ifield], self.dec[ifield]

#                 # Apply the selection.
#                 iselected = self.apply_selection(gflux, rflux, zflux)

#                 # Compute Ntotal and eff
#                 Ntotal = np.sum(iselected)/area_sample
#                 Ntotal_weighted = np.sum(w[iselected])/area_sample

#                 # Boolean vectors
#                 iELG_DESI = (oii>8) & (redz>0.6) & (redz<1.6) & iELG
#                 iselected_ELG_DESI = iselected & iELG_DESI
#                 N_ELG_DESI = np.sum(iselected_ELG_DESI)/area_sample
#                 N_ELG_DESI_weighted = np.sum(w[iselected_ELG_DESI])/area_sample

#                 iselected_ELG_NonDESI = iselected & ((oii<8) & (redz>0.6) & (redz<1.6)) & iELG
#                 N_ELG_NonDESI = np.sum(iselected_ELG_NonDESI)/area_sample
#                 N_ELG_NonDESI_weighted = np.sum(w[iselected_ELG_NonDESI])/area_sample

#                 iselected_NonELG = iselected & iNonELG
#                 N_NonELG = np.sum(iselected_NonELG)/area_sample
#                 N_NonELG_weighted = np.sum(w[iselected_NonELG])/area_sample

#                 iselected_NoZ = iselected & iNoZ
#                 N_NoZ = np.sum(iselected_NoZ)/area_sample
#                 N_NoZ_weighted = np.sum(w[iselected_NoZ])/area_sample

#                 # Left over?
#                 iselected_leftover = np.logical_and.reduce((~iselected_ELG_DESI, ~iselected_ELG_NonDESI, ~iselected_NonELG, ~iselected_NoZ, iselected))
#                 N_leftover = np.sum(iselected_leftover)/area_sample
#                 N_leftover_weighted = np.sum(w[iselected_leftover])/area_sample

#                 # Efficiency
#                 eff_val = (N_ELG_DESI_weighted+self.f_NoZ*N_NoZ_weighted)/float(Ntotal_weighted)

#                 print "Raw/Weigthed/Predicted number of selection"
#                 print "----------"
#                 print "DESI ELGs: %.1f, %.1f, %.1f" % (N_ELG_DESI, N_ELG_DESI_weighted, N_ELG_DESI_pred)
#                 print "NonDESI ELGs: %.1f, %.1f, %.1f" % (N_ELG_NonDESI, N_ELG_NonDESI_weighted, N_ELG_NonDESI_pred)
#                 print "NoZ: %.1f, %.1f, %.1f" % (N_NoZ, N_NoZ_weighted, N_NoZ_pred)
#                 print "NonELG: %.1f, %.1f, %.1f" % (N_NonELG, N_NonELG_weighted, N_NonELG_pred)
#                 print "Poorly characterized objects (not included in density modeling, no prediction): %.1f, %.1f, NA" % (N_leftover, N_leftover_weighted)
#                 print "----------"
#                 print "Total based on individual parts: NA, %.1f, NA" % ((N_NonELG_weighted + N_NoZ_weighted+ N_ELG_DESI_weighted+ N_ELG_NonDESI_weighted+N_leftover_weighted))        
#                 print "Total number: %.1f, %.1f, %.1f" % (Ntotal, Ntotal_weighted, Ntotal_pred)
#                 print "----------"
#                 print "Efficiency, weighted vs. prediction (DESI/Ntotal): %.3f, %.3f" % (eff_val, eff_pred)            
#                 print "\n\n"

#             # Return the answer
#             return eff_pred, Ntotal_pred, Ngood_pred, N_NonELG_pred, N_NoZ_pred, N_ELG_DESI_pred, N_ELG_NonDESI_pred                




#     def gen_selection_volume(self, use_kernel=False):
#         """
#         Given the generated sample (intrinsic val + noise), generate a selection volume.

#         If use_kernel, when use kernel approximation to the number density calculation.
#         That is, when tallying up the number of objects in each cell, use a gaussian kernel
#         centered at the cell where the particle happens to fall.
        
#         Strategy:
#             Generate cell number for each sample in each category based on var_x, var_y and gmag.
#             The limits are already specified by the class.

#             Given the cell number, we can order all the samples in a category by cell number. (Do this separately)
#             We then create a cell grid and compute Ntotal and FoM corresponding to each cell. 
#             Note that the total number is the sum of completeness weight of objects in the cell,
#             and when computing aggregate FoM, you have to weight it.

#             We then compute utility, say FoM/Ntot, and order the cells according to utility
#             and accept until we have the desired number. The cell number gives our desired selection.
#         """

#         # Compute cell number and order the samples according to the cell number
#         for i in range(3):
#             samples = [self.var_x_obs[i], self.var_y_obs[i], self.gmag_obs[i]]
#             # Generating 
#             self.cell_number_obs[i] = multdim_grid_cell_number(samples, 3, [self.var_x_limits, self.var_y_limits, self.gmag_limits], self.num_bins)

#             # Sorting
#             idx_sort = self.cell_number_obs[i].argsort()
#             self.cell_number_obs[i] = self.cell_number_obs[i][idx_sort]
#             self.cw_obs[i] = self.cw_obs[i][idx_sort] 
#             self.FoM_obs[i] = self.FoM_obs[i][idx_sort]

#             # Unncessary to sort these for computing the selection volume.
#             # However, would be good to self-validate by applying the selection volume generated
#             # to the derived sample to see if you get the proper number density and aggregate FoM.
#             self.var_x_obs[i] = self.var_x_obs[i][idx_sort] 
#             self.var_y_obs[i] = self.var_y_obs[i][idx_sort] 
#             self.gmag_obs[i] = self.gmag_obs[i][idx_sort]
#             if i == 2: # For ELGs 
#                 self.var_z_obs[i] = self.var_z_obs[i][idx_sort] 
#                 self.redz_obs[i] = self.redz_obs[i][idx_sort]
        
#         # Placeholder for cell grid linearized. Cell index corresponds to cell number. 
#         N_cell = np.multiply.reduce(self.num_bins)
#         FoM = np.zeros(N_cell, dtype = float)
#         # Ntotal_cell = np.zeros(N_cell, dtype = float)
#         # N_NoZ_cell = np.zeros(N_cell, dtype = float)
#         # N_ELG_DESI_cell = np.zeros(N_cell, dtype = float)
#         # N_ELG_NonDESI_cell = np.zeros(N_cell, dtype = float)
#         # N_NonELG_cell = np.zeros(N_cell, dtype = float)

#         # Iterate through each sample in all three categories and compute N_categories, N_total and FoM.
#         if use_kernel:
#             # NonELG
#             i=0
#             FoM_tmp, N_NonELG_cell, _ = tally_objects_kernel(N_cell, self.cell_number_obs[i], self.cw_obs[i], self.FoM_obs[i], self.num_bins)
#             FoM += FoM_tmp
#             # NoZ
#             i=1
#             FoM_tmp, N_NoZ_cell, _ = tally_objects_kernel(N_cell, self.cell_number_obs[i], self.cw_obs[i], self.FoM_obs[i], self.num_bins)
#             FoM += FoM_tmp
#             # ELG (DESI and NonDESI)
#             i=2
#             FoM_tmp, N_ELG_all_cell, N_ELG_DESI_cell = tally_objects_kernel(N_cell, self.cell_number_obs[i], self.cw_obs[i], self.FoM_obs[i], self.num_bins)
#             N_ELG_NonDESI_cell = N_ELG_all_cell - N_ELG_DESI_cell
#             FoM += FoM_tmp
#         else:
#             # NonELG
#             i=0
#             FoM_tmp, N_NonELG_cell, _ = tally_objects(N_cell, self.cell_number_obs[i], self.cw_obs[i], self.FoM_obs[i])
#             FoM += FoM_tmp
#             # NoZ
#             i=1
#             FoM_tmp, N_NoZ_cell, _ = tally_objects(N_cell, self.cell_number_obs[i], self.cw_obs[i], self.FoM_obs[i])
#             FoM += FoM_tmp
#             # ELG (DESI and NonDESI)
#             i=2
#             FoM_tmp, N_ELG_all_cell, N_ELG_DESI_cell = tally_objects(N_cell, self.cell_number_obs[i], self.cw_obs[i], self.FoM_obs[i])
#             N_ELG_NonDESI_cell = N_ELG_all_cell - N_ELG_DESI_cell
#             FoM += FoM_tmp

#         # Computing the total and good number of objects.
#         Ntotal_cell = N_NonELG_cell + N_NoZ_cell + N_ELG_all_cell
#         Ngood_cell = self.f_NoZ * N_NoZ_cell + N_ELG_DESI_cell

#         # Compute utility
#         utility = FoM/(Ntotal_cell+ (self.N_regular * self.area_MC / float(np.multiply.reduce(self.num_bins)))) # Note the multiplication by the area.

#         # Order cells according to utility
#         # This corresponds to cell number of descending order sorted array.
#         idx_sort = (-utility).argsort()

#         utility = utility[idx_sort]
#         Ntotal_cell = Ntotal_cell[idx_sort]
#         Ngood_cell = Ngood_cell[idx_sort]
#         N_NonELG_cell = N_NonELG_cell[idx_sort]
#         N_NoZ_cell = N_NoZ_cell[idx_sort]
#         N_ELG_DESI_cell = N_ELG_DESI_cell[idx_sort]
#         N_ELG_NonDESI_cell = N_ELG_NonDESI_cell[idx_sort]

#         # Starting from the keep including cells until the desired number is eached.        
#         Ntotal = 0
#         counter = 0
#         for ntot in Ntotal_cell:
#             if Ntotal > (self.num_desired * self.area_MC): 
#                 break            
#             Ntotal += ntot
#             counter +=1

#         # Predicted numbers in the selection.
#         Ntotal = np.sum(Ntotal_cell[:counter])/float(self.area_MC)
#         Ngood = np.sum(Ngood_cell[:counter])/float(self.area_MC)
#         N_NonELG = np.sum(N_NonELG_cell[:counter])/float(self.area_MC)
#         N_NoZ = np.sum(N_NoZ_cell[:counter])/float(self.area_MC)
#         N_ELG_DESI = np.sum(N_ELG_DESI_cell[:counter])/float(self.area_MC)
#         N_ELG_NonDESI = np.sum(N_ELG_NonDESI_cell[:counter])/float(self.area_MC)

#         # Save the selection
#         self.cell_select = np.sort(idx_sort[:counter])
#         eff = (Ngood/float(Ntotal))

#         return eff, Ntotal, Ngood, N_NonELG, N_NoZ, N_ELG_DESI, N_ELG_NonDESI


#     def apply_selection(self, gflux, rflux, zflux):
#         """
#         Model 3
#         Given gflux, rflux, zflux of samples, return a boolean vector that gives the selection.
#         """
#         mu_g = flux2asinh_mag(gflux, band = "g")
#         mu_r = flux2asinh_mag(rflux, band = "r")
#         mu_z = flux2asinh_mag(zflux, band = "z")

#         var_x = mu_g - mu_z
#         var_y = mu_g - mu_r
#         gmag = flux2mag(gflux)

#         samples = [var_x, var_y, gmag]

#         # Generate cell number 
#         cell_number = multdim_grid_cell_number(samples, 3, [self.var_x_limits, self.var_y_limits, self.gmag_limits], self.num_bins)

#         # Sort the cell number
#         idx_sort = cell_number.argsort()
#         cell_number = cell_number[idx_sort]

#         # Placeholder for selection vector
#         iselect = check_in_arr2(cell_number, self.cell_select)

#         # The last step is necessary in order for iselect to have the same order as the input sample variables.
#         idx_undo_sort = idx_sort.argsort()        
#         return iselect[idx_undo_sort]

#     def cell_select_centers(self):
#         """
#         Return selected cells centers. Model3.
#         """
#         limits = [self.var_x_limits, self.var_y_limits, self.gmag_limits]
#         Ncell_select = self.cell_select.size # Number of cells in the selection
#         centers = [None, None, None]

#         for i in range(3):
#             Xmin, Xmax = limits[i]
#             bin_edges, dX = np.linspace(Xmin, Xmax, self.num_bins[i]+1, endpoint=True, retstep=True)
#             bin_centers = (bin_edges[1:]+bin_edges[:-1])/2.
#             idx = (self.cell_select % np.multiply.reduce(self.num_bins[i:])) //  np.multiply.reduce(self.num_bins[i+1:])
#             centers[i] = bin_centers[idx.astype(int)]

#         return np.asarray(centers).T



#     def gen_select_boundary_slices(self, slice_dir = 2, model_tag="", cv_tag="", centers=None, plot_ext=False,\
#         gflux_ext=None, rflux_ext=None, zflux_ext=None, ibool_ext = None,\
#         var_x_ext=None, var_y_ext=None, gmag_ext=None, use_parameterized_ext=False,\
#         pt_size=10, pt_size_ext=10, alpha_ext=0.5, guide=False, output_sparse=False, increment=10):
#         """
#         Model3

#         Given slice direction, generate slices of boundary

#         0: var_x
#         1: var_y
#         2: gmag

#         If plot_ext True, then plot user supplied external objects.

#         If centers is not None, then use it instead of generating one.

#         If use_parameterized_ext, then the user may provide already parameterized version of the external data points.

#         If guide True, then plot the guide line.

#         If output_sparse=True, then only 10% of the boundaries are plotted and saved.
#         """

#         slice_var_tag = ["mu_gz", "mu_gr", "gmag"]
#         var_names = [self.var_x_name, self.var_y_name, self.gmag_name]

#         if centers is None:
#             centers = self.cell_select_centers()

#         if guide:
#             x_guide, y_guide = gen_guide_line()

#         if plot_ext:
#             if use_parameterized_ext:
#                 if ibool_ext is not None:
#                     var_x_ext = var_x_ext[ibool_ext]
#                     var_y_ext = var_y_ext[ibool_ext]
#                     gmag_ext = gmag_ext[ibool_ext]                
#             else:     
#                 if ibool_ext is not None:
#                     gflux_ext = gflux_ext[ibool_ext]
#                     rflux_ext = rflux_ext[ibool_ext]
#                     zflux_ext = zflux_ext[ibool_ext]

#                 mu_g, mu_r, mu_z = flux2asinh_mag(gflux_ext, band="g"), flux2asinh_mag(rflux_ext, band="r"), flux2asinh_mag(zflux_ext, band="z")
#                 var_x_ext = mu_g-mu_z
#                 var_y_ext = mu_g-mu_r
#                 gmag_ext = flux2mag(gflux_ext)

#             variables = [var_x_ext, var_y_ext, gmag_ext]

#         limits = [self.var_x_limits, self.var_y_limits, self.gmag_limits]        
#         Xmin, Xmax = limits[slice_dir]
#         bin_edges, dX = np.linspace(Xmin, Xmax, self.num_bins[slice_dir]+1, endpoint=True, retstep=True)

#         print slice_var_tag[slice_dir]
#         if output_sparse:
#             iterator = range(0, self.num_bins[slice_dir], increment)
#         else:
#             iterator = range(self.num_bins[slice_dir])

#         for i in iterator: 
#             ibool = (centers[:, slice_dir] < bin_edges[i+1]) & (centers[:, slice_dir] > bin_edges[i])
#             centers_slice = centers[ibool, :]
#             fig = plt.figure(figsize=(7, 7))
#             idx = range(3)
#             idx.remove(slice_dir)
#             plt.scatter(centers_slice[:,idx[0]], centers_slice[:,idx[1]], edgecolors="none", c="green", alpha=0.5, s=pt_size)
#             if plot_ext:
#                 ibool = (variables[slice_dir] < bin_edges[i+1]) & (variables[slice_dir] > bin_edges[i])
#                 plt.scatter(variables[idx[0]][ibool], variables[idx[1]][ibool], edgecolors="none", c="red", s=pt_size_ext, alpha=alpha_ext)
#             plt.xlabel(var_names[idx[0]], fontsize=15)
#             plt.ylabel(var_names[idx[1]], fontsize=15)

#             if guide and (slice_dir==2):
#                 plt.plot(x_guide, y_guide, c="orange", lw = 2)
#             # plt.axis("equal")
#             plt.xlim(limits[idx[0]])
#             plt.ylim(limits[idx[1]])
#             title_str = "%s [%.3f, %.3f]" % (var_names[slice_dir], bin_edges[i], bin_edges[i+1])
#             print i, title_str
#             plt.title(title_str, fontsize=15)
#             plt.savefig("%s-%s-boundary-%s-%d.png" % (model_tag, cv_tag, slice_var_tag[slice_dir], i), bbox_inches="tight", dpi=200)
#         #     plt.show()
#             plt.close()        

#     def set_FoM_option(self, FoM_option):
#         self.FoM_option = FoM_option
#         return None
    
#     def set_f_NoZ(self, fNoZ):
#         self.f_NoZ = fNoZ
#         return None

#     def set_num_desired(self, Ntot):
#         self.num_desired = Ntot
#         return None

#     def load_calibration_data(self):
#         g, r, z, w1, w2, A = load_DR5_calibration()
#         # Asinh magnitude
#         gmag = flux2mag(g)
#         asinh_r = flux2asinh_mag(r, band="r")
#         asinh_g = flux2asinh_mag(g, band="g")
#         asinh_z = flux2asinh_mag(z, band="z")

#         # Variable changes
#         varx = asinh_g - asinh_z
#         vary = asinh_g - asinh_r

#         # Samples
#         samples = np.array([varx, vary, gmag]).T
        

#         hist, _ = np.histogramdd(samples, bins=self.num_bins, range=[self.var_x_limits, self.var_y_limits, self.gmag_limits]) # A is for normalization.
#         self.MD_hist_N_cal_flat = hist.flatten() / float(A)
#         return None
