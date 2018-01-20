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
        # mag_max, mag_min: Magnitue range considered for MC sampling and intrinsic density modeling. 
        # Additional cuts might be made later on, however.
        self.mag_max = 24.25 
        self.mag_min = 17.                

        # Areas of Field 2, 3, and 4
        self.areas = np.load(dir_derived+"spec-area.npy")        

        # Model variables
        self.gflux, self.gf_err, self.rflux, self.rf_err, self.zflux, self.zf_err, self.red_z,\
        self.z_err, self.oii, self.oii_err, self.w, self.field,\
        self.ra, self.dec, self.w1_flux, self.w2_flux, self.w1_err, self.w2_err, self.cn,\
        self.iELG_DESI, self.iNoZ, self.iNoOII, self.iNonELG = self.import_data_DEEP2_full()

        # Training is based on Field 3 and 4 data.
        self.iTrain = np.logical_or(self.field==3, self.field==4)
        self.area_train = np.sum(self.  areas[1:])

        # Test data is base d on Field 2 data.
        self.iTest = self.field == 2
        self.area_test = self.areas[0]

        # Reparameterization
        # asinh mag g-r (x), asinh mag g-z (y), asinh mag g-oii (z), and gmag
        self.var_x, self.var_y, self.var_z, self.gmag =\
            self.var_reparam(self.gflux, self.rflux, self.zflux, self.oii) 

        # ---- Place holders for other variables used
        self.dNdm_model = [None] * 5

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

    def load_dNdm_models(self, save_dir=dir_derived):
        """
        Load best fit dNdm models.
        """
        for i in range(5):
            self.dNdm_model[i] = np.load(save_dir + ("broken-power-law-params-cn%d.npy" % i))
        return

    def var_reparam(self, gflux, rflux, zflux, oii = None):
        """
        Given the input variables return the model3 parametrization as noted above.
        """
        mu_g = flux2asinh_mag(gflux, band = "g")
        mu_r = flux2asinh_mag(rflux, band = "r")
        mu_z = flux2asinh_mag(zflux, band = "z")
        if oii is not None:
            mu_oii = flux2asinh_mag(oii, band = "oii")
            return mu_g - mu_z, mu_g - mu_r, mu_g - mu_oii, flux2mag(gflux)
        else:
            return mu_g - mu_z, mu_g - mu_r, flux2mag(gflux)

    def set_area_MC(self, val):
        self.area_MC = val
        return

    def plot_dNdm_all(self, show=True, savefig=False, save_dir="../figures/", fits=False, bw = 0.025, train=True):
        """
        Plot magnitude histogram of all classes (on the same plot).

        If train True, only plot Field 3 and 4.
        
        If fits True, then plot the Models.
        """
        if train:
            ibool = np.logical_or((self.field == 3), (self.field == 4))
            A = self.area_train
        else:
            ibool = np.ones(self.field.size, dtype=bool)
            A = np.sum(self.areas)


        mag_bins = np.arange(self.mag_min, self.mag_max+bw/2., bw)
        fig, ax = plt.subplots(1, figsize=(10, 7))
        for i in range(5):
            itmp = ibool & (self.cn==i)
            ax.hist(self.gmag[itmp], bins=mag_bins, histtype="step", \
                lw=1, label=cnames[i], weights=self.w[itmp]/A, color=colors[i])
            if fits: # Plot fitted models.
                assert self.dNdm_model[i] is not None
                bin_centers = (mag_bins[1:]+mag_bins[:-1])/2.
                ax.plot(bin_centers, bw * mag_broken_pow_law(self.dNdm_model[i], bin_centers), lw=1.5, c=colors[i], ls="--")
        ax.set_xlim([self.mag_min, 24.])
        ax.legend(loc="upper left", fontsize=20)
        if savefig: plt.savefig(save_dir+"Intersection-dNdm-by-class.png", dpi=400, bbox_inches="tight")
        if show: plt.show()
        plt.close()

        return


    def plot_colors(self, cn, show=True, savefig=False, save_dir="../figures/", fits=False, num_bins = 100., train=True):
        """
        Plot colors of each pair of variables being modeled.

        The user must specify the class number.

        If train True, only plot Field 3 and 4.
        
        If fits True, then plot the Models. If False, K set to zero.

        pari_num variable
        - 0: g-z vs. g-r
        - 1: g-z vs. g-oii
        - 2: g-r vs. g-oii
        """
        if train:
            ibool = np.logical_or((self.field == 3), (self.field == 4))
            A = self.area_train
        else:
            ibool = np.ones(self.field.size, dtype=bool)
            A = np.sum(self.areas)

        # If fit requested, count the number of components.
        if fits:
            K = 0 # FIX later.
        else:
            K = 0
            pair_num = 0

        # Bins to use for each color variables
        var_x_bins = np.arange(-2, 5.5+1e-3, 7.5/num_bins)
        var_y_bins = np.arange(-1, 2.5+1e-3, 3.5/num_bins)
        var_z_bins = np.arange(-1, 8+1e-3, 11/num_bins)

        #---- For all classes, plot g-z vs. g-r
        itmp = ibool & (self.cn==cn) # Sub select class of samples to plot.
        fig, ax_list = plt.subplots(2, 2, figsize=(16, 16))
        ax_list[1, 1].axis("off") # Turn off unused axis. 

        # (0,0) Scatter plot 
        ax_list[0, 0].scatter(self.var_x[itmp], self.var_y[itmp], s=5, marker="o", color="black", edgecolor="none")
        ax_list[0, 0].set_xlim([var_x_bins[0], var_x_bins[-1]])
        ax_list[0, 0].set_ylim([var_y_bins[0], var_y_bins[-1]])
        ax_list[0, 0].set_xlabel(r"$\mu_g - \mu_z$", fontsize=25)
        ax_list[0, 0].set_ylabel(r"$\mu_g - \mu_r$", fontsize=25)

        # (0,1) g-r histogram
        ax_list[0, 1].hist(self.var_y[itmp], bins=var_y_bins, color="black", histtype="step", \
            lw=1.5, orientation="horizontal", weights=self.w[itmp])
        ax_list[0, 1].set_ylim([var_y_bins[0], var_y_bins[-1]])
        ax_list[0, 1].set_ylabel(r"$\mu_g - \mu_r$", fontsize=25)

        # (1,0) g-z histogram
        ax_list[1, 0].hist(self.var_x[itmp], bins=var_x_bins, color="black", histtype="step", \
            lw=1.5, weights=self.w[itmp])
        ax_list[1, 0].set_xlim([var_x_bins[0], var_x_bins[-1]])
        ax_list[1, 0].set_xlabel(r"$\mu_g - \mu_z$", fontsize=25)

        # Fits
        # if fits: # Plot fitted models.
        #     assert self.dNdm_model[i] is not None
        #     bin_centers = (mag_bins[1:]+mag_bins[:-1])/2.
        #     ax.plot(bin_centers, bw * mag_broken_pow_law(self.dNdm_model[i], bin_centers), lw=1.5, c=colors[i], ls="--")

        if savefig: plt.savefig(save_dir+"Intersection-colors-cn%d-K%d-pair%d.png" % (cn, K, pair_num), dpi=400, bbox_inches="tight")
        if show: plt.show()
        plt.close()


        #---- For Gold and Silver classes only
        if (cn==0) or (cn==1):
            #---- plot g-z vs. g-oii            
            pair_num = 1            
            fig, ax_list = plt.subplots(2, 2, figsize=(16, 16))
            ax_list[1, 1].axis("off") # Turn off unused axis. 

            # (0,0) Scatter plot 
            ax_list[0, 0].scatter(self.var_x[itmp], self.var_z[itmp], s=5, marker="o", color="black", edgecolor="none")
            ax_list[0, 0].set_xlim([var_x_bins[0], var_x_bins[-1]])
            ax_list[0, 0].set_ylim([var_z_bins[0], var_z_bins[-1]])
            ax_list[0, 0].set_xlabel(r"$\mu_g - \mu_z$", fontsize=25)
            ax_list[0, 0].set_ylabel(r"$\mu_g - \mu_{OII}$", fontsize=25)

            # (0,1) g-oii histogram
            ax_list[0, 1].hist(self.var_z[itmp], bins=var_z_bins, color="black", histtype="step", \
                lw=1.5, orientation="horizontal", weights=self.w[itmp])
            ax_list[0, 1].set_ylim([var_z_bins[0], var_z_bins[-1]])
            ax_list[0, 1].set_ylabel(r"$\mu_g - \mu_{OII}$", fontsize=25)

            # (1,0) g-z histogram
            ax_list[1, 0].hist(self.var_x[itmp], bins=var_x_bins, color="black", histtype="step", \
                lw=1.5, weights=self.w[itmp])
            ax_list[1, 0].set_xlim([var_x_bins[0], var_x_bins[-1]])
            ax_list[1, 0].set_xlabel(r"$\mu_g - \mu_z$", fontsize=25)
            # Fits
            # if fits: # Plot fitted models.
            #     assert self.dNdm_model[i] is not None
            #     bin_centers = (mag_bins[1:]+mag_bins[:-1])/2.
            #     ax.plot(bin_centers, bw * mag_broken_pow_law(self.dNdm_model[i], bin_centers), lw=1.5, c=colors[i], ls="--")

            if savefig: plt.savefig(save_dir+"Intersection-colors-cn%d-K%d-pair%d.png" % (cn, K, pair_num), dpi=400, bbox_inches="tight")
            if show: plt.show()
            plt.close()


            #---- plot g-r vs. g-oii
            pair_num = 2
            fig, ax_list = plt.subplots(2, 2, figsize=(16, 16))
            ax_list[1, 1].axis("off") # Turn off unused axis. 

            # (0,0) Scatter plot 
            ax_list[0, 0].scatter(self.var_y[itmp], self.var_z[itmp], s=5, marker="o", color="black", edgecolor="none")
            ax_list[0, 0].set_xlim([var_y_bins[0], var_y_bins[-1]])
            ax_list[0, 0].set_ylim([var_z_bins[0], var_z_bins[-1]])
            ax_list[0, 0].set_xlabel(r"$\mu_g - \mu_r$", fontsize=25)
            ax_list[0, 0].set_ylabel(r"$\mu_g - \mu_{OII}$", fontsize=25)

            # (0,1) g-oii histogram
            ax_list[0, 1].hist(self.var_z[itmp], bins=var_z_bins, color="black", histtype="step", \
                lw=1.5, orientation="horizontal", weights=self.w[itmp])
            ax_list[0, 1].set_ylim([var_z_bins[0], var_z_bins[-1]])
            ax_list[0, 1].set_ylabel(r"$\mu_g - \mu_{OII}$", fontsize=25)

            # (1,0) g-z histogram
            ax_list[1, 0].hist(self.var_y[itmp], bins=var_y_bins, color="black", histtype="step", \
                lw=1.5, weights=self.w[itmp])
            ax_list[1, 0].set_xlim([var_y_bins[0], var_y_bins[-1]])
            ax_list[1, 0].set_xlabel(r"$\mu_g - \mu_r$", fontsize=25)
            # Fits
            # if fits: # Plot fitted models.
            #     assert self.dNdm_model[i] is not None
            #     bin_centers = (mag_bins[1:]+mag_bins[:-1])/2.
            #     ax.plot(bin_centers, bw * mag_broken_pow_law(self.dNdm_model[i], bin_centers), lw=1.5, c=colors[i], ls="--")

            if savefig: plt.savefig(save_dir+"Intersection-colors-cn%d-K%d-pair%d.png" % (cn, K, pair_num), dpi=400, bbox_inches="tight")
            if show: plt.show()
            plt.close()




        #---- Only for Gold and Silver, plot g-z vs. g-OII and g-r vs. g-OII

        return        

    def fit_dNdm_broken_pow(self, save_dir=dir_derived, Niter=5, bw=0.025):
        """
        This function is exclusively used for fitting the dNdm broken power law. 
        """
        for i in range(5): # Fit densities of classes 0 through 4.
            print "Fitting broken power law for %s" % cnames[i]
            ifit = self.iTrain & (self.cn==i)
            if i == 4:
                mag_max = 24.
                mag_min = 17.
            else:
                mag_max = 24.25                 
                mag_min = 22.

            params = dNdm_fit_broken_pow(self.gmag[ifit], self.w[ifit], bw, mag_min, mag_max, self.area_train, niter = Niter)
            np.save(save_dir+"broken-power-law-params-cn%d.npy" % i, params)
            print "\n"

        return None


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
