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

        # ---- Models
        self.dNdm_model = [None] * 5
        self.MoG_model = [None] * 5

        # Fraction of objects we believe to be desired DESI objects
        self.f_Gold = 1.
        self.f_Silver = 1.
        self.f_NoOII = 0.6
        self.f_NoZ = 0.25
        self.f_NonELG = 0.

        #---- MC parameters
        self.area_MC = 10

        # Mag Power law from which to generate importance samples.
        self.alpha_q = [20, 20, 20, 20, 9]

        # Grid parameters
        self.var_x_limits = [-.25, 3.5] # g-z
        self.var_y_limits = [-1, 1.4] # g-r
        self.gmag_limits = [19.5, 24.]
        self.num_bins = [375, 240, 450]

        # Cell_number in selection. Together with grid parameters this
        # is a representation of the selection region.
        self.cell_select = None

        # Desired nubmer of objects
        self.num_desired = 2400

        # External total density histogram (properly normalized with area)
        self.MD_hist_N_cal_flat = None

        # Utility metric options
        self.U_Gold = 1
        self.U_Silver = 1
        self.U_NoOII = 0.6
        self.U_NoZ = 0.25
        self.U_NonELG = 0

        # Normal noise model parameters
        self.glim_err = 23.8
        self.rlim_err = 23.4
        self.zlim_err = 22.4
        self.oii_lim_err = 8 # 7 sigma

        # sigma factor for the proposal        
        self.sigma_proposal = 1.5         

        # After noise addition, we make a cut at 24 since we do not use those.
        self.fcut = mag2flux(24.) 


        #---- MC variable place holder
        # Original sample.
        # 0: NonELG, 1: NoZ, 2: ELG
        self.NSAMPLE = [None] * 5
        self.gflux0 = [None] * 5 # 0 for original
        self.rflux0 = [None] * 5 # 0 for original
        self.zflux0 = [None] * 5 # 0 for original
        self.oii0 = [None] * 5 # Although only ELG class has oii and redz, for consistency, we have three elements lists.
        self.redz0 = [None] * 5
    
        # Noise seed. err_seed ~ N(0, 1). This can be transformed by scaling appropriately.
        self.g_err_seed = [None] * 5 # Error seed.
        self.r_err_seed = [None] * 5 # Error seed.
        self.z_err_seed = [None] * 5 # Error seed.
        self.oii_err_seed = [None] * 5 # Error seed.
        
        # Noise convolved values
        self.gflux_obs = [None] * 5 # obs for observed
        self.rflux_obs = [None] * 5 # obs for observed
        self.zflux_obs = [None] * 5 # obs for observed
        self.oii_obs = [None] * 5 # Although only ELG class has oii and redz, for consistency, we have three elements lists.
        
        # Importance weight: Used when importance sampling is asked for
        self.iw = [None] * 5
        self.iw0 = [None] * 5 # 0 denotes intrinsic sample

        # Observed final distributions
        self.var_x_obs = [None] * 5 # g-z
        self.var_y_obs = [None] * 5 # g-r
        self.var_z_obs = [None] * 5 # g-oii
        self.redz_obs = [None] * 5        
        self.gmag_obs = [None] * 5        

        # Observed utility
        self.utility_obs = [None] * 5

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

    def set_err_lims(self, glim, rlim, zlim, oii_lim):
        """
        Set the error characteristics.
        """
        self.glim_err = glim
        self.rlim_err = rlim 
        self.zlim_err = zlim
        self.oii_lim_err = oii_lim

        return        

    def set_num_desired(self, Ntot):
        self.num_desired = Ntot
        return None

    def load_calibration_data(self, option=0):
        """
        Load calibration data
        - option=0: DR5
        - option=1: DR4 

        Both catalogs have g < 24 cut.
        """
        print "Loading calibration data"
        start = time.time()

        if option==0:
            g, r, z, _, _, A = load_DR5_calibration()
        elif option==1:
            assert False
            g, r, z, _, _, A = load_DR4_calibration()            
        else:
            assert False

        # Asinh magnitude
        gmag = flux2mag(g)
        asinh_g = flux2asinh_mag(g, band="g")
        asinh_r = flux2asinh_mag(r, band="r")        
        asinh_z = flux2asinh_mag(z, band="z")

        # Variable changes
        varx = asinh_g - asinh_z
        vary = asinh_g - asinh_r

        # Samples
        samples = np.array([varx, vary, gmag]).T
        hist, _ = np.histogramdd(samples, bins=self.num_bins, range=[self.var_x_limits, self.var_y_limits, self.gmag_limits]) # A is for normalization.
        self.MD_hist_N_cal_flat = hist.flatten() / float(A)

        print "Time taken: %.1f seconds" % (time.time()-start)

        return

    def gen_sample_intrinsic_mag(self):
        """
        Given load models (dNdm and MoG), draw intrinsic samples proportional to MC area.

        Importance sampling is always used.
        """
        for i in range(5): # For each class
            #---- Import model parameters
            # MoG model
            amps, means, covs = self.MoG_model[i]["amps"], self.MoG_model[i]["means"], self.MoG_model[i]["covs"]

            #---- Compute the number of sample to draw.
            NSAMPLE = int(integrate_mag_broken_pow_law(self.dNdm_model[i], self.mag_min, self.mag_max, area = self.area_MC))
            print "%s sample number: %d" % (cnames[i], NSAMPLE)

            #---- Generate magnitudes
            gmag = gen_mag_pow_law_samples([1, self.alpha_q[i]], self.mag_min, self.mag_max, NSAMPLE)
            r_tilde = mag_broken_pow_law(self.dNdm_model[i], gmag)/mag_pow_law([1, self.alpha_q[i]], gmag)
            self.iw0[i] = (r_tilde/r_tilde.sum()) 
            gflux = mag2flux(gmag)

            #---- Generate Nsample from MoG.
            MoG_sample, iw = sample_MoG(amps, means, covs, NSAMPLE, importance_sampling=True, \
                factor_importance = self.sigma_proposal)
            self.iw0[i] *= iw            

            #---- Compute variables of interest for all categories
            mu_g = flux2asinh_mag(gflux, band = "g")
            mu_gz, mu_gr = MoG_sample[:,0], MoG_sample[:,1]
            mu_z = mu_g - mu_gz
            mu_r = mu_g - mu_gr
            zflux = asinh_mag2flux(mu_z, band = "z")
            rflux = asinh_mag2flux(mu_r, band = "r")
            
            # Save grz-fluxes
            self.gflux0[i] = gflux
            self.rflux0[i] = rflux
            self.zflux0[i] = zflux
            self.NSAMPLE[i] = NSAMPLE

            # Gen err seed and save
            # Also, collect unormalized importance weight factors, multiply and normalize.
            self.g_err_seed[i], iw = gen_err_seed(self.NSAMPLE[i], sigma=self.sigma_proposal, return_iw_factor=True)
            # print "g_err_seed importance weights. First 10", iw[]
            self.iw0[i] *= iw
            self.r_err_seed[i], iw = gen_err_seed(self.NSAMPLE[i], sigma=self.sigma_proposal, return_iw_factor=True)
            self.iw0[i] *= iw        
            self.z_err_seed[i], iw = gen_err_seed(self.NSAMPLE[i], sigma=self.sigma_proposal, return_iw_factor=True)
            self.iw0[i] *= iw
            if (i==0) or (i==1): #ELG 
                mu_goii = MoG_sample[:, 2]
                mu_oii = mu_g - mu_goii
                self.oii0[i] = asinh_mag2flux(mu_oii, band = "oii")

                # oii error seed
                self.oii_err_seed[i], iw = gen_err_seed(self.NSAMPLE[i], sigma=self.sigma_proposal, return_iw_factor=True)
                self.iw0[i] *= iw
                # Saving
            self.iw0[i] = (self.iw0[i]/self.iw0[i].sum()) * self.NSAMPLE[i] # Normalization and multiply by the number of samples generated.

        return                

    def load_dNdm_models(self, save_dir=dir_derived):
        """
        Load best fit dNdm models.
        """
        for i in range(5):
            self.dNdm_model[i] = np.load(save_dir + ("broken-power-law-params-cn%d.npy" % i))
        return

    def load_MoG_models(self, K_list=None, save_dir=dir_derived):
        """
        K list specified the number of components to be imported.

        If None, author specified (best fit) number is used.
        """
        if K_list is None:
            K_list = [2, 2, 2, 2, 6] # Based on visual inspection
        for i in range(5):
            self.MoG_model[i] = np.load(save_dir + ("MoG-params-cn%d-K%d.npy" % (i, K_list[i]))).item()

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
            return mu_g - mu_z, mu_g - mu_r, None, flux2mag(gflux)

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


    def plot_colors(self, cn, K=0, show=True, savefig=False, save_dir="../figures/",\
     num_bins = 100., train=True, plot_ext=False, gflux=None, rflux=None, zflux=None, \
     oii=None, iw=None, A_ext=None, gmag_cut=24.):
        """
        Plot colors of each pair of variables being modeled.

        The user must specify the class number.

        If train True, only plot Field 3 and 4.

        If plot_ext, then plot external objects given by gflux, rflux, zflux, and oii.
        K is used as an additional user provided tag number. iw is the associated importance
        number.
        
        Trainig data is plotted only up to gmag_cut

        pari_num variable
        - 0: g-z vs. g-r
        - 1: g-z vs. g-oii
        - 2: g-r vs. g-oii
        """
        if train:
            ibool = np.logical_or((self.field == 3), (self.field == 4)) & (self.gmag < gmag_cut)
            A = self.area_train
        else:
            ibool = self.gmag < gmag_cut
            A = np.sum(self.areas)

        # If ext requested, count the number of components.
        if plot_ext:
            # Reparameterize the input external data sets and insist on A_ext.
            var_x_ext, var_y_ext, var_z_ext, _ = self.var_reparam(gflux, rflux, zflux, oii)
            assert A_ext is not None
            if iw is None:
                iw = np.ones(var_x_ext.size, dtype=float)

        # Bins to use for each color variables
        var_x_bins = np.arange(-2, 5.5+1e-3, 7.5/num_bins)
        var_y_bins = np.arange(-1, 2.5+1e-3, 3.5/num_bins)
        var_z_bins = np.arange(-1, 8+1e-3, 11/num_bins)

        #---- For all classes, plot g-z vs. g-r
        pair_num = 0        
        itmp = ibool & (self.cn==cn) # Sub select class of samples to plot.
        fig, ax_list = plt.subplots(2, 2, figsize=(16, 16))
        ax_list[1, 1].axis("off") # Turn off unused axis. 

        if plot_ext:
            ax_list[0, 0].scatter(var_x_ext, var_y_ext, s=5, marker="o", color="red", edgecolor="none")
            ax_list[0, 1].hist(var_y_ext, bins=var_y_bins, color="red", histtype="step", \
                lw=1.5, orientation="horizontal", weights=iw/A_ext)
            ax_list[1, 0].hist(var_x_ext, bins=var_x_bins, color="red", histtype="step", \
                lw=1.5, weights=iw/A_ext)            

        # (0,0) Scatter plot 
        ax_list[0, 0].scatter(self.var_x[itmp], self.var_y[itmp], s=5, marker="o", color="black", edgecolor="none")
        ax_list[0, 0].set_xlim([var_x_bins[0], var_x_bins[-1]])
        ax_list[0, 0].set_ylim([var_y_bins[0], var_y_bins[-1]])
        ax_list[0, 0].set_xlabel(r"$\mu_g - \mu_z$", fontsize=25)
        ax_list[0, 0].set_ylabel(r"$\mu_g - \mu_r$", fontsize=25)

        # (0,1) g-r histogram
        ax_list[0, 1].hist(self.var_y[itmp], bins=var_y_bins, color="black", histtype="step", \
            lw=1.5, orientation="horizontal", weights=self.w[itmp]/A)
        ax_list[0, 1].set_ylim([var_y_bins[0], var_y_bins[-1]])
        ax_list[0, 1].set_ylabel(r"$\mu_g - \mu_r$", fontsize=25)

        # (1,0) g-z histogram
        ax_list[1, 0].hist(self.var_x[itmp], bins=var_x_bins, color="black", histtype="step", \
            lw=1.5, weights=self.w[itmp]/A)
        ax_list[1, 0].set_xlim([var_x_bins[0], var_x_bins[-1]])
        ax_list[1, 0].set_xlabel(r"$\mu_g - \mu_z$", fontsize=25)

        if savefig: plt.savefig(save_dir+"Intersection-colors-cn%d-K%d-pair%d.png" % (cn, K, pair_num), dpi=400, bbox_inches="tight")
        if show: plt.show()
        plt.close()


        #---- For Gold and Silver classes only
        if (cn==0) or (cn==1):
            #---- plot g-z vs. g-oii            
            pair_num = 1            
            fig, ax_list = plt.subplots(2, 2, figsize=(16, 16))
            ax_list[1, 1].axis("off") # Turn off unused axis. 

            if plot_ext:
                ax_list[0, 0].scatter(var_x_ext, var_z_ext, s=5, marker="o", color="red", edgecolor="none")
                ax_list[0, 1].hist(var_z_ext, bins=var_z_bins, color="red", histtype="step", \
                    lw=1.5, orientation="horizontal", weights=iw/A_ext)
                ax_list[1, 0].hist(var_x_ext, bins=var_x_bins, color="red", histtype="step", \
                    lw=1.5, weights=iw/A_ext)            

            # (0,0) Scatter plot 
            ax_list[0, 0].scatter(self.var_x[itmp], self.var_z[itmp], s=5, marker="o", color="black", edgecolor="none")
            ax_list[0, 0].set_xlim([var_x_bins[0], var_x_bins[-1]])
            ax_list[0, 0].set_ylim([var_z_bins[0], var_z_bins[-1]])
            ax_list[0, 0].set_xlabel(r"$\mu_g - \mu_z$", fontsize=25)
            ax_list[0, 0].set_ylabel(r"$\mu_g - \mu_{OII}$", fontsize=25)

            # (0,1) g-oii histogram
            ax_list[0, 1].hist(self.var_z[itmp], bins=var_z_bins, color="black", histtype="step", \
                lw=1.5, orientation="horizontal", weights=self.w[itmp]/A)
            ax_list[0, 1].set_ylim([var_z_bins[0], var_z_bins[-1]])
            ax_list[0, 1].set_ylabel(r"$\mu_g - \mu_{OII}$", fontsize=25)

            # (1,0) g-z histogram
            ax_list[1, 0].hist(self.var_x[itmp], bins=var_x_bins, color="black", histtype="step", \
                lw=1.5, weights=self.w[itmp]/A)
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

            if plot_ext:
                ax_list[0, 0].scatter(var_y_ext, var_z_ext, s=5, marker="o", color="red", edgecolor="none")
                ax_list[0, 1].hist(var_z_ext, bins=var_z_bins, color="red", histtype="step", \
                    lw=1.5, orientation="horizontal", weights=iw/A_ext)
                ax_list[1, 0].hist(var_y_ext, bins=var_y_bins, color="red", histtype="step", \
                    lw=1.5, weights=iw/A_ext)            

            # (0,0) Scatter plot 
            ax_list[0, 0].scatter(self.var_y[itmp], self.var_z[itmp], s=5, marker="o", color="black", edgecolor="none")
            ax_list[0, 0].set_xlim([var_y_bins[0], var_y_bins[-1]])
            ax_list[0, 0].set_ylim([var_z_bins[0], var_z_bins[-1]])
            ax_list[0, 0].set_xlabel(r"$\mu_g - \mu_r$", fontsize=25)
            ax_list[0, 0].set_ylabel(r"$\mu_g - \mu_{OII}$", fontsize=25)

            # (0,1) g-oii histogram
            ax_list[0, 1].hist(self.var_z[itmp], bins=var_z_bins, color="black", histtype="step", \
                lw=1.5, orientation="horizontal", weights=self.w[itmp]/A)
            ax_list[0, 1].set_ylim([var_z_bins[0], var_z_bins[-1]])
            ax_list[0, 1].set_ylabel(r"$\mu_g - \mu_{OII}$", fontsize=25)

            # (1,0) g-z histogram
            ax_list[1, 0].hist(self.var_y[itmp], bins=var_y_bins, color="black", histtype="step", \
                lw=1.5, weights=self.w[itmp]/A)
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


    def fit_MoG(self, cn, K, Niter=5, save_dir=dir_derived):
        """
        Fit MoGs to data. Only used for fitting and nothing else. 

        cn is the class to fit and K is the number of components to fit.
        """
        ifit = (self.cn==cn) & self.iTrain        
        if (cn > 1): # Other than Gold or Silver
            ND = 2 # Dimension of model
            Ydata = np.array([self.var_x[ifit], self.var_y[ifit]]).T
        else: # If Gold or Silver
            ND = 3 # Dimension of model
            Ydata = np.array([self.var_x[ifit], self.var_y[ifit], self.var_z[ifit]]).T
        Ycovar = self.gen_covar(ifit, ND=ND)
        weight = self.w[ifit]
        params = fit_GMM(Ydata, Ycovar, ND, K=K, Niter=Niter, weight=weight)
        np.save(save_dir+"MoG-params-cn%d-K%d.npy" % (cn, K), params)

        return

    def gen_covar(self, ifit, ND=2):
        """
        Covariance matrix corresponding to the new parametrization.

        Original parameterization: zf, rf, oii, gf
        New parameterization is given by the model3.
        """
        Nsample = np.sum(ifit)
        Covar = np.zeros((Nsample, ND, ND))

        zflux, rflux, gflux, oii = self.zflux[ifit], self.rflux[ifit], self.gflux[ifit], self.oii[ifit]
        var_err_list = [self.zf_err[ifit], self.rf_err[ifit], self.oii_err[ifit], self.gf_err[ifit]]

        # constant converion factor.
        const = 0.542868

        # Softening factors for asinh mags
        b_g = 1.042 * 0.0285114
        b_r = 1.042 * 0.0423106
        b_z = 1.042 * 0.122092
        b_oii = 1.042 * 0.581528277909

        for i in range(Nsample):
            if ND == 2:
                # Construct the original space covariance matrix in 3 x 3 subspace.
                tmp = []
                for j in [0, 1, 3]:
                    tmp.append(var_err_list[j][i]**2) # var = err^2
                Cx = np.diag(tmp)

                g, r, z = gflux[i], rflux[i], zflux[i]
                M00, M01, M02 = const/np.sqrt(b_z**2+z**2/4.), 0, -const/(g*np.sqrt(b_g**2+g**2/4.))
                M10, M11, M12 = 0, const/np.sqrt(b_r**2+r**2/4.), -const/(g*np.sqrt(b_g**2+g**2/4.))
                M = np.array([[M00, M01, M02],
                            [M10, M11, M12]])
                
                Covar[i] = np.dot(np.dot(M, Cx), M.T)
            elif ND == 3:
                # Construct the original space covariance matrix in 5 x 5 subspace.
                tmp = []
                for j in range(4):
                    tmp.append(var_err_list[j][i]**2) # var = err^2
                Cx = np.diag(tmp)

                # Construct the affine transformation matrix.
                g, r, z, o = gflux[i], rflux[i], zflux[i], oii[i]
                M00, M01, M02, M03 = const/np.sqrt(b_z**2+z**2/4.), 0, 0, -const/(g*np.sqrt(b_g**2+g**2/4.))
                M10, M11, M12, M13 = 0, const/np.sqrt(b_r**2+r**2/4.), 0, -const/(g*np.sqrt(b_g**2+g**2/4.))
                M20, M21, M22, M23 = 0, 0, const/np.sqrt(b_oii**2+o**2/4.), -const/(g*np.sqrt(b_g**2+g**2/4.))
                
                M = np.array([[M00, M01, M02, M03],
                                [M10, M11, M12, M13],
                                [M20, M21, M22, M23]])
                Covar[i] = np.dot(np.dot(M, Cx), M.T)
            else: 
                print "The input number of variables need to be either 2 or 3."
                assert False

        return Covar

    def gen_err_conv_sample(self):
        """
        Given the error properties glim_err, rlim_err, zlim_err, oii_lim_err, 
        add noise to the intrinsic density sample and compute the parametrization.
        """
        for i in range(5):
            print "Noise convolution for %s" % cnames[i]
            self.gflux_obs[i] = self.gflux0[i] + self.g_err_seed[i] * mag2flux(self.glim_err)/5.
            self.rflux_obs[i] = self.rflux0[i] + self.r_err_seed[i] * mag2flux(self.rlim_err)/5.
            self.zflux_obs[i] = self.zflux0[i] + self.z_err_seed[i] * mag2flux(self.zlim_err)/5.

            # Make flux cut
            ifcut = self.gflux_obs[i] > self.fcut
            self.gflux_obs[i] = self.gflux_obs[i][ifcut]
            self.rflux_obs[i] = self.rflux_obs[i][ifcut]
            self.zflux_obs[i] = self.zflux_obs[i][ifcut]

            # Compute model parametrization
            mu_g = flux2asinh_mag(self.gflux_obs[i], band="g")
            mu_r = flux2asinh_mag(self.rflux_obs[i], band="r")
            mu_z = flux2asinh_mag(self.zflux_obs[i], band="z")
            self.var_x_obs[i] = mu_g - mu_z
            self.var_y_obs[i] = mu_g - mu_r
            self.gmag_obs[i] = flux2mag(self.gflux_obs[i])

            # Updating the importance weight with the cut
            self.iw[i] = self.iw0[i][ifcut]

            # Number of samples after the cut.
            Nsample = self.gmag_obs[i].size

            # More parametrization to compute for ELGs. Also, compute FoM.
            if (i==0) or (i==1):
                # oii paramerization
                self.oii_obs[i] = self.oii0[i] + self.oii_err_seed[i] * (self.oii_lim_err/7.) # 
                self.oii_obs[i] = self.oii_obs[i][ifcut]
                mu_oii = flux2asinh_mag(self.oii_obs[i], band="oii")
                self.var_z_obs[i] = mu_g - mu_oii

            # Gen utility_obs
            self.utility_obs[i] = self.gen_utility(i, Nsample, self.oii_obs[i])

        return


    def gen_utility(self, cat, Nsample, oii=None):
        """
        Given the user specified utility metric of the form in the NDM paper,
        provide the utility numbers.
        """
        if cat == 0: # If Gold
            util = np.ones(Nsample, dtype=float) * self.U_Gold
            util[oii<8] = 0
        if cat == 1: # If Silver
            util = np.ones(Nsample, dtype=float) * self.U_Silver
            util[oii<8] = 0            
        if cat == 2: # If NoOII
            util = np.ones(Nsample, dtype=float) * self.U_NoOII
        if cat == 3: # If Gold
            util = np.ones(Nsample, dtype=float) * self.U_NoZ
        if cat == 4: # If Gold
            util = np.ones(Nsample, dtype=float) * self.U_NonELG

        return util                    


    def gen_selection_volume_ext_cal(self, num_batches=1, batch_size=1000, gaussian_smoothing=True, sig_smoothing_window=[5, 5, 5], \
        dNdm_mag_reg=True, fake_density_fraction = 0.03, marginal_eff=True, \
        Ndesired_arr=np.arange(0, 3500, 10)):
        """
        Given the generated sample (intrinsic val + noise), generate a selection volume 
        following the procedure outlined in the paper. Note that external dataset is used for
        number density calibration purpose.

        There are two regularization scheme: 
        - Gaussian smoothing: If gaussian_smoothing is True, then gaussian convolution with pixel
        window of size smoothing_window for each dimension is performed.
        - Magnitude dependent density of fake objects: If dNdm_mag_reg is True, then magnitude dependent
        density of fake objects are added.

        If marginal_eff is True, then compute marginal efficiency. As the selection region grows
        compute selection efficiency in each bin in Ndesired_arr.

        num_batches * batch_size = total MC area.
    
        General strategy
        - Batch generate samples and construct histograms
        - Smooth the MC sample histograms.
        - Add in the regularization. 
        - Compute utility and sort.
        - Sort other histograms from which other relevant information can be gained.
        - Compute total as well as marginal utility.

        Quantities of interest:
        - Total and marginal utility.
        - Predicted contribution from Gold (OII>8) and Silver (OII>8), seperately.
        - Predicted contribution from NoZ and NoOII, seperately.
        """
        print "/---- Selection volume generation starts here."
        print "Set global area_MC to batch_size = %d" % batch_size
        self.area_MC = batch_size

        #---- Calculate number of batches to work on.
        print "Number of batches to process: %d" % num_batches

        #---- Placeholder for the histograms
        MD_hist_Nj_good = [None] * 5 # Histogram of objects in class j that are desired for DESI
        MD_hist_N_util_total = None
        MD_hist_N_total = None 

        #---- Generate samples, convolve error, construct histogram, tally, and repeat."
        start = time.time()
        for batch in range(1, num_batches+1):
            print "/---- Batch %d" % batch
            print "Generate intrinic samples."
            self.gen_sample_intrinsic_mag()

            print "Add noise to the generated samples."
            self.gen_err_conv_sample() # Perform error convolution

            print "Consturct histogram and tally"
            f_arr = [self.f_Gold, self.f_Silver, self.f_NoOII, self.f_NoZ, self.f_NonELG]
            for i in range(5):
                print "%s" % cnames[i]
                samples = np.array([self.var_x_obs[i], self.var_y_obs[i], self.gmag_obs[i]]).T

                Nj, _ = np.histogramdd(samples, bins=self.num_bins, \
                    range=[self.var_x_limits, self.var_y_limits, self.gmag_limits], weights=self.iw[i])

                Nj_util, _ = np.histogramdd(samples, bins=self.num_bins, \
                    range=[self.var_x_limits, self.var_y_limits, self.gmag_limits], weights=self.utility_obs[i]*self.iw[i])

                # Special weights for number of desired objects calculation
                weights = np.copy(self.iw[i]) * f_arr[i]
                if (i < 2): # Gold and Silver get special treatment because of OII.
                    weights[self.oii_obs[i] < 8] = 0
                Nj_good, _ = np.histogramdd(samples, bins=self.num_bins, \
                    range=[self.var_x_limits, self.var_y_limits, self.gmag_limits], weights=weights)

                if gaussian_smoothing: # Applying Gaussian filtering
                    sigma_smoothing_limit=5                
                    gaussian_filter(Nj, sig_smoothing_window, order=0, output=Nj, mode='constant', cval=0.0, truncate=sigma_smoothing_limit)
                    gaussian_filter(Nj_util, sig_smoothing_window, order=0, output=Nj_util, mode='constant', cval=0.0, truncate=sigma_smoothing_limit)
                    gaussian_filter(Nj_good, sig_smoothing_window, order=0, output=Nj_good, mode='constant', cval=0.0, truncate=sigma_smoothing_limit)

                if (batch == 1) & (i==0):
                    MD_hist_N_total = Nj
                    MD_hist_N_util_total = Nj_util
                    MD_hist_Nj_good[i] = Nj_good
                elif (batch == 1) & (i>0):
                    MD_hist_N_total += Nj
                    MD_hist_N_util_total += Nj_util
                    MD_hist_Nj_good[i] = Nj_good
                else:
                    MD_hist_N_total += Nj
                    MD_hist_N_util_total += Nj_util
                    MD_hist_Nj_good[i] += Nj_good                    

        print "Time taken: %.2f seconds\n" % (time.time() - start)

        if dNdm_mag_reg:
            # For each magnitude bin, sum up the number of objects and evenly disperse throughout the grid
            # in that magnitude
            print "Computing magnitude dependent regularization.\n"
            start = time.time()
            num_bins_xy = MD_hist_N_total.shape[0] * MD_hist_N_total.shape[1] # Number of bins in xy subspace
            for k in range(MD_hist_N_total.shape[2]):
                MD_hist_N_total[:, :, k] += np.sum(MD_hist_N_total[:, :, k]) * fake_density_fraction / float(num_bins_xy)

            # #--- dNdm - broken pow law version. ** Save for future reference **
            # for e in self.MODELS_mag_pow: 
            #     m_min, m_max = self.gmag_limits[0], self.gmag_limits[1]
            #     m_Nbins = self.num_bins[2]
            #     m = np.linspace(m_min, m_max, m_Nbins, endpoint=False)
            #     dm = (m_max-m_min)/m_Nbins
            #     for i, m_tmp in enumerate(m):
            #         MD_hist_N_regular[:, :, i] += self.frac_regular * integrate_mag_broken_pow_law(e, m_tmp, m_tmp+dm, area=self.area_MC) / np.multiply.reduce((self.num_bins[:2]))

        print "Computing utility and sorting."
        start = time.time()        
        # Compute utility
        utility = MD_hist_N_util_total/MD_hist_N_total

        # Flatten utility array
        utility_flat = utility.flatten()

        # Order cells according to utility
        # This corresponds to cell number of descending order sorted array.
        idx_sort = (-utility_flat).argsort()
        print "Time taken: %.2f seconds" % (time.time() - start)        

        print "Flatten and sort the MD histograms including the calibration"
        start = time.time()        
        # Flatten the histograms
        MD_hist_N_total = MD_hist_N_total.flatten()[idx_sort]
        for i in range(5):
            MD_hist_Nj_good[i] = MD_hist_Nj_good[i].flatten()[idx_sort]
        MD_hist_N_cal_flat = self.MD_hist_N_cal_flat[idx_sort]
        print "Time taken: %.2f seconds" % (time.time() - start)
                                   

        #---- Selection generation.
        # 1) Compute the total efficiency given self.num_desired
        Ntotal = 0
        counter = 0
        for ncell in MD_hist_N_cal_flat:
            if Ntotal > self.num_desired: 
                break
            Ntotal += ncell
            counter += 1
        print np.sum(MD_hist_N_cal_flat[:counter])

        # Save the selection to be used later.
        self.cell_select = np.sort(idx_sort[:counter])            

        # Report the overall efficiency.
        print "\nStats on sample with N_tot = %d" % self.num_desired
        # Note that the quantities below are un-normalized.
        Ntotal_pred = np.sum(MD_hist_N_total[:counter])
        Ngood_pred = 0
        print "Class: (Expected number in desired sample)"
        for i in range(5):
            tmp = np.sum(MD_hist_Nj_good[i][:counter])
            Ngood_pred += tmp
            print "%s: %.1f%% (%d)" % (cnames[i], tmp/Ntotal_pred * 100, self.num_desired*tmp/Ntotal_pred)
        eff_pred = Ngood_pred/Ntotal_pred
        print "Eff of the sample: %.3f\n" % eff_pred

        # 2) Compute the marginal efficiency as a function of bins in Ndesired_arr
        # For each bin, compute the efficiency of the bin and its center.
        # 0-4: f_j_bin: Fraction of desired objects in class j (nj_pred/ntot_pred)
        # 5: eff_bin
        bin_centers = (Ndesired_arr[1:] + Ndesired_arr[:-1])/2.
        summary_arr = np.zeros((bin_centers.size, 6))

        start_idx = 0
        end_idx = 0
        Ntotal = 0        
        for i, n in enumerate(Ndesired_arr[1:]):
            if (i % 50) == 0:
                print "Working on bin i = %d out of %d" % (i, bin_centers.size)
            for ncell in MD_hist_N_cal_flat:
                if Ntotal > n: 
                    break            
                Ntotal += ncell
                end_idx += 1

            # Computing cell efficiency of each objects class.
            tmp = np.zeros(6, dtype=float) # place holder for i th array
            for j in range(5):
                tmp[j] = np.sum(MD_hist_Nj_good[j][start_idx:end_idx])
            tmp[-1] = np.sum(tmp[:-1])
            tmp /= np.sum(MD_hist_N_total[start_idx:end_idx])
            summary_arr[i, :] = tmp

            start_idx = end_idx
        print "\n\n"

        return bin_centers, summary_arr


    def validate_on_DEEP2(self):
        """
        Apply the generated selection to each DEEP2 Field data. 
        """
        if self.cell_select is None:
            print "Selection volume must be generated."
            assert False

        for fnum in range(2, 5):
            # Selecting only objects in the field.
            ifield = (self.field == fnum)
            area_sample = self.areas[fnum-2]
            gflux = self.gflux[ifield] 
            rflux = self.rflux[ifield]
            zflux = self.zflux[ifield]
            var_x = self.var_x[ifield]
            var_y = self.var_y[ifield]
            gmag = self.gmag[ifield]
            oii = self.oii[ifield]
            redz = self.red_z[ifield]
            w = self.w[ifield]
            cn = self.cn[ifield]

            # Apply the selection.
            iselected = self.apply_selection(gflux, rflux, zflux)

            f_arr = [self.f_Gold, self.f_Silver, self.f_NoOII, self.f_NoZ, self.f_NonELG]
            # Report the number of objects in each catagory that are desired by DESI.
            print "\---- DEEP2 Field %d" % fnum
            Ntot_selected = np.sum(w[iselected])/area_sample # Properly normalized total            
            print "Total selected: %d" % Ntot_selected
            print "Class: Fraction of objects desired by DESI (Density)"
            Ngood_selected = 0
            for i in range(4):
                if i < 2: 
                    ibool = (cn == i) & (oii > 8) & iselected
                else:
                    ibool = (cn == i) & iselected
                tmp = f_arr[i]* np.sum(w[ibool])/area_sample
                Ngood_selected += tmp
                print "%s: %.1f%% (%d)" % (cnames[i], tmp/Ntot_selected * 100, tmp)
            print "Total good: %.1f%% (%d)" % (Ngood_selected/Ntot_selected * 100, Ngood_selected)
            print "\n\n"
        return None



    def apply_selection(self, gflux, rflux, zflux):
        """
        Model 3
        Given gflux, rflux, zflux of samples, return a boolean vector that gives the selection.
        """
        mu_g = flux2asinh_mag(gflux, band = "g")
        mu_r = flux2asinh_mag(rflux, band = "r")
        mu_z = flux2asinh_mag(zflux, band = "z")

        var_x = mu_g - mu_z
        var_y = mu_g - mu_r
        gmag = flux2mag(gflux)

        samples = [var_x, var_y, gmag]

        # Generate cell number 
        cell_number = multdim_grid_cell_number(samples, 3, [self.var_x_limits, self.var_y_limits, self.gmag_limits], self.num_bins)

        # Sort the cell number
        idx_sort = cell_number.argsort()
        cell_number = cell_number[idx_sort]

        # Placeholder for selection vector
        iselect = check_in_arr2(cell_number, self.cell_select)

        # The last step is necessary in order for iselect to have the same order as the input sample variables.
        idx_undo_sort = idx_sort.argsort()        
        return iselect[idx_undo_sort]

    def cell_select_centers(self):
        """
        Return selected cells centers given cell numbers.
        """
        limits = [self.var_x_limits, self.var_y_limits, self.gmag_limits]
        Ncell_select = self.cell_select.size # Number of cells in the selection
        centers = [None, None, None]

        for i in range(3):
            Xmin, Xmax = limits[i]
            bin_edges, dX = np.linspace(Xmin, Xmax, self.num_bins[i]+1, endpoint=True, retstep=True)
            bin_centers = (bin_edges[1:]+bin_edges[:-1])/2.
            idx = (self.cell_select % np.multiply.reduce(self.num_bins[i:])) //  np.multiply.reduce(self.num_bins[i+1:])
            centers[i] = bin_centers[idx.astype(int)]

        return np.asarray(centers).T

    def gen_select_boundary_slices(self, slice_dir = 2, save_dir="../figures/", \
        prefix = "test", output_sparse=True, increment=10, centers=None, plot_ext=False,\
        gflux_ext=None, rflux_ext=None, zflux_ext=None, ibool_ext = None,\
        var_x_ext=None, var_y_ext=None, gmag_ext=None, use_parameterized_ext=False,\
        pt_size=10, pt_size_ext=10, alpha_ext=0.5, guide=False):
        """
        Given slice direction, generate slices of boundary

        0: var_x
        1: var_y
        2: gmag

        If plot_ext True, then plot user supplied external objects.

        If centers is not None, then use it instead of generating one.

        If use_parameterized_ext, then the user may provide already parameterized version of the external data points.

        If guide True, then plot the guide line.

        If output_sparse=True, then only 10% of the boundaries are plotted and saved.
        """

        slice_var_tag = ["mu_gz", "mu_gr", "gmag"]
        var_names = ["$\mu_g - \mu_z$", "$\mu_g - \mu_r$", "$g$"]

        if centers is None:
            centers = self.cell_select_centers()

        if guide:
            x_guide, y_guide = gen_guide_line()

        if plot_ext:
            if use_parameterized_ext:
                if ibool_ext is not None:
                    var_x_ext = var_x_ext[ibool_ext]
                    var_y_ext = var_y_ext[ibool_ext]
                    gmag_ext = gmag_ext[ibool_ext]                
            else:     
                if ibool_ext is not None:
                    gflux_ext = gflux_ext[ibool_ext]
                    rflux_ext = rflux_ext[ibool_ext]
                    zflux_ext = zflux_ext[ibool_ext]

                mu_g, mu_r, mu_z = flux2asinh_mag(gflux_ext, band="g"), flux2asinh_mag(rflux_ext, band="r"), flux2asinh_mag(zflux_ext, band="z")
                var_x_ext = mu_g-mu_z
                var_y_ext = mu_g-mu_r
                gmag_ext = flux2mag(gflux_ext)

            variables = [var_x_ext, var_y_ext, gmag_ext]

        limits = [self.var_x_limits, self.var_y_limits, self.gmag_limits]        
        Xmin, Xmax = limits[slice_dir]
        bin_edges, dX = np.linspace(Xmin, Xmax, self.num_bins[slice_dir]+1, endpoint=True, retstep=True)

        print slice_var_tag[slice_dir]
        if output_sparse:
            iterator = range(0, self.num_bins[slice_dir], increment)
        else:
            iterator = range(self.num_bins[slice_dir])

        for i in iterator: 
            ibool = (centers[:, slice_dir] < bin_edges[i+1]) & (centers[:, slice_dir] > bin_edges[i])
            centers_slice = centers[ibool, :]
            fig = plt.figure(figsize=(7, 7))
            idx = range(3)
            idx.remove(slice_dir)
            plt.scatter(centers_slice[:,idx[0]], centers_slice[:,idx[1]], edgecolors="none", c="green", alpha=0.5, s=pt_size)
            if plot_ext:
                ibool = (variables[slice_dir] < bin_edges[i+1]) & (variables[slice_dir] > bin_edges[i])
                plt.scatter(variables[idx[0]][ibool], variables[idx[1]][ibool], edgecolors="none", c="red", s=pt_size_ext, alpha=alpha_ext)
            plt.xlabel(var_names[idx[0]], fontsize=25)
            plt.ylabel(var_names[idx[1]], fontsize=25)

            if guide and (slice_dir==2):
                plt.plot(x_guide, y_guide, c="orange", lw = 2)
            # plt.axis("equal")
            plt.xlim(limits[idx[0]])
            plt.ylim(limits[idx[1]])
            title_str = "%s [%.3f, %.3f]" % (var_names[slice_dir], bin_edges[i], bin_edges[i+1])
            print i, title_str
            plt.title(title_str, fontsize=25, y =1.05)
            plt.savefig(save_dir+prefix+"-boundary-%s-%d.png" % (slice_var_tag[slice_dir], i), bbox_inches="tight", dpi=200)
            plt.close()        


