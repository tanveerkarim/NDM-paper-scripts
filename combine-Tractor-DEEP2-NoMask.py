# Load modules
from utils import *

def load_tractor_DR5(fname, ibool=None):
    """
    Load select columns
    """
    tbl = load_fits_table(fname)    
    if ibool is not None:
        tbl = tbl[ibool]

    ra, dec = load_radec(tbl)
    bid = tbl["brickid"]
    bp = tbl["brick_primary"]
    r_dev, r_exp = tbl["shapedev_r"], tbl["shapeexp_r"]
    gflux_raw, rflux_raw, zflux_raw = tbl["flux_g"], tbl["flux_r"], tbl["flux_z"]
    gflux, rflux, zflux = gflux_raw/tbl["mw_transmission_g"], rflux_raw/tbl["mw_transmission_r"],zflux_raw/tbl["mw_transmission_z"]
    givar, rivar, zivar = tbl["flux_ivar_g"], tbl["flux_ivar_r"], tbl["flux_ivar_z"]
    g_allmask, r_allmask, z_allmask = tbl["allmask_g"], tbl["allmask_r"], tbl["allmask_z"]
    objtype = tbl["type"]
    tycho = tbl["TYCHOVETO"]    
    
    return bid, objtype, tycho, bp, ra, dec, gflux_raw, rflux_raw, zflux_raw, gflux, rflux, zflux, givar, rivar, zivar, r_dev, r_exp, g_allmask, r_allmask, z_allmask

def load_DEEP2(fname, ibool=None):
    tbl = load_fits_table(fname)
    if ibool is not None:
        tbl = tbl[ibool]

    ra, dec = tbl["RA_DEEP"], tbl["DEC_DEEP"]
    tycho = tbl["TYCHOVETO"]
    B, R, I = tbl["BESTB"], tbl["BESTR"], tbl["BESTI"]
    cn = tbl["cn"]
    w = tbl["TARG_WEIGHT"]
    return ra, dec, tycho, B, R, I, cn, w


dir_derived = "../data/derived/"

glim = 24.25


##############################################################################
for i, fnum in enumerate([2, 3, 4]):
    print("----- Processing %d Field -----" % fnum)
    print("1. Load processed DEEP2 and Tractor catalogs.")
    print("DEEP2 catalogs.")
    ra_d2, dec_d2, tycho, B, R, I, cn, w = load_DEEP2(dir_derived + ("deep2-f%d-photo-redz-oii.fits" % fnum))
    deep2 = load_fits_table(dir_derived + ("deep2-f%d-photo-redz-oii.fits" % fnum))
    nobjs_d2_total = ra_d2.size
    print("# in DEEP2 pcat: %d\n" % nobjs_d2_total)


    # Tractor catalog and load stark mask immediately.
    print("DECaLS catalogs.")
    print("Apply Tycho-2 stellar mask and other masks. glim < %.1f" % glim)
    bid, objtype, tycho, bp, ra, dec, gflux_raw, rflux_raw, zflux_raw, gflux, rflux, zflux, givar, rivar, zivar, r_dev, r_exp, g_allmask, r_allmask, z_allmask = load_tractor_DR5(dir_derived + "DR5-Tractor-D2f%d.fits"%fnum)
    ibool = bp & (gflux > mag2flux(glim))
    bid, objtype, tycho, bp, ra, dec, gflux_raw, rflux_raw, zflux_raw, gflux, rflux, zflux, givar, rivar, zivar, r_dev, r_exp, g_allmask, r_allmask, z_allmask = load_tractor_DR5(dir_derived + "DR5-Tractor-D2f%d.fits"%fnum, ibool=ibool)
    trac = load_fits_table(dir_derived + "DR5-Tractor-D2f%d.fits"%fnum)[ibool]
    nobjs_trac_total = ra.size
    print("# in Tractor: %d\n" % nobjs_trac_total)

    print("Completed.\n")

    ##############################################################################
    print("2. Cross-matching Tractor objects with g<%.2f and DEEP2 objects within each field. Check astrometric difference." % glim) 

    idx1, idx2 = crossmatch_cat1_to_cat2(ra, dec, ra_d2, dec_d2, tol=1./3600.)
    ra_med_diff_f2, dec_med_diff_f2 = check_astrometry(ra[idx1], dec[idx1], ra_d2[idx2], dec_d2[idx2], pt_size=0.1)
    nobjs_matched = idx1.size
    print("# of matches %d" % nobjs_matched)
    print("ra, dec median differences in arcsec: %.3f, %.3f\n" %(ra_med_diff_f2*deg2arcsec, dec_med_diff_f2*deg2arcsec))

    print("Completed.\n")


    # ##############################################################################
    print("3. Crossmatch the catalogs taking into account astrometric differences.")
    print("Save 1) DECaLS appended with DEEP2 and 2) Unmatched DEEP2.")
    idx1, idx2 = crossmatch_cat1_to_cat2(ra+ra_med_diff_f2, dec+dec_med_diff_f2, ra_d2, dec_d2, tol=1./3600.)
    
    # # Save unmatched DEEP2
    # deep2_unmatched = deep2[np.setdiff1d(np.arange(nobjs_d2_total, dtype=int),idx2)]
    # save_fits(deep2_unmatched, dir_derived + "unmatched-deep2-f%d-photo-redz-oii.fits" % fnum)

    # Append DEEP2 info for matched DR3 objects.
    append_list = [('OBJNO', '>i4'), ('BESTB', '>f4'), ('BESTR', '>f4'), ('BESTI', '>f4'), ('BESTBERR', '>f4'), ('BESTRERR', '>f4'), ('BESTIERR', '>f4'), ('BADFLAG', 'u1'), ('OII_3727', '>f8'), ('OII_3727_ERR', '>f8'), ('RED_Z', '>f8'), ('Z_ERR', '>f8'), ('ZQUALITY', '>f8'), ('TARG_WEIGHT', '>f8'), ('BRI_cut', '>f8'), ('cn', '>f8')]
    print("Following columns have been appended.")
    print(append_list)
    trac_new = None
    for e in append_list:
        if trac_new is None:
            trac_new = fits_append(trac, deep2[e[0]], e[0], idx1, idx2)
        else:
            trac_new = fits_append(trac_new, deep2[e[0]], e[0], idx1, idx2)

    # One more field to indicate whether Trac objects were matched in DEEP2 or not. If found 1.
    DEEP2_matched = np.zeros(nobjs_trac_total, dtype=int)
    DEEP2_matched[idx1] = 1
    trac_new = rec.append_fields(trac_new, "DEEP2_matched", DEEP2_matched, dtypes=DEEP2_matched.dtype, usemask=False, asrecarray=True)
    save_fits(trac_new, dir_derived + "DR5-matched-to-DEEP2-f%d-glim24p25-NoMask.fits" % fnum)
    print("Completed.\n")

    print("Comparison of number of objects before and after save.")
    nobjs = trac_new.size
    trac_new = load_fits_table(dir_derived + "DR5-matched-to-DEEP2-f%d-glim24p25-NoMask.fits" % fnum)
    nobjs_after = trac_new.size

    print "F%d: %d / %d" % (fnum, nobjs, nobjs_after)

    print("\n\n\n")