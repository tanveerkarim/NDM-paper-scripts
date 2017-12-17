from utils import *

large_random_constant = -999119283571
deg2arcsec=3600

dr_v = "5" # Data release version
data_directory = "../data/DR"+dr_v+"/"
tycho_directory = "../data/"
windowf_directory= "../data/DEEP2/windowf/"
save_dir = "../data/derived/"

##############################################################################	
print("Combine all Tractor files by field, append Tycho-2 stellar mask column and mask objects using DEEP2 window funtions.")
# Field 2
tracf2 = combine_tractor_nocut(data_directory+"f2/", all_models = False)
# Field 3
tracf3 = combine_tractor_nocut(data_directory+"f3/", all_models = False)
# Field 4
tracf4 = combine_tractor_nocut(data_directory+"f4/", all_models = False)
print("Completed.")


print("2. Impose DEEP2 window functions.")
# Field 2
idx = np.logical_or(window_mask(tracf2["ra"], tracf2["dec"], windowf_directory+"windowf.21.fits"), window_mask(tracf2["ra"], tracf2["dec"], windowf_directory+"windowf.22.fits"))
tracf2 = tracf2[idx]

# Field 3
idx = np.logical_or.reduce((window_mask(tracf3["ra"], tracf3["dec"], windowf_directory+"windowf.31.fits"), window_mask(tracf3["ra"], tracf3["dec"], windowf_directory+"windowf.32.fits"),window_mask(tracf3["ra"], tracf3["dec"], windowf_directory+"windowf.33.fits")))
tracf3 = tracf3[idx]

# Field 4
idx = np.logical_or(window_mask(tracf4["ra"], tracf4["dec"], windowf_directory+"windowf.41.fits"), window_mask(tracf4["ra"], tracf4["dec"], windowf_directory+"windowf.42.fits"))
tracf4 = np.copy(tracf4[idx])
print("Completed.")


print("3. Append Tycho2 stark mask field.")
# Field 2 
tracf2 = apply_tycho(tracf2, tycho_directory+"tycho2.fits", galtype="ELG")
# Field 3
tracf3 = apply_tycho(tracf3, tycho_directory+"tycho2.fits", galtype="ELG")
# Field 4 
tracf4 = apply_tycho(tracf4, tycho_directory+"tycho2.fits", galtype="ELG")
print("Completed.")




##############################################################################	
print("4. Save the trimmed catalogs.")
# Field 2
cols = fits.ColDefs(tracf2)
tbhdu = fits.BinTableHDU.from_columns(cols)
tbhdu.writeto(save_dir+"DR"+dr_v+"-Tractor-D2f2.fits", clobber=True)

# Field 3
cols = fits.ColDefs(tracf3)
tbhdu = fits.BinTableHDU.from_columns(cols)
tbhdu.writeto(save_dir+"DR"+dr_v+"-Tractor-D2f3.fits", clobber=True)

# Field 4
cols = fits.ColDefs(tracf4)
tbhdu = fits.BinTableHDU.from_columns(cols)
tbhdu.writeto(save_dir+"DR"+dr_v+"-Tractor-D2f4.fits", clobber=True)
print("Completed.")


##############################################################################	
print("5. Check that the files were correctly saved by comparing the number of objects.")

# Field 2
nobjs_before = tracf2.size
tracf2 = load_fits_table(save_dir+"DR5-Tractor-D2f2.fits")
nobjs_after = tracf2.size
print "Field: Before vs. After"
print "F2: %d / %d" % (nobjs_before, nobjs_after)

# Field 3
nobjs_before = tracf3.size
tracf3 = load_fits_table(save_dir+"DR5-Tractor-D2f3.fits")
nobjs_after = tracf3.size
print "Field: Before vs. After"
print "F3: %d / %d" % (nobjs_before, nobjs_after)

# Field 4
nobjs_before = tracf4.size
tracf4 = load_fits_table(save_dir+"DR5-Tractor-D2f4.fits")
nobjs_after = tracf4.size
print "Field: Before vs. After"
print "F4: %d / %d" % (nobjs_before, nobjs_after)


