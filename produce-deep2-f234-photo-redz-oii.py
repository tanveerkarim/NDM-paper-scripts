from utils import *

# Data directory (set by the user)
data_directory = "../data/DEEP2/pcats/"
data_directory_cleaned = "../data/derived/"
zcat_directory = "../data/DEEP2/redz-oii-catalog/"
save_dir = "../data/derived/"
color_directory = "../data/DEEP2/"
windowf_directory= "../data/DEEP2/windowf/"
tycho_directory = "../data/"


##############################################################################
print("1. Load and combine DEEP2 extended pcat files for Fields 2, 3, and 4, respectively.")
print("File names for Fields 2, 3 and 4: pcat_ext.**.fits")
print("Special treatment to Field 4 data.")

# fp = 21, 22
pcat21 = fits.open(data_directory+"pcat_ext.21.fits")
pcat22 = fits.open(data_directory+"pcat_ext.22.fits")
pcat2 = np.hstack((pcat21[1].data, pcat22[1].data))
pcat21.close() # Closing files
pcat22.close()

# fp = 31, 32, 33
pcat31 = fits.open(data_directory+"pcat_ext.31.fits")
pcat32 = fits.open(data_directory+"pcat_ext.32.fits")
pcat33 = fits.open(data_directory+"pcat_ext.33.fits")
pcat3 = np.hstack((pcat31[1].data, pcat32[1].data,pcat33[1].data))
pcat31.close() # Closing files
pcat32.close()
pcat33.close()

# field4
# fp = 41, 42, 43
pcat41 = fits.open(data_directory+"pcat_ext.41.fits")[1].data
pcat42 = fits.open(data_directory+"pcat_ext.42.fits")[1].data
pcat43 = fits.open(data_directory+"pcat_ext.43.fits")[1].data
# Cross-match 43 and 41 to 42.

idx41in42, _ = cross_match_catalogs(pcat41, pcat42)
idx43in42, _ = cross_match_catalogs(pcat43, pcat42)

ibool1 = np.ones(pcat41.shape[0], dtype=bool)
ibool1[idx41in42] = False
ibool3 = np.ones(pcat43.shape[0], dtype=bool)
ibool3[idx43in42] = False

pcat4 = np.hstack((np.copy(pcat41[ibool1]), np.copy(pcat42), np.copy(pcat43[ibool3])))

# Or use cleaned file provided by Newman
# pcat4 = fits.open(data_directory_cleaned+"deep2-f4-photo-newman.fits")[1].data

print("Completed.\n")


##############################################################################
print("2. Mask BADFLAG==0 objects.")
# Field 2
pcat2_good = pcat2[pcat2["BADFLAG"]==0]

# Field 3
pcat3_good = pcat3[pcat3["BADFLAG"]==0]

# Field 4
pcat4_good = pcat4[pcat4["BADFLAG"]==0]

print("Completed.\n")


##############################################################################
print("3. Mask objects that lie outside DEEP2 window functions.")
# Field 2
idx = np.logical_or(window_mask(pcat2_good["RA_DEEP"], pcat2_good["DEC_DEEP"], windowf_directory+"windowf.21.fits"), window_mask(pcat2_good["RA_DEEP"], pcat2_good["DEC_DEEP"], windowf_directory+"windowf.22.fits"))
pcat2_trimmed = pcat2_good[idx]

# Field 3
idx = np.logical_or.reduce((window_mask(pcat3_good["RA_DEEP"], pcat3_good["DEC_DEEP"], windowf_directory+"windowf.31.fits"), window_mask(pcat3_good["RA_DEEP"], pcat3_good["DEC_DEEP"], windowf_directory+"windowf.32.fits"),window_mask(pcat3_good["RA_DEEP"], pcat3_good["DEC_DEEP"], windowf_directory+"windowf.33.fits")))
pcat3_trimmed = pcat3_good[idx]

# Field 4
idx = np.logical_or(window_mask(pcat4_good["RA_DEEP"], pcat4_good["DEC_DEEP"], windowf_directory+"windowf.41.fits"), window_mask(pcat4_good["RA_DEEP"], pcat4_good["DEC_DEEP"], windowf_directory+"windowf.42.fits"))
# idx = np.logical_or(window_mask(pcat4_good["RA"], pcat4_good["DEC"], windowf_directory+"windowf.41.fits"), window_mask(pcat4_good["RA"], pcat4_good["DEC"], windowf_directory+"windowf.42.fits"))
pcat4_trimmed = np.copy(pcat4_good[idx])

print("Completed.\n")


##############################################################################
print("4. Save the trimmed DEEP2 photometric catalogs as deep2-f**-photo-trimmed.fits.")
# Field 2
cols = fits.ColDefs(np.copy(pcat2_trimmed))
tbhdu = fits.BinTableHDU.from_columns(cols)
tbhdu.writeto(data_directory_cleaned+'deep2-f2-photo-trimmed.fits', clobber=True)

# Field 3
cols = fits.ColDefs(np.copy(pcat3_trimmed))
tbhdu = fits.BinTableHDU.from_columns(cols)
tbhdu.writeto(data_directory_cleaned+'deep2-f3-photo-trimmed.fits', clobber=True)

# Field 4
cols = fits.ColDefs(np.copy(pcat4_trimmed))
tbhdu = fits.BinTableHDU.from_columns(cols)
tbhdu.writeto(data_directory_cleaned+'deep2-f4-photo-trimmed.fits', clobber=True)

print("Completed.\n")



##############################################################################
print("6. Load other catalogs.")
print("color-selection.txt: Catalog that provides DEEP2 BRI color selection information.\n \
    Contains object ID, RA, dec, whether the object would have been targeted if in EGS. \n \
    (1=yes, 0=no), and whether it would have been targeted in a non-EGS field.\n \
    Provided by Jeff Newman")
DEEP2color = ascii.read(color_directory+"color-selection.txt")
DEEP2color_OBJNO = np.asarray(DEEP2color["col1"])
DEEP2color_ra = np.asarray(DEEP2color["col2"])
DEEP2color_dec = np.asarray(DEEP2color["col3"])
# DEEP2color_EGS=np.asarray(DEEP2color["col4"])
DEEP2color_BRI=np.asarray(DEEP2color["col5"])
print("Completed.\n")


print("deep2-f**-redz-oii.fits: DEEP2 redshift catalogs that John Moustakas provided.\n \
	Extract OBJNO, RA, DEC, OII_3727, OII_3727_ERR, ZHELIO, ZHELIO_ERR, ZQUALITY.\n \
	Note 1: Negative errors have the following meaning\n \
    \t-1.0 = line not detected with amplitude S/N > 1.5. Upper limit calculated.\n \
    \t-2.0 = line not measured (not in spectral range)\n \
	Note 2: For ZQUALITY values, see http://deep.ps.uci.edu/DR4/zquality.html.")
# Field 2
f2_objno, f2_ra, f2_dec, f2_oii, f2_oii_err, f2_z, f2_z_err, f2_zquality, f2_weight = import_zcat(zcat_directory+"deep2-f2-redz-oii.fits")

# Field 3
f3_objno, f3_ra, f3_dec, f3_oii, f3_oii_err, f3_z, f3_z_err, f3_zquality, f3_weight = import_zcat(zcat_directory+"deep2-f3-redz-oii.fits")

# Field 4
f4_objno, f4_ra, f4_dec, f4_oii, f4_oii_err, f4_z, f4_z_err, f4_zquality, f4_weight = import_zcat(zcat_directory+"deep2-f4-redz-oii.fits")
print("Completed.\n")



##############################################################################
print("7. Append additional columns to the photometric catalogs from others.")
print("7a. Append redshift catalogs.")
col_name_list = ["OBJNO_zcat", "RA_zcat", "DEC_zcat", "OII_3727","OII_3727_ERR", "RED_Z", "Z_ERR", "Z_QUALITY", "TARG_WEIGHT"]
print("Columns added: " + ", ".join(col_name_list))
# Field 2
pcat2 = fits.open(save_dir + "deep2-f2-photo-trimmed.fits")[1].data
idx1, idx2 = match_objno(pcat2["OBJNO"], f2_objno)
new_col_list = [f2_objno, f2_ra, f2_dec, f2_oii, f2_oii_err, f2_z, f2_z_err, f2_zquality, f2_weight]
for i in range(len(new_col_list)):
    pcat2 = pcat_append(pcat2, new_col_list[i], col_name_list[i], idx1, idx2)
    
# Field 3
pcat3 = fits.open(save_dir + "deep2-f3-photo-trimmed.fits")[1].data
idx1, idx2 = match_objno(pcat3["OBJNO"], f3_objno)
new_col_list = [f3_objno, f3_ra, f3_dec, f3_oii, f3_oii_err, f3_z, f3_z_err, f3_zquality, f3_weight]
for i in range(len(new_col_list)):
    pcat3 = pcat_append(pcat3, new_col_list[i], col_name_list[i], idx1, idx2)

# Field 4
del pcat4
pcat4 = fits.open(save_dir + "deep2-f4-photo-trimmed.fits")[1].data
idx1, idx2 = match_objno(pcat4["OBJNO"], f4_objno)
new_col_list = [f4_objno, f4_ra, f4_dec, f4_oii, f4_oii_err, f4_z, f4_z_err, f4_zquality, f4_weight]
for i in range(len(new_col_list)):
    pcat4 = pcat_append(pcat4, new_col_list[i], col_name_list[i], idx1, idx2)

print("f2: # in zcat minus # in pcat matched %d" % (f2_objno.size-(pcat2["RED_Z"]>-1000).sum()))
print("f3: # in zcat minus # in pcat matched %d" % (f3_objno.size-(pcat3["RED_Z"]>-1000).sum()))
print("f4: # in zcat minus # in pcat matched %d" % (f4_objno.size-(pcat4["RED_Z"]>-1000).sum()))    
print("The number of overlapping objects are smaller because certain\n \
spectroscopic areas were masked out in previous steps (the area\n \
estimates above are compatible). I ignore the small number of object\n \
loss.")
print("Completed.\n")



print("7c. Append color-selection information.")
# List of new columns to be appended
col_name_list = ["OBJNO_color", "BRI_cut"]
print("Columns added: " + ", ".join(col_name_list))
new_col_list = [DEEP2color_OBJNO, DEEP2color_BRI]
# Field 2
idx1, idx2 = match_objno(pcat2["OBJNO"], DEEP2color_OBJNO)
for i in range(len(new_col_list)):
    pcat2 = pcat_append(pcat2, new_col_list[i], col_name_list[i], idx1, idx2)
    
# Field 3
idx1, idx2 = match_objno(pcat3["OBJNO"], DEEP2color_OBJNO)
for i in range(len(new_col_list)):
    pcat3 = pcat_append(pcat3, new_col_list[i], col_name_list[i], idx1, idx2)    

# Field 4
idx1, idx2 = match_objno(pcat4["OBJNO"], DEEP2color_OBJNO)
for i in range(len(new_col_list)):
    pcat4 = pcat_append(pcat4, new_col_list[i], col_name_list[i], idx1, idx2)
print("f2: # in pcat minus # in pcat matched %d" % (pcat2.shape[0]-(pcat2["BRI_cut"]>-1000).sum()))
print("f3: # in pcat minus # in pcat matched %d" % (pcat3.shape[0]-(pcat3["BRI_cut"]>-1000).sum()))
print("f4: # in pcat minus # in pcat matched %d" % (pcat4.shape[0]-(pcat4["BRI_cut"]>-1000).sum()))    
print("The last 223 objects will be classified as DEEP2 BRI color rejected objects.")
print("Completed.\n")


print("7d. Append Tycho2 stark mask field.")
# Field 2 
pcat2 = apply_tycho_pcat(pcat2, tycho_directory+"tycho2.fits", galtype="ELG")
# Field 3
pcat3 = apply_tycho_pcat(pcat3, tycho_directory+"tycho2.fits", galtype="ELG")
# Field 4 
pcat4 = apply_tycho_pcat(pcat4, tycho_directory+"tycho2.fits", galtype="ELG")
print("Completed.")





##############################################################################
print("8. Compute the class number based on the information above and append to the table.")
print("Recall: ")
for cn in range(len(cnames)):
    print("cn%d: %s"% (cn, cnames[cn]))

col_name_list = ["cn"]
# Field 2
new_col_list = [generate_class_col(pcat2)]
idx = range(pcat2.shape[0])
for i in range(len(new_col_list)):
    pcat2 = pcat_append(pcat2, new_col_list[i], col_name_list[i], idx, idx)

    
# Field 3
new_col_list = [generate_class_col(pcat3)]
idx = range(pcat3.shape[0])
for i in range(len(new_col_list)):
    pcat3 = pcat_append(pcat3, new_col_list[i], col_name_list[i], idx, idx)
      

# Field 4
new_col_list = [generate_class_col(pcat4)]
idx = range(pcat4.shape[0])
for i in range(len(new_col_list)):
    pcat4 = pcat_append(pcat4, new_col_list[i], col_name_list[i], idx, idx)

print("Category counts 0 through %d" % (len(cnames)-1))
print(np.bincount(pcat2["cn"].astype(int)))
print(np.bincount(pcat3["cn"].astype(int)))
print(np.bincount(pcat4["cn"].astype(int)))
print("Completed.\n")


##############################################################################
print("9. Save the resulting catalogs.")
# Field 2
cols = fits.ColDefs(pcat2)
tbhdu = fits.BinTableHDU.from_columns(cols)
tbhdu.writeto(save_dir + 'deep2-f2-photo-redz-oii.fits', clobber=True)

# Field 3
cols = fits.ColDefs(pcat3)
tbhdu = fits.BinTableHDU.from_columns(cols)
tbhdu.writeto(save_dir + 'deep2-f3-photo-redz-oii.fits', clobber=True)

# Field 4
cols = fits.ColDefs(pcat4)
tbhdu = fits.BinTableHDU.from_columns(cols)
tbhdu.writeto(save_dir + 'deep2-f4-photo-redz-oii.fits', clobber=True)
print("Completed.\n")

print "Check number of objects before and after save."
print "Field: Before vs. After"
# Field 2
nobjs_before = pcat2.size
pcat2 = load_fits_table(save_dir + "deep2-f2-photo-redz-oii.fits")
nobjs_after = pcat2.size
print "F2: %d / %d" % (nobjs_before, nobjs_after)

# Field 3
nobjs_before = pcat3.size
pcat3 = load_fits_table(save_dir + "deep2-f3-photo-redz-oii.fits")
nobjs_after = pcat3.size
print "F3: %d / %d" % (nobjs_before, nobjs_after)

# Field 4
nobjs_before = pcat4.size
pcat4 = load_fits_table(save_dir + "deep2-f4-photo-redz-oii.fits")
nobjs_after = pcat4.size
print "F4: %d / %d" % (nobjs_before, nobjs_after)