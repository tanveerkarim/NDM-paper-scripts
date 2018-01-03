# - Import DR1 catalogs.
# - Check whether the original catalogs were compiled using the correct condition. 
# - Match observed targets to DR1 catalogs
# - Match DR3 to DR1 and append appropriate columns.

# Load modules
import numpy as np
from xd_elg_utils import *
from astropy.io import fits as FITS
import matplotlib.pyplot as plt


# Constants
colors = ["orange", "grey", "brown", "purple", "red", "salmon","black", "white","blue"]
cnames = ["Gold", "Silver", "LowOII", "NoOII", "LowZ", "NoZ", "D2reject", "DR3unmatched","D2unobserved"]


MMT_data_dir = "./MMT_data/"
check_color = False # Only if you want to check DEEP2 rejection.
plot_radec_23 = False # True if you want to plot radec of targets in 23hr field 


##############################################################################
print("1. Import original targets DR1 and check whether they conform to the\n\
proper cuts in DR1.")
print("16hr targets")
fits = load_fits_table(MMT_data_dir+"MMTtargets-DEEP2-16hrs.fits")
ra16, dec16 = load_radec(fits)
grz16 = load_grz(fits)
iColor = MMT_study_color(grz16, 0)
iDECaLS_quality = MMT_DECaLS_quality(fits)
print("# of targets: %d" % iColor.size)
print("# color violation: ",-(iColor.sum()-iColor.size))
print("# decals quality violation", -(iDECaLS_quality.sum()-iDECaLS_quality.size))
print("\n")


print("23hr targets")
fits = load_fits_table(MMT_data_dir+"MMTtargets-DEEP2-23hrs.fits")
ra23, dec23 = load_radec(fits)
# bp23 = load_brick_primary(fits)
grz23 = load_grz(fits)
iColor = MMT_study_color(grz23, 1)
iDECaLS_quality = MMT_DECaLS_quality(fits)
print("# of targets: %d" % iColor.size)
print("# color violation: ",-(iColor.sum()-iColor.size))
print("# decals quality violation", -(iDECaLS_quality.sum()-iDECaLS_quality.size))
print("\n")


print("Note: We do not deal with re-obs targets as there are very few of them.")


##############################################################################
print("Check whether the targets are DEEP2 color rejected objects.")
if check_color:
	print("color-selection.txt: Catalog that provides DEEP2 BRI color selection information.\n \
	    Contains object ID, RA, dec, whether the object would have been targeted if in EGS. \n \
	    (1=yes, 0=no), and whether it would have been targeted in a non-EGS field.\n \
	    Provided by Jeff Newman")
	DEEP2color = ascii.read("color-selection.txt")
	DEEP2color_OBJNO = np.asarray(DEEP2color["col1"])
	DEEP2color_ra = np.asarray(DEEP2color["col2"])
	DEEP2color_dec = np.asarray(DEEP2color["col3"])
	# DEEP2color_EGS=np.asarray(DEEP2color["col4"])
	DEEP2color_BRI=np.asarray(DEEP2color["col5"])
	print("Completed.\n")

	print("Match ra/dec and verify the BRI color selection")
	print("16hr")
	idx1, idx2 = crossmatch_cat1_to_cat2(ra16, dec16, DEEP2color_ra,DEEP2color_dec)
	print("# of unique objects matched in the original catalog: %d"% np.unique(idx1.size))
	ireject = DEEP2color_BRI[idx2]==0
	print("# with color rejection %d"% ireject.sum())
	print("\n")

	print("23hr obs")
	idx1, idx2 = crossmatch_cat1_to_cat2(ra23, dec23,  DEEP2color_ra,DEEP2color_dec)
	print("# of unique objects matched in the original catalog: %d"% np.unique(idx1.size))
	ireject = DEEP2color_BRI[idx2]==0
	print("# with color rejection %d"% ireject.sum())
	print("\n")
else:
	print("Color-check skipped.\n")


##############################################################################
print("2. Import observed target ra/dec and their fiber number.")

ra1, dec1, fib1 = MMT_radec(0) # 16hr_1
ra2, dec2, fib2 = MMT_radec(1) # 16hr_2
ra3, dec3, fib3 = MMT_radec(2) # 16hr_3

print("# of targets")
print("16hr 1/2: %d, %d"%(ra1.size, ra2.size))
print("23hr: %d"%(ra3.size))
print("Total: %d"%(ra1.size+ra2.size+ra3.size))

##############################################################################
print("Match observed target to DR1.")

print("# Matched in 16hr_1")
idx1_16_1, idx2_16_1 = crossmatch_cat1_to_cat2(ra1, dec1, ra16, dec16)
print("# observed: %d"%ra1.size)
print("# of unique objects matched in the original catalog: %d"% np.unique(idx1_16_1.size))
print("\n")

print("# Matched in 16hr_2")
idx1_16_2, idx2_16_2 = crossmatch_cat1_to_cat2(ra2, dec2, ra16, dec16)
print("# observed: %d"%ra2.size)
print("# of unique objects matched in the original catalog: %d"% np.unique(idx1_16_2.size))
print("\n")

print("# Matched in 23hr")
idx1_23, idx2_23 = crossmatch_cat1_to_cat2(ra3, dec3, ra23, dec23)
print("# observed: %d"%ra3.size)
print("# of unique objects matched in the original catalog: %d"% np.unique(idx1_23.size))
print("\n")
num_primary_23 = idx1_23.size

print("To account for the unmatched objects in 23hr field, we plot ra/dec.\n\
See radec-MMT-23hr-targets.png")
if plot_radec_23:
	pt_size=15
	fig = plt.figure(figsize=(7,7))
	plt.scatter(ra3[idx1_23], dec3[idx1_23], edgecolors="none", s=pt_size, label="Matched")
	idx1_complement = np.setdiff1d(range(ra3.size),idx1_23)
	plt.scatter(ra3[idx1_complement], dec3[idx1_complement], edgecolors="none",c="red",s=pt_size, label="Obs. unmatched")
	iraLT352 =ra23<352.3
	plt.scatter(ra23[iraLT352], dec23[iraLT352], edgecolors="none",c="black", s=pt_size, label="Originally proposed")
	plt.axis("equal")
	plt.legend(loc="lower right")
	plt.savefig("radec-MMT-23hr-targets.png", dpi=400, bbox_inches="tight")
	plt.close()
	print("Completed.\n")
else:
	print("Skipped.\n")






##############################################################################
print("3. Import DR3 catalogs and match DR1 to these catalogs.\n")
# Field 2
DR3_16 = load_fits_table("DECaLS-DR3-Tractor-DEEP2f2-untrimmed.fits")
ra_DR3_16, dec_DR3_16 = load_radec(DR3_16)

# Field 3
DR3_23 = load_fits_table("DECaLS-DR3-Tractor-DEEP2f3-untrimmed.fits")
ra_DR3_23, dec_DR3_23 = load_radec(DR3_23)


print("Match observed objects to DR3 catalogs directly.")
print("16hr_1 to DR3 16hr")
idx1_16_1_DR3, idx2_16_1_DR3 = crossmatch_cat1_to_cat2(ra1, dec1, ra_DR3_16, dec_DR3_16)
print("# observed: %d"%ra1.size)
print("# of unique objects matched in the DR3 catalog: %d"% np.unique(idx1_16_1_DR3).size)
print("\n")

print("16hr_2 to DR3 16hr")
idx1_16_2_DR3, idx2_16_2_DR3 = crossmatch_cat1_to_cat2(ra2, dec2, ra_DR3_16, dec_DR3_16)
print("# observed: %d"%ra2.size)
print("# of unique objects matched in the DR3 catalog: %d"% np.unique(idx1_16_2_DR3).size)
print("\n")

print("23hr obs to DR3 23hr")
idx1_23_DR3, idx2_23_DR3 = crossmatch_cat1_to_cat2(ra3, dec3, ra_DR3_23, dec_DR3_23)
num_primary_23 = 135
print("# observed: %d"%num_primary_23)
print("# of unique objects matched in the DR3 catalog: %d"% np.unique(idx1_23_DR3).size)
print("\n")

print("Completed.\n")





##############################################################################
print("Match DR1 to DR3.")

print("16hr field")
table_16 = load_fits_table(MMT_data_dir+"MMTtargets-DEEP2-16hrs.fits")
numrows_16 = table_16.shape[0]
# col_data/col_name
col_data = []
col_name = []

# Observed in 16hr_1
observed_16 = np.zeros(numrows_16,dtype=np.int16)
observed_16[idx2_16_1] = 1
fib_16 = np.ones(numrows_16,dtype=np.int16)*-1
fib_16[idx2_16_1] = fib1
col_data+=[observed_16,fib_16]
col_name+=["OBSERVED_ONE", "FIB_NUM_ONE"]

# Observed in 16hr_2
observed_16 = np.zeros(numrows_16,dtype=np.int16)
observed_16[idx2_16_2] = 1
fib_16 = np.ones(numrows_16,dtype=np.int16)*-1
fib_16[idx2_16_2] = fib2
col_data+=[observed_16,fib_16]
col_name+=["OBSERVED_TWO", "FIB_NUM_TWO"]

# Match DR1 with DR3
matched_16 = np.zeros(numrows_16,dtype=np.int16)
idx1, idx2 = crossmatch_cat1_to_cat2(ra16, dec16, ra_DR3_16, dec_DR3_16)
matched_16[idx1]=1
col_data.append(matched_16)
col_name.append("DR3")

col_type = ["i4"]*len(col_data)
for i in range(len(col_data)):
    table_16 = fits_append(table_16, col_data[i], col_name[i], range(numrows_16), range(numrows_16), dtype="user", dtype_user=col_type[i])

# For matched objects
col_data=[]
col_name=[]
col_type=[]
# RA/DEC
ra, dec =  load_radec(DR3_16)
col_data+=[ra, dec]
col_name+=["DR3_RA", "DR3_DEC"]
col_type+=["f4"]*2
# Flux
gflux, rflux, zflux =  load_grz_flux(DR3_16)
col_data+=[gflux,rflux,zflux]
col_name+=["DR3_GFLUX","DR3_RFLUX","DR3_ZFLUX"]
col_type+=["f4"]*3
# Mag
g,r,z =  load_grz(DR3_16)
col_data+=[g,r,z]
col_name+=["DR3_G","DR3_R","DR3_Z"]
col_type+=["f4"]*3
# grz invar
gi, ri, zi =  load_grz_invar(DR3_16)
col_data+=[gi,ri,zi]
col_name+=["DR3_GINVAR","DR3_RINVAR","DR3_ZINVAR"]
col_type+=["f4"]*3
# grz all and any
gall, rall, zall =  load_grz_allmask(DR3_16)
gany, rany, zany =  load_grz_anymask(DR3_16)
col_data+=[gall, rall, zall, gany, rany, zany]
col_name+=["DR3_gall", "DR3_rall", "DR3_zall", "DR3_gany", "DR3_rany", "DR3_zany"]
col_type+=["i2"]*6
# brick primary
bp =  load_brick_primary(DR3_16)
col_data+=[bp]
col_name+=["DR3_BP"]
col_type+=["bool"]
# shape
r_dev, r_exp =  load_shape(DR3_16)
col_data+=[r_dev, r_exp]
col_name+=["DR3_RDEV", "DR3_REXP"]
col_type+=["f4"]*2


for i in range(len(col_data)):
    table_16 = fits_append(table_16, col_data[i], col_name[i], idx1, idx2, dtype="user", dtype_user=col_type[i])

cols = FITS.ColDefs(table_16)
tbhdu = FITS.BinTableHDU.from_columns(cols)
tbhdu.writeto('DR1-MMT-DR3-16hr.fits', clobber=True)

print("Completed.\n")
    
##############################################################################
print("23hr field")
table_23 = load_fits_table(MMT_data_dir+"MMTtargets-DEEP2-23hrs.fits")
numrows_23 = table_23.shape[0]
# col_data/col_name
col_data = []
col_name = []

# Observed in 23hr_1
observed_23 = np.zeros(numrows_23,dtype=np.int16)
observed_23[idx2_23] = 1
fib_23 = np.ones(numrows_23,dtype=np.int16)*-1
fib_23[idx2_23] = fib3
col_data+=[observed_23,fib_23]
col_name+=["OBSERVED", "FIB_NUM"]

# Match DR1 with DR3
matched_23 = np.zeros(numrows_23,dtype=np.int16)
idx1, idx2 = crossmatch_cat1_to_cat2(ra23, dec23, ra_DR3_23, dec_DR3_23)
matched_23[idx1]=1
col_data.append(matched_23)
col_name.append("DR3")

col_type = ["i4"]*len(col_data)
for i in range(len(col_data)):
    table_23 = fits_append(table_23, col_data[i], col_name[i], range(numrows_23), range(numrows_23), dtype="user", dtype_user=col_type[i])

# For matched objects
col_data=[]
col_name=[]
col_type=[]
# RA/DEC
ra, dec =  load_radec(DR3_23)
col_data+=[ra, dec]
col_name+=["DR3_RA", "DR3_DEC"]
col_type+=["f4"]*2
# Flux
gflux, rflux, zflux =  load_grz_flux(DR3_23)
col_data+=[gflux,rflux,zflux]
col_name+=["DR3_GFLUX","DR3_RFLUX","DR3_ZFLUX"]
col_type+=["f4"]*3
# Mag
g,r,z =  load_grz(DR3_23)
col_data+=[g,r,z]
col_name+=["DR3_G","DR3_R","DR3_Z"]
col_type+=["f4"]*3
# grz invar
gi, ri, zi =  load_grz_invar(DR3_23)
col_data+=[gi,ri,zi]
col_name+=["DR3_GINVAR","DR3_RINVAR","DR3_ZINVAR"]
col_type+=["f4"]*3
# grz all and any
gall, rall, zall =  load_grz_allmask(DR3_23)
gany, rany, zany =  load_grz_anymask(DR3_23)
col_data+=[gall, rall, zall, gany, rany, zany]
col_name+=["DR3_gall", "DR3_rall", "DR3_zall", "DR3_gany", "DR3_rany", "DR3_zany"]
col_type+=["i2"]*6
# brick primary
bp =  load_brick_primary(DR3_23)
col_data+=[bp]
col_name+=["DR3_BP"]
col_type+=["bool"]
# shape
r_dev, r_exp =  load_shape(DR3_23)
col_data+=[r_dev, r_exp]
col_name+=["DR3_RDEV", "DR3_REXP"]
col_type+=["f4"]*2

for i in range(len(col_data)):
    table_23 = fits_append(table_23, col_data[i], col_name[i], idx1, idx2, dtype="user", dtype_user=col_type[i])

cols = FITS.ColDefs(table_23)
tbhdu = FITS.BinTableHDU.from_columns(cols)
tbhdu.writeto('DR1-MMT-DR3-23hr.fits', clobber=True)

print("Completed.\n")


##############################################################################
num_16_1_in_DR3 = (table_16["DR3"][table_16["OBSERVED_ONE"]==1]).sum()
num_16_2_in_DR3 = (table_16["DR3"][table_16["OBSERVED_TWO"]==1]).sum()
num_23_in_DR3 = (table_23["DR3"][table_23["OBSERVED"]==1]).sum()

print("Matches.")
print("16hr_1 to DR3 16hr")
print("# observed: %d"%ra1.size)
print("# directly matched in DR3: %d"% idx1_16_1_DR3.size)
print("# matched DR3: %d"% num_16_1_in_DR3)
print("\n")

print("16hr_2 to DR3 16hr")
print("# observed: %d"%ra2.size)
print("# directly matched in DR3: %d"% idx1_16_2_DR3.size)
print("# matched DR3: %d"% num_16_2_in_DR3)
print("\n")

print("23hr obs to DR3 23hr")
print("# observed: %d"%num_primary_23)
print("# directly matched in DR3: %d"% idx1_23_DR3.size)
print("# matched DR3: %d"% num_23_in_DR3)
print("\n")