# Load modules
import numpy as np
from xd_elg_utils import *
from astropy.io import fits as FITS
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join


# from scipy.signal import savgol_filter


# Constants
colors = ["orange", "grey", "brown", "purple", "red", "salmon","black", "white","blue"]
cnames = ["Gold", "Silver", "LowOII", "NoOII", "LowZ", "NoZ", "D2reject", "DR3unmatched","D2unobserved"]


MMT_data_dir = "./MMT_data/"

##############################################################################
print("Appending analyzed results to 23hr")
photo_data_fname = "DR1-MMT-DR3-23hr.fits"
inspection_fname = "23hr-panel-review.csv"
panel_dir = "./panel/23hr/"

# Load targets
print("Load target photo info.")
table = load_fits_table(photo_data_fname) 

print("Load inspection data")
inspection = np.loadtxt(inspection_fname, delimiter=",").astype("int")
panel_list = inspection[:,0]
guess_list = inspection[:,1]
confidence_list = inspection[:,2]

print("Get panel names")
onlyfiles = [f for f in listdir(panel_dir) if isfile(join(panel_dir, f))][1:]

print("For each panel in the panel list, get the corresponding info.")
num_panels = panel_list.size
fib_num = np.zeros(num_panels, dtype=np.int32)
redz = np.zeros(num_panels)
oii = np.zeros(num_panels)
s2n = np.zeros(num_panels)
# for i in range(num_panels):
for j in range(len(onlyfiles)):
    fname = onlyfiles[j]
    pn = int(fname.split("-")[0])# panel number
    fbn = int(fname.split("-")[2])
    guess = int(fname.split("guess")[1].split("-")[0])
    z = float(fname.split("z")[1].split("-")[0])
    oii_tmp = float(fname.split("oii")[1].split("-")[0])
    s2n_tmp = float(fname.split("s2n")[1].split(".png")[0])
    
    if pn in panel_list:
        idx = np.where(panel_list==pn)
        if guess_list[idx] == guess:
            fib_num[idx] = fbn
            redz[idx] = z
            oii[idx] = oii_tmp
            s2n[idx] = s2n_tmp
            
            print(pn, fbn, guess, confidence_list[idx][0], z, oii_tmp, "%.2f"%s2n_tmp)
            
print("Create new columns for the summary table.")
pn_col = np.ones(table.shape[0], dtype=np.int32)*-999
guess_col = np.ones(table.shape[0], dtype=np.int32)*-999
confidence_col = np.ones(table.shape[0], dtype=np.int32)*-999
z_col = np.ones(table.shape[0])*-999
oii_col = np.ones(table.shape[0])*-999
s2n_col = np.ones(table.shape[0])*-999

FIB_NUM = table["FIB_NUM"]
for i in range(num_panels):
    idx = np.where(FIB_NUM==int(fib_num[i]))[0][0]
    pn_col[idx] = panel_list[i]
    guess_col[idx] = guess_list[i]
    confidence_col[idx] = confidence_list[i]
    z_col[idx] = redz[i]
    oii_col[idx] = oii[i]
    s2n_col[idx] = s2n[i]
    print(idx)

print("Append to the table and save.")
col_data=[pn_col, guess_col, confidence_col, z_col, oii_col, s2n_col]
col_name=["PANEL_NUM", "GUESS_NUM", "CONFIDENCE", "REDZ", "OII", "S2N"]
col_type=["i4","i4","i4","f4","f4","f4"]

for i in range(len(col_data)):
    table = fits_append(table, col_data[i], col_name[i], range(table.shape[0]), range(table.shape[0]), dtype="user", dtype_user=col_type[i])

cols = FITS.ColDefs(table)
tbhdu = FITS.BinTableHDU.from_columns(cols)
tbhdu.writeto('DR1-MMT-DR3-23hr-oii-redz.fits', clobber=True)


print("Completed.\n")
##############################################################################
print("Appending analyzed results to 16hr")

#16hr_1

photo_data_fname = "DR1-MMT-DR3-16hr.fits"
inspection_fname = "16hr_1-panel-review.csv"
panel_dir = "./panel/16hr_1/"

# Load targets
print("Load target photo info.")
table = load_fits_table(photo_data_fname) 

print("Load inspection data")
inspection = np.loadtxt(inspection_fname, delimiter=",").astype("int")
panel_list = inspection[:,0]
guess_list = inspection[:,1]
confidence_list = inspection[:,2]


print("Get panel names")
onlyfiles = [f for f in listdir(panel_dir) if isfile(join(panel_dir, f))][1:]

print("For each panel in the panel list, get the corresponding info.")
num_panels = panel_list.size
fib_num = np.ones(num_panels, dtype=np.int32)*-999
redz = np.zeros(num_panels)
oii = np.zeros(num_panels)
s2n = np.zeros(num_panels)
print("pn", "fbn", "guess", "conf", "z", "oii","s2n")
for j in range(len(onlyfiles)):
    fname = onlyfiles[j]
    pn = int(fname.split("-")[0])# panel number
    fbn = int(fname.split("-")[2])
    guess = int(fname.split("guess")[1].split("-")[0])
    z = float(fname.split("z")[1].split("-")[0])
    oii_tmp = float(fname.split("oii")[1].split("-")[0])
    s2n_tmp = float(fname.split("s2n")[1].split(".png")[0])


    if pn in panel_list:
        idx = np.where(panel_list==pn)
        if guess_list[idx] == guess:
            fib_num[idx] = fbn
            redz[idx] = z
            oii[idx] = oii_tmp
            s2n[idx] = s2n_tmp            
            print(pn, fbn, guess, confidence_list[idx][0], z, oii_tmp, "%.2f"%s2n_tmp)
            
print("Create new columns for the summary table.")
pn_col = np.ones(table.shape[0], dtype=np.int32)*-999
guess_col = np.ones(table.shape[0], dtype=np.int32)*-999
confidence_col = np.ones(table.shape[0], dtype=np.int32)*-999
z_col = np.ones(table.shape[0])*-999
oii_col = np.ones(table.shape[0])*-999
s2n_col = np.ones(table.shape[0])*-999

FIB_NUM = table["FIB_NUM_ONE"]
for i in range(num_panels):
    idx = np.where(FIB_NUM==int(fib_num[i]))[0]
    if idx.size!=0:
        pn_col[idx] = panel_list[i]
        guess_col[idx] = guess_list[i]
        confidence_col[idx] = confidence_list[i]
        z_col[idx] = redz[i]
        oii_col[idx] = oii[i]
        s2n_col[idx] = s2n[i]

print("Append to the table and save.")
col_data=[pn_col, guess_col, confidence_col, z_col, oii_col, s2n_col]
col_name=["PANEL_NUM_ONE", "GUESS_NUM_ONE", "CONFIDENCE_ONE", "REDZ_ONE", "OII_ONE", "S2N_ONE"]
col_type=["i4","i4","i4","f4","f4","f4"]

for i in range(len(col_data)):
    table = fits_append(table, col_data[i], col_name[i], range(table.shape[0]), range(table.shape[0]), dtype="user", dtype_user=col_type[i])

    

    
# 16hr_2
inspection_fname = "16hr_2-panel-review.csv"
panel_dir = "./panel/16hr_2/"

# Load targets
print("Load inspection data")
inspection = np.loadtxt(inspection_fname, delimiter=",").astype("int")
panel_list = inspection[:,0]
guess_list = inspection[:,1]
confidence_list = inspection[:,2]

print("Get panel names")
onlyfiles = [f for f in listdir(panel_dir) if isfile(join(panel_dir, f))][1:]

print("For each panel in the panel list, get the corresponding info.")
num_panels = panel_list.size
fib_num = np.zeros(num_panels, dtype=np.int32)
redz = np.zeros(num_panels)
oii = np.zeros(num_panels)
s2n = np.zeros(num_panels)

print("pn", "fbn", "guess", "conf", "z", "oii","s2n")
for j in range(len(onlyfiles)):
    fname = onlyfiles[j]
    pn = int(fname.split("-")[0])# panel number
    fbn = int(fname.split("-")[2])
    guess = int(fname.split("guess")[1].split("-")[0])
    z = float(fname.split("z")[1].split("-")[0])
    oii_tmp = float(fname.split("oii")[1].split("-")[0])
    s2n_tmp = float(fname.split("s2n")[1].split(".png")[0])
    
    if pn in panel_list:
        idx = np.where(panel_list==pn)
        if guess_list[idx] == guess:
            fib_num[idx] = fbn
            redz[idx] = z
            oii[idx] = oii_tmp
            s2n[idx] = s2n_tmp
            print(pn, fbn, guess, confidence_list[idx][0], z, oii_tmp, "%.2f"%s2n_tmp)
            
print("Create new columns for the summary table.")
pn_col = np.ones(table.shape[0], dtype=np.int32)*-999
guess_col = np.ones(table.shape[0], dtype=np.int32)*-999
confidence_col = np.ones(table.shape[0], dtype=np.int32)*-999
z_col = np.ones(table.shape[0])*-999
oii_col = np.ones(table.shape[0])*-999
s2n_col = np.ones(table.shape[0])*-999

FIB_NUM = table["FIB_NUM_TWO"]
for i in range(num_panels):
    idx = np.where(FIB_NUM==int(fib_num[i]))[0][0]
    pn_col[idx] = panel_list[i]
    guess_col[idx] = guess_list[i]
    confidence_col[idx] = confidence_list[i]
    z_col[idx] = redz[i]
    oii_col[idx] = oii[i]
    s2n_col[idx] = s2n[i]


print("Append to the table and save.")
col_data=[pn_col, guess_col, confidence_col, z_col, oii_col, s2n_col]
col_name=["PANEL_NUM_TWO", "GUESS_NUM_TWO", "CONFIDENCE_TWO", "REDZ_TWO", "OII_TWO", "S2N_TWO"]
col_type=["i4","i4","i4","f4","f4","f4"]

for i in range(len(col_data)):
    table = fits_append(table, col_data[i], col_name[i], range(table.shape[0]), range(table.shape[0]), dtype="user", dtype_user=col_type[i])

    
cols = FITS.ColDefs(table)
tbhdu = FITS.BinTableHDU.from_columns(cols)
tbhdu.writeto('DR1-MMT-DR3-16hr-oii-redz.fits', clobber=True)
print("Completed.\n")