from utils import *

dir_derived = "../data/derived/"
print("Import estimated areas")
areas = np.load(dir_derived+"spec-area.npy")    

print("Import DEEP2 combined catalogs")
cns = []
weights = []
for fnum in [2, 3, 4]:
    pcat = load_fits_table(dir_derived+"/deep2-f%d-photo-redz-oii.fits" % fnum)
    cns.append(pcat["cn"].astype(int))
    weights.append(pcat["TARG_WEIGHT"])

table_header = generate_table_header()
print("Raw number density")
print(table_header))
class_breakdown([2, 3, 4], cns, weights, areas, rwd="R")
print("\n")

print("Weighted number density")
print(table_header))
class_breakdown([2, 3, 4], cns, weights, areas, rwd="W")
print("\n")

print("The total densities must match.")

# print("Completed.\n"))
