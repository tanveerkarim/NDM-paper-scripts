from utils import *

dir_derived = "../data/derived/"
print("Import estimated areas")
areas = np.load(dir_derived+"spec-area.npy")    

print("Import Intersection catalogs")
cns = []
weights = []
print("Field: Unmatched fraction")
for i, fnum in enumerate([2, 3, 4]):
    pcat = load_fits_table(dir_derived+"DR5-matched-to-DEEP2-f%d-glim24p25.fits" % fnum)
    cns.append(pcat["cn"].astype(int))
    weights.append(pcat["TARG_WEIGHT"])
    num_unmatched = (pcat["DEEP2_matched"] == 0).sum()
    print("F%d: %.2f%%" % (fnum, num_unmatched/float(cns[i].size) * 100))
print("\n")


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

# print("Completed.\n")
