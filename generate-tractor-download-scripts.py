# Loading modules
from utils import *

large_random_constant = -999119283571
deg2arcsec=3600

dr_v = "5" # Data release version
data_directory = "../data/DR"+dr_v+"/"
brick_fname = "survey-bricks-dr"+dr_v+".fits"


##############################################################################
print("This script generates three files that the user can use to download the relevant tractor files.")
print("To identify relevant bricks, use survey-bricks-dr"+dr_v+".fits which the user should already have downloaded. Approximate field ranges.\n\
\n\
Field 2\n\
RA bounds: [251.3, 253.7]\n\
DEC bounds: [34.6, 35.3]\n\
\n\
Field 3\n\
RA bounds: [351.25, 353.8]\n\
DEC bounds: [-.2, .5]\n\
\n\
Field 4\n\
RA bounds: [36.4, 38]\n\
DEC bounds: [.3, 1.0]\n\
")

fits_bricks = fits.open(data_directory+brick_fname)[1].data
ra = fits_bricks['ra'][:]
dec = fits_bricks['dec'][:]
br_name = fits_bricks['brickname'][:]

# Getting the brick names near the ranges specified below.
tol = 0.25
f2_bricks = return_bricknames(ra, dec, br_name,[251.3, 253.7],[34.6, 35.3],tol)
f3_bricks = return_bricknames(ra, dec, br_name,[351.25, 353.8],[-.2, .5],tol)
f4_bricks = return_bricknames(ra, dec, br_name,[36.4,38.],[.3, 1.0],tol)
bricks = [f2_bricks, f3_bricks, f4_bricks]

print("Generating download scripts. tractor-download-D2f**.sh")
portal_address = "http://portal.nersc.gov/project/cosmo/data/legacysurvey/dr"+dr_v+"/tractor/"
postfix = ".fits\n"
prefix = "wget "
for i in range(3):
    f = open("tractor-download-d2f%d.sh"%(i+2),"w")
    for brick in bricks[i]:
        tractor_directory = brick[:3]
        brick_address = tractor_directory+"/tractor-"+brick+postfix
        download_command = prefix + portal_address + brick_address
        f.write(download_command)
    f.close()
print("Completed")