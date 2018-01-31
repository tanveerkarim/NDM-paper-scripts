from NDM_models import *

# Load marginal efficiency of fiducial selection.
data = np.load("./DR5-NDM1/NDM1-marginal_eff.npz")
centers, summary = data["centers"], data["summary"]

width = 10

print "/---- One priority"

idx_1st = closest_idx(centers, 2400)
print "Number of ELGs in the one priority system before fiber assignment."
Nelg = np.sum(summary[:idx_1st, -1]) * width
print "Nelg: %d" % Nelg
assignment_rate = 0.75
print "After taking into account assignment rate of %.3f"  % assignment_rate
print "Nelg: %d" % (Nelg * assignment_rate)

print "Total number of fibers filled over 14K: %.1f M" % (2400 * assignment_rate * 14000 / float(10**6))
print "\n\n"

print "/---- One priority extended"
N_desired = 3000
idx_1st = closest_idx(centers, N_desired)
print "Number of ELGs in the one priority system before fiber assignment."
Nelg = np.sum(summary[:idx_1st, -1]) * width
print "Nelg: %d" % Nelg
assignment_rate = 0.7
print "After taking into account assignment rate of %.3f"  % assignment_rate
Nelg_extended_single = (Nelg * assignment_rate)
print "Nelg: %d" % Nelg_extended_single

print "Total number of fibers filled over 14K: %.1f M" % (N_desired * assignment_rate * 14000 / float(10**6))
print "\n\n"


print "/---- Two priority"
idx_1st = closest_idx(centers, 1500)
idx_2nd = closest_idx(centers, 3000)
print "Number of ELGs in the two priority system before fiber assignment."
Nelg1 = np.sum(summary[:idx_1st, -1]) * width
Nelg2 = np.sum(summary[idx_1st:idx_2nd, -1]) * width
print "Nelg1/Nelg2: %d/%d" % (Nelg1, Nelg2)

assignment_rate1 = 0.9
assignment_rate2 = 0.5

print "After taking into account assignment rate of %.3f/ %.3f"  % (assignment_rate1, assignment_rate2)
print "Nelg1/Nelg2: %d/%d" % (Nelg1 * assignment_rate1, Nelg2 * assignment_rate2)
Nelg_extended_two = (Nelg1 * assignment_rate1 + Nelg2 * assignment_rate2)
print "Total: %d" % Nelg_extended_two
print "Total number of fibers filled over 14K: %.1f M" % \
(1500 * (assignment_rate1 + assignment_rate2) * 14000 / float(10**6))
print "\n\n"


print "/---- Improvement one ext. to two"
print "Absolute improvement in density: %.1f" % (Nelg_extended_two-Nelg_extended_single)
print "Fractional improvement %.3f" % ((Nelg_extended_two-Nelg_extended_single)/float(Nelg_extended_single))