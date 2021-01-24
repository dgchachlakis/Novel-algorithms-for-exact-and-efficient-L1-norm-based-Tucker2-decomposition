import numpy as np
import algorithms as l1tucker2
import matplotlib.pyplot as plt
from utils import *
# Form 3-way tensor of 10 10-by-10 matrix measurements. 
D, M, N = 10, 10, 10
tensor = np.random.randn(D, M, N)
# Solve rank-1 L1-Tucker2 exactly, by exhaustive search. 
uopt_exact, vopt_exact, bopt_exact, metopt_exact, numOfCandidates0 = l1tucker2.exact(tensor)
# Approximate the solution to rank-1 L1-Tucker2 by bit-flipping iterations without user-defined initialization
uest1, vest1, best1, evolution1 = l1tucker2.bitflipping(tensor)
# Approximate the solution to rank-1 L1-Tucker2 by bit-flipping iterations with user-defined initialization
b = np.sign(np.random.randn(N, ))
uest2, vest2, best2, evolution2 = l1tucker2.bitflipping(tensor, b)
# Plot the metric evolution (include the maximum attainable metric as a benchmark)
xax_max = np.max([len(evolution1), len(evolution2)])
plt.figure()
plt.plot([0, xax_max], [metopt_exact, metopt_exact], '-k', label = 'Exact')
plt.plot(evolution1, '-r', label = 'Bit-flipping')
plt.plot(evolution2, '-b', label = 'Bit-flipping')
plt.legend()
plt.xlabel('Iteration index')
plt.ylabel('rank-1 L1-Tucker2 metric')
plt.show()
# Extract multiple components by rank-1 BF-Tucker2 and defltation
number_of_components = 3
left_factor, right_factor = l1tucker2.bitflipping_deflation(tensor, number_of_components)