import numpy as np
from scipy.optimize import fsolve
import scipy.special as sp

def natural_freq(n, S1, k):
    # Define the determinant function that we want to find the roots of
    def determinant(omega):
        Jn_omega = sp.jv(n, omega)
        Jn_n1_omega = sp.jv(n-1, omega)
        Yn_omega = sp.yv(n, omega)
        Yn_n1_omega = sp.yv(n-1, omega)

        Jn_omega_k = sp.jv(n, omega * k)
        Jn_n1_omega_k = sp.jv(n-1, omega * k)
        Yn_omega_k = sp.yv(n, omega * k)
        Yn_n1_omega_k = sp.yv(n-1, omega * k)
        
        M11 = Jn_omega * S1 - omega * Jn_n1_omega
        M12 = Yn_omega * S1 - omega * Yn_n1_omega
        M21 = Jn_omega_k * S1 - omega * k * Jn_n1_omega_k
        M22 = Yn_omega_k * S1 - omega * k * Yn_n1_omega_k

        det = M11 * M22 - M12 * M21
        return det
    # Find the roots numerically
    initial_guess = [i * 0.25 for i in range(1, 480)] # [1.01178, 3.33527, 6.37694, 9.48676, 12.61272, 15.74499, 18.88038, 22.01756, 25.15584, 28.29486]

    ans = []

    for guess in initial_guess:
    	solutions = fsolve(determinant, guess)

    	# Filter out duplicate or close solutions and keep unique ones
    	unique_solutions = np.unique(np.round(solutions, decimals=5))
    	if unique_solutions not in ans and abs(determinant(unique_solutions)) < 1:
    		ans.append(unique_solutions[0])
    	# print(unique_solutions)
    	# print(determinant(unique_solutions))

    return ans
