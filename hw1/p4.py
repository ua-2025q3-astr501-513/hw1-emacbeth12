#!/usr/bin/env python3
#
# Please look for "TODO" in the comments, which indicate where you
# need to write your code.
#
# Part 4: Solve the Coupled Simple Harmonic Oscillator Problem (1 point)
#
# * Objective:
#   Take the coupled harmonic oscillator problem we solved in class
#   and rewrite it using a well-structured Python class.
# * Details:
#   The description of the problem and the solution template can be
#   found in `hw1/p4.py`.
#
# From lecture `02w`, we solve systems of coupled harmonic oscillators
# semi-analytically by numerically solving eigenvalue problems.
# However, the code structure was not very clean, making the code hard
# to reuse.
# Although numerical analysis in general does not require
# object-oriented programming, it is sometime useful to package
# stateful caluation into classes.
# For this assignment, we will provide a template class.
# Your responsibility to implement the methods in the class.


import numpy as np

class CoupledOscillators:
    """A class to model a system of coupled harmonic oscillators.

    Attributes:
        Omega (np.ndarray): array of angular frequencies of the normal modes.
        V     (np.ndarray): matrix of eigenvectors representing normal modes.
        M0    (np.ndarray): initial amplitudes of the normal modes.

    """

    def __init__(self, X0=[-0.5, 0, 0.5], m=1.0, k=1.0):
        """Initialize the coupled harmonic oscillator system.

        Args:
            X0 (list or np.ndarray): initial displacements of the oscillators.
            m  (float):              mass of each oscillator (assumed identical for all oscillators).
            k  (float):              spring constant (assumed identical for all springs).

        """
        # TODO: Construct the stiffness matrix K
        # construct the stiffness matrix K generally
        K = np.zeros((len(X0), len(X0)))
        for i in range(K.shape[0]):
            for j in range(K.shape[1]):
                if i == j:
                    K[i, j] = 2*k
                else:
                    K[i,j] = -1*k
    
        # TODO: Solve the eigenvalue problem for K to find normal modes
        # divide by m to solve for w^2 as eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(K / m)   
        
        # TODO: Store angular frequencies and eigenvectors
        # take square root so values are omega
        self.Omega = np.sqrt(eigenvalues)
        self.V = eigenvectors

        # TODO: Compute initial modal amplitudes M0 (normal mode decomposition)
        # move to mode space 
        self.M0 = self.V.T @ X0

    def __call__(self, t):
        """Calculate the displacements of the oscillators at time t.

        Args:
            t (float): time at which to compute the displacements.

        Returns:
            np.ndarray: displacements of the oscillators at time t.

        """
        # TODO: Reconstruct the displacements from normal modes
        M = self.M0 * np.cos(self.Omega * t)
        # move back into position space
        return self.V @ M

if __name__ == "__main__":

    # Initialize the coupled oscillator system with default parameters
    co = CoupledOscillators()

    # Print displacements of the oscillators at each time step
    print("Time(s)  Displacements")
    print("----------------------")
    for t in np.linspace(0, 10, num=101):
        X = co(t)             # compute displacements at time t
        print(f"{t:.2f}", X)  # print values for reference
