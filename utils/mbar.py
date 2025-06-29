import warnings

try:
    import fastMBAR
    fastmbar_available = True
except ImportError:
    import pymbar
    package_available = False
    warnings.warn("fastMBAR is not installed. Importing pymbar instead.", ImportWarning)

class MBARSolution (object):
    """
    MBARSolution is a class that encapsulates the solution of the MBAR problem.
    Based on the MBAR solution, it provides methods to compute the free energy differences,
    uncertainties, and the convergence diagnostics.
    """

    def __init__(self, mbar):
        """
        Initialize the MBARSolution object.
        """
        if fastmbar_available:
            # using fastMBAR
            self._free_energy = mbar.F

        else:
            # using pymbar 
            self._free_energy = mbar.F