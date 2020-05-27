"""Replaces PLOG reactions with Arrhenius reactions at a single pressure.
"""

# Python 2 compatibility
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

# Standard libraries
import copy
import math
import os
import warnings
from multiprocessing import Pool
from itertools import repeat
import bisect
import matplotlib.pyplot as plt

try:
    import numpy as np
except ImportError:
    print('Error: NumPy must be installed.')
    raise
try:
    from scipy.optimize import leastsq
    from scipy.optimize import curve_fit
except ImportError:
    print('Error: SciPy must be installed.')
    raise

# Local imports
from . import chem_utilities as chem
from . import mech_interpret as mech
from .write_mech import write_mech


def calc_rate_coeff(p, T):
    """Calculate Arrhenius reaction rate coefficient."""
    A, b, E = p
    k = A * np.exp(b * np.log(T) - (E.to('K').m / T))
    return k


def residuals(p, y, x):
    """Residual for calculating rate coefficient."""
    A, b, E = p
    err = y - calc_rate_coeff(p, x)
    return err


def double_arrhenius(invT, logA1, b1, E1, logA2, b2, E2):
    " Natural log of double Arrhenius rate."
    k1 = np.exp(single_arrhenius(invT, logA1, b1, E1))
    k2 = np.exp(single_arrhenius(invT, logA2, b2, E2))
    return np.log(k1 + k2)


def double_arrhenius_negA(invT, logA1, b1, E1, logA2, b2, E2):
    """ Natural log of double Arrhenius rate.
    Assume second A-factor is negative. """
    k1 = np.exp(single_arrhenius(invT, logA1, b1, E1))
    k2 = np.exp(single_arrhenius(invT, logA2, b2, E2))
    return np.log(k1 - k2)


def single_arrhenius(invT, logA, b, E):
    """ Returns the natural log of reaction rate

    k = A * T**b * exp(-E / T)
    logk = logA + b*logT - E / T
    logk = logA - b*loginvT - E * invT
    """
    return logA - b * np.log(invT) - E * invT


def refit_reaction(reaction, pressure, temp_range, permissive=False):
    """ Create a single Arrhenius function for the PLOG reaction at the given
    pressure and temperature range [K].

    """
    logk, T, p0 = interpolate_k(reaction, pressure, temp_range, permissive)

    bounds1 = ((-7e2, -np.inf, -np.inf), (7e2, np.inf, np.inf))
    bounds2 = ((-7e2, -np.inf, -np.inf, -7e2, -np.inf, -np.inf),
               (7e2, np.inf, np.inf, 7e2, np.inf, np.inf))
    # Fit a new Arrhenius function
    invT = 1 / T
    if len(p0) == 3:
        p0[2] = p0[2].to('K').m # Use activation energy in Kelvin
        if np.sign(p0[0]) == -1:
            raise ValueError('Single Arrhenius reaction rate must be positive')
        p0[0] = np.log(p0[0])
        try:
            popt, pcov = curve_fit(single_arrhenius, invT, logk, p0, bounds=bounds1)
        except RuntimeError:
            print('Error in ' + str(reaction))
            raise
        logA, b, E = popt
    elif len(p0) == 6:
        p0[2] = p0[2].to('K').m # Use activation energy in Kelvin
        p0[5] = p0[5].to('K').m # Use activation energy in Kelvin
        s_A = (np.sign(p0[0]), np.sign(p0[3]))
        p0[0] = np.log(abs(p0[0]))
        p0[3] = np.log(abs(p0[3]))
        if s_A == (1, 1):
            func = double_arrhenius
        elif s_A == (1, -1):
            func = double_arrhenius_negA
        elif s_A == (-1, 1):
            p0 = p0[3:] + p0[:3]
            func = double_arrhenius_negA
        else:
            raise ValueError('Double Arrhenius reaction rate must have one positive component')

        try:
            popt, pcov = curve_fit(func, invT, logk, p0, bounds=bounds2)
            logA, b, E, logA2, b2, E2 = popt
            if func == double_arrhenius_negA:
                A2 = -1 * np.exp(logA2)
            else:
                A2 = np.exp(logA2)
        except RuntimeError:
            try:
                popt, pcov = curve_fit(single_arrhenius, invT, logk,
                                       (10, 0, 50), bounds=bounds1)
                logA, b, E = popt
            except RuntimeError:
                print('Error in ' + str(reaction))
                raise

    # Save the results and return the reaction
    reaction1 = copy.deepcopy(reaction)
    reaction1.plog = False
    reaction1.A = np.exp(logA)
    reaction1.b = b
    reaction1.E = chem.Q_(E, 'K')

    if len(popt) == 6:
        reaction2 = copy.deepcopy(reaction)
        reaction2.plog = False
        reaction2.A = A2
        reaction2.b = b2
        reaction2.E = chem.Q_(E2, 'K')
        reaction2.dup = True
        reaction1.dup = True
        return reaction1, reaction2
    else:
        return reaction1, None


def interpolate_k(reaction, pressure, temp_range, permissive):
    """
    Calculate reaction rate as a function of T for PLOG reactions.

    Parameters
    ----------
    reaction : ReacInfo
        Reaction object
    pressure : pint quantity
        Pressure to evaluate reaction rate.
    temp_range : list
        Temperature range [K] over which to evaluate the reaction rate.
    permissive : bool, int
        If pressure is outside the PLOG range:
            0, False: raise error
            1, True: print warning
            2: ignore

    Returns
    -------
    logk : array
        Natural log of the reaction rate as a function of temperature.
    k : array
        Natural log of the reaction rate as a function of temperature
    T : array
        Temperature - linearly spaced in 1/T
    p0 : list
        Initial guess for the fit. 3 parameters for a single Arrhenius, 6
        parameters for a double Arrhenius.

    """
    interp_flag = False

    pressures = [x[0] for x in reaction.plog_par]
    pressures.sort()

    # If outside the range, use the max or min plog parameters
    if pressures[-1] < pressure or pressures[0] > pressure:
        msg = ('Reaction {:}. The given pressure of '
               '{:.2g} atm is outside the given PLOG pressures, '
               '{:.2g} - {:.2g} atm.'.format(
                  str(reaction), pressure.to('atm').m,
                  pressures[0].to('atm').m, pressures[-1].to('atm').m))
        if permissive == 2:
            pass
        elif permissive:
            print('WARNING: ' + msg)
        else:
            raise ValueError(msg)

    index = bisect.bisect(pressures, pressure)
    if index == 0:
        pressure_1 = pressures[0]
        pressure_2 = 0
    elif index == len(pressures):
        pressure_1 = pressures[-1]
        pressure_2 = 0
    else:
        pressure_1 = pressures[index - 1]
        pressure_2 = pressures[index]
        interp_flag = True

    if pressure_1 == pressure or pressure_2 == pressure:
        pressure_1 = pressure

    T = 1/np.linspace(1/temp_range[0], 1/temp_range[1], 200)

    k1 = np.zeros(200)
    params1 = []
    for par in reaction.plog_par:
        if par[0] == pressure_1:
            k1 += calc_rate_coeff((par[1], par[2], par[3]), T)
            params1.append([par[1], par[2], par[3]])

    # Initial guess for new fit based on parameters at pressure_1
    p0 = params1[0]
    try:
        p0.extend(params1[1])
    except IndexError:
        pass

    if interp_flag:
        k2 = np.zeros(200)
        params2 = []
        for par in reaction.plog_par:
            if par[0] == pressure_2:
                k2 += calc_rate_coeff((par[1], par[2], par[3]), T)
                params2.append([par[1], par[2], par[3]])
        p0_new = params2[0]
        p0_mod = [(p0[0]*p0_new[0])**0.5, np.mean((p0[1], p0_new[1])),
                  (p0[2] + p0_new[2])/2]
        try:
            p0_new = params2[1]
            p0_mod.extend([(p0[3]*p0_new[0])**0.5, np.mean((p0[4], p0_new[1])),
                           (p0[5] + p0_new[2])/2])
        except IndexError:
            pass

        # Linear interpolation in log-space
        logk1 = np.log(k1)
        logk2 = np.log(k2)
        logp1 = np.log(pressure_1.to('Pa').m)
        logp2 = np.log(pressure_2.to('Pa').m)
        logp = np.log(pressure.to('Pa').m)
        log_k_combined = logk1 + (logp - logp1)*(logk2 - logk1)/(logp2 - logp1)
        return log_k_combined, T, p0

    else:

        return np.log(k1), T, p0


def plot_fit(r_orig, r1_mod, r2_mod, mech_name, pressure, temp_range,
             permissive=False, plot=True):
    """ Calculate fit error and plot the new and original reactions against
    each other for the given pressure and temperature range. For a duplicate
    reaction, r1_mod and r2_mod are the duplicate rates.

    If RMS error is too high:
        permissive: print warning message and plot the fit
        not permissive: raise error.
    """
    T = 1/np.linspace(1/temp_range[0], 1/temp_range[1], 200)
    logk_0, T0, _ = interpolate_k(r_orig, pressure, temp_range, 2)
    assert np.allclose(T, T0)
    k_0 = np.exp(logk_0)

    if r1_mod is not None:
        k_mod = calc_rate_coeff((r1_mod.A, r1_mod.b, r1_mod.E), T)
    if r2_mod is not None:
        k_mod += calc_rate_coeff((r2_mod.A, r2_mod.b, r2_mod.E), T)

    # Error checking
    if r1_mod is not None:
        error = np.sqrt(np.mean((k_mod / k_0 - 1)**2))
        if error > 0.1:
            msg = 'Error in fit is too high: {:.2g} for {:}'.format(error,
                                                                    str(r_orig))
            if permissive:
                print('WARNING: ' + msg)
                plot = True
            else:
                raise ValueError(msg)

    if plot:
        fig, axes = plt.subplots(2, 1, sharex=True)
        invT = 1000 / T
        ax = axes[0]  # Top figure
        title = 'Fit for {:} at {:.2f} atm'.format(str(r_orig), pressure.to('atm').m)
        ax.set_title(title)
        ax.plot(invT, k_0, 'b-', label='original')

        if r1_mod is not None:
            ax.plot(invT, k_mod, 'r--', label='modified')
        ax.set_yscale('log')
        ax.legend()
        ax.set_ylabel('Reaction rate (k)')

        if r1_mod is not None:
            ax = axes[1]  # Bottom figure
            ax.plot(invT, 100 * (k_mod / k_0 - 1), 'k-')
            ax.set_ylabel('% error')
            ax.set_xlabel('1000 / T $[K^{-1}]$')

        fig.savefig(os.path.join(os.path.dirname(mech_name), str(r_orig) + '.png'))
        plt.close(fig)



def convert_mech_un_plog(mech_name, therm_name=None, pressure=1.0,
                         temp_range=[300.,5000.], permissive=False, plot=True):
    """


    Parameters
    ----------
    mech_name : string
        Reaction mechanism filename (e.g. 'mech.dat')
    therm_name : string, optional
        Thermodynamic database filename (e.g. 'therm.dat') or None
    pressure : TYPE, optional
        Pressure with units for hard-coding PLOG reactions.
        If no units specified, atm assumed.
    temp_range : list, optional
        Temperature range for PLOG fitting. The default is [300.,5000.].
    permissive : bool, optional
        Allow larger uncertainties with warnings. The default is False.
    plot : TYPE, optional
        Plot all new reaction fits. The default is True.

    Returns
    -------
    None.

    """
    pressure = chem.Q_(pressure)
    try:
        pressure.ito('pascal')
    except DimensionalityError:
        warnings.warn(
            'No units specified, or units incompatible with pressure. ',
            'Assuming atm.'
            )
        pressure = (pressure.magnitude * chem.units.atm).to('pascal')

    # interpret reaction mechanism file
    [elems, specs, reacs] = mech.read_mech(mech_name, therm_name)
    # I don't think the thermo file will be necessary

    # Delete old figures
    dirname = os.path.dirname(mech_name)
    for file in os.listdir(dirname):
        if '=' in file and '.png' in file:
            os.remove(os.path.join(dirname, file))

    # Convert the reactions
    converted_reacs = []
    for reac in reacs:
        if reac.plog:
            reac.E = reac.E.to('K').m
            try:
                reac1, reac2 = refit_reaction(reac, pressure, temp_range,
                                              permissive)
            except RuntimeError as e:
                if 'Optimal parameters' in str(e):
                    print('Error for {}. Plotting'.format(str(reac)))
                    plot_fit(reac, None, None, mech_name, pressure, temp_range,
                             False, True)
                raise
            converted_reacs.append(reac1)
            if reac2 is not None:
                converted_reacs.append(reac2)
            plot_fit(reac, reac1, reac2, mech_name, pressure, temp_range,
                     permissive, plot)
        else:
            converted_reacs.append(reac)

    # write new reaction list to new file
    head, tail = os.path.split(mech_name)
    output_file = os.path.join(head, 'un_plog_' + tail)
    header = ('! This mechanism was created by un_plog at a pressure of '
              '{:.4f} atm.\n'.format(pressure.to('atm').m))

    # write new reaction list to new file
    write_mech(output_file, elems, specs, converted_reacs, header)
    return
