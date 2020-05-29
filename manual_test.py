# -*- coding: utf-8 -*-
"""
Manually test the code to run specific cases.

This code should be run by a human, not through pytest. It will create a figure
comparing the constant-pressure "ignition delay" between the original model
and the model with PLOG removed.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import cantera
import cantera.ck2cti
from mech_util.un_plog import convert_mech_un_plog

def comparison_plot(orig_mech, new_mech, therm, P, phis, T, fuel):
    parser = cantera.ck2cti.Parser()
    parser.convertMech(orig_mech, therm, permissive=True)
    parser.convertMech(new_mech, therm, permissive=True)
    orig_cti = orig_mech[:-3] + 'cti'
    new_cti = new_mech[:-3] + 'cti'


    orig_gas = cantera.Solution(orig_cti)
    new_gas = cantera.Solution(new_cti)

    fig, axes = plt.subplots(2, len(phis), sharey='row', sharex='col')
    fig.suptitle('Pressure is {:.2f} atm'.format(P))
    for phi, axs in zip(phis, axes.T):
        orig_T_t = simulate(orig_gas, phi, fuel, P, T)
        new_T_t = simulate(new_gas, phi, fuel, P, T)
        axs[0].plot(1000 * orig_T_t[:, 0], orig_T_t[:, 1], 'b-', label='original')
        axs[0].plot(1000 * new_T_t[:, 0], new_T_t[:, 1], 'r--', label='modified')
        ind = np.argwhere(orig_T_t[:,1] > orig_T_t[-1, 1]*0.95)[0]
        axs[0].set_xlim([0, orig_T_t[ind, 0] * 1100])
        axs[0].text(0.5, 0.9, r'$\phi$ = {:.2f}'.format(phi),
                transform=axs[0].transAxes)
        axs[0].legend(loc='center left')

        axs[1].plot(1000 * orig_T_t[:, 0], orig_T_t[:, 1] - new_T_t[:, 1])
        axs[1].set_xlabel('Time [ms]')
    axes[0, 0].set_ylabel('Temperature [K]')
    axes[1, 0].set_ylabel('Change in Temperature [K]')

    fig.savefig(os.path.join(os.path.dirname(orig_mech), 'Ignition.png'))
    plt.close(fig)



def simulate(gas, phi, fuel, P, T):
    gas.set_equivalence_ratio(phi, fuel, 'O2:0.21, N2: 0.79')
    gas.TP = (T, P * 101325)
    reac = cantera.ConstPressureReactor(gas)
    sim = cantera.ReactorNet([reac])
    T_t = []
    t = np.linspace(0, 5, 500)
    for time in t:
        sim.advance(time)
        T_t.append([time, gas.T])

    return np.array(T_t)


if __name__ == '__main__':
    ###############################################################
    # User-defined parameters
    test_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            'mech_util', 'tests', 'mechanisms')
    mech_name = 'chem.inp'  # Assumed to be within the test_dir
    therm_name = 'therm.dat'
    pressure = 10  # Pressure in atm
    fuel = 'H2'
    T = 1200  # Temperature in Kelvin
    phis = (0.75, 1.0)
    ##########################

    mech_name = os.path.join(test_dir, mech_name)
    therm_name = os.path.join(test_dir, therm_name)
    convert_mech_un_plog(mech_name, therm_name, pressure, permissive=True,
                         plot=False)
    head, tail = os.path.split(mech_name)
    new_mech = os.path.join(head, 'un_plog_' + tail)
    comparison_plot(mech_name, new_mech, therm_name, pressure, phis, T, fuel)

