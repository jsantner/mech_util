"""Test the output of the irrev_mech script"""
import os
import random
import numpy as np
from ..un_plog import convert_mech_un_plog


def test_convert_mech():
    pth = os.path.dirname(os.path.realpath(__file__))
    P = 10 ** (-2 + 4*random.random())  # Pressure for the test.
    P = '{:.2g} atm'.format(P)
    print('Testing at {:}'.format(P))
    args = {'mech_name': os.path.join(pth, 'mechanisms', 'butane_100spec.inp'),
            'therm_name': os.path.join(pth, 'mechanisms', 'butane_100spec_therm.dat'),
            'pressure': P, 'temp_range': [300.0, 5000.0]}
    convert_mech_un_plog(**args, permissive=True, plot=False)
    compare_rate(5, **args)
    # blessed = open(os.path.join(pth, 'mech_blessed.txt'), 'r').read()
    # output = open(os.path.join(pth, 'mech_output.txt'), 'r').read()
    # assert output == blessed
    # if os.path.exists(os.path.join(pth, 'mech_output.txt')):
    #     os.remove(os.path.join(pth, 'mech_output.txt'))

def compare_rate(rxn_num, mech_name, therm_name, pressure, temp_range):
    """ Compare rate # rxn_num in the mechanism to its modified version.

    Check in in cantera implementation to be safe
    """
    # TODO: Reorganize so that the loop calls a helper function. Also, run that
    # helper function with a condition that I know makes sense, like one from
    # manual_test.
    import cantera
    import cantera.ck2cti

    pth = os.path.dirname(os.path.realpath(__file__))
    parser = cantera.ck2cti.Parser()
    parser.convertMech(mech_name, therm_name,
                       outName=os.path.join(pth, 'mechanisms', 'original.cti'))
    head, tail = os.path.split(mech_name)
    new_mech = os.path.join(head, 'un_plog_' + tail)
    parser.convertMech(new_mech, therm_name,
                       outName=os.path.join(pth, 'mechanisms', 'un_plog.cti'))

    gas1 = cantera.Solution('original.cti')
    gas2 = cantera.Solution('un_plog.cti')

    success_counter = 0
    for i in range(10):  # Test with 10 random parameters
        print('Iteration {}'.format(i))
        # x1 = random.random()
        # mix = {'R1A': x1, 'R1B': 1 - x1, 'H': 0.01*random.random(),
        #        'R5': 0.01*random.random(), 'P1': 0.01*random.random()}
        phi = 0.1 + random.random() * 2
        mix = {'C4H10': phi*0.21/6.5, 'O2': 0.21, 'N2': 0.79}
        T = temp_range[0] + random.random() * (temp_range[1] / 2 - temp_range[0])
        print('Mixture: ' + str(mix))
        print('Temperature: {:.2f} K'.format(T))
        T_outputs = []
        X_outputs = []
        t_outputs = []

        # Find appropriate end time
        gas1.TPX = T, pressure, mix
        reac = cantera.ConstPressureReactor(gas1, energy='on')
        sim = cantera.ReactorNet([reac])
        try:
            sim.advance_to_steady_state()
        except cantera.CanteraError:
            print('Cantera Error solving for steady state. Moving on.')
            continue
        endtime = sim.time / 5
        if endtime < 1e-6:
            print('Too fast. Skipping condition')
            continue
        time = np.linspace(0, endtime)
        print('End Time: {:.4g}'.format(time[-1]))

        try:
            for sol in (gas1, gas2):
                sol.TPX = T, pressure, mix
                reac = cantera.ConstPressureReactor(sol, energy='on')
                sim = cantera.ReactorNet([reac])

                T_hist = []
                X_hist = []
                t_hist = []
                row = [None]*4
                for t in time:
                    sim.advance(t)
                    T_hist.append(sol.T)
                    X_hist.append(sol.X)
                T_hist = np.array(T_hist)
                X_hist = np.array(X_hist)
                t_hist = np.array(t_hist)

                T_outputs.append(T_hist)
                X_outputs.append(X_hist)
                t_outputs.append(t_hist)
        except cantera.CanteraError:
            print('Cantera Error. Moving onto next case')
            continue
        if not np.allclose(T_outputs[1], T_outputs[0], rtol=5e-2):
            print('Printing T histories')
            print(T_outputs[0], T_outputs[1], T_outputs[1] / T_outputs[0])
            assert False
        # t_tol = max(t_outputs[1])*1e-5
        # assert np.allclose(t_outputs[1], t_outputs[0], rtol=1e-2, atol=t_tol
        if not np.allclose(X_outputs[1], X_outputs[0], rtol=5e-2, atol=1e-4):
            for ind in [0, 2, 3, 4, 5]:
                if not np.allclose(X_outputs[1][:, ind], X_outputs[0][:, ind],
                                   rtol=5e-2, atol=1e-4):
                    print('Printing mole fraction of ' + gas1.species(ind).name)
                    print(X_outputs[0][:, ind], X_outputs[1][:, ind],
                          X_outputs[1][:, ind] / X_outputs[0][:, ind])
            assert False
        success_counter += 1

    assert success_counter >= 5
