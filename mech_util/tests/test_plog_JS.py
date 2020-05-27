"""Test the output of the irrev_mech script"""
import os
import random
import numpy as np
from ..un_plog import convert_mech_un_plog
from ..mech_interpret import read_mech


def test_convert_mech():
    pth = os.path.dirname(os.path.realpath(__file__))
    P = 10 ** (-2 + 4*random.random())  # Pressure for the test.
    P = '{:.2g} atm'.format(P)
    print('Testing at {:}'.format(P))
    args = {'mech_name': os.path.join(pth, 'mechanisms', 'butane_100spec.inp'),
            'therm_name': os.path.join(pth, 'mechanisms', 'butane_100spec_therm.dat'),
            'pressure': P, 'temp_range': [300.0, 5000.0]}
    convert_mech_un_plog(**args, permissive=True, plot=False)

    head, tail = os.path.split(args['mech_name'])
    new_mech = os.path.join(head, 'un_plog_' + tail)
    all_removed(new_mech, args['therm_name'])

    compare_rate(**args)
    # blessed = open(os.path.join(pth, 'mech_blessed.txt'), 'r').read()
    # output = open(os.path.join(pth, 'mech_output.txt'), 'r').read()
    # assert output == blessed
    # if os.path.exists(os.path.join(pth, 'mech_output.txt')):
    #     os.remove(os.path.join(pth, 'mech_output.txt'))


def all_removed(mech, therm):
    " Check that all PLOG rates have been removed. "
    _, _, reacs = read_mech(mech, therm)
    for reac in reacs:
        assert reac.plog == False

def compare_rate(mech_name, therm_name, pressure, temp_range):
    """ Compare simulation between original and modified version.

    Check in in cantera implementation to be safe
    """
    # TODO: Reorganize so that the loop calls a helper function. Also, run that
    # helper function with a condition that I know makes sense, like one from
    # manual_test.
    import cantera
    import cantera.ck2cti

    pressure = float(pressure.split()[0]) * 101325 # Convert to Pa
    pth = os.path.dirname(os.path.realpath(__file__))
    wdir = os.path.join(pth, 'mechanisms')

    parser = cantera.ck2cti.Parser()
    parser.convertMech(mech_name, therm_name,
                       outName=os.path.join(wdir, 'original.cti'))
    head, tail = os.path.split(mech_name)
    new_mech = os.path.join(head, 'un_plog_' + tail)
    parser.convertMech(new_mech, therm_name,
                       outName=os.path.join(wdir, 'un_plog.cti'))

    gas1 = cantera.Solution(os.path.join(wdir, 'original.cti'))
    gas2 = cantera.Solution(os.path.join(wdir, 'un_plog.cti'))

    success_counter = 0
    for i in range(10):  # Test with 10 random parameters
        print('\nIteration {}'.format(i))
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
        time = np.linspace(0, endtime, 500)
        print('End Time: {:.4g}'.format(time[-1]))

        try:
            for sol in (gas1, gas2):
                sol.TPX = T, pressure, mix
                reac = cantera.ConstPressureReactor(sol, energy='on')
                sim = cantera.ReactorNet([reac])

                T_hist = []
                X_hist = []
                row = [None]*4
                for t in time:
                    sim.advance(t)
                    T_hist.append(sol.T)
                    X_hist.append(sol.X)
                T_hist = np.array(T_hist)
                X_hist = np.array(X_hist)

                T_outputs.append(T_hist)
                X_outputs.append(X_hist)
        except cantera.CanteraError:
            print('Cantera Error. Moving onto next case')
            continue

        if not np.allclose(T_outputs[1], T_outputs[0], rtol=5e-2):
            print('\nTemperature histories do not match:')
            print_time_history_comparison(T_outputs[0], T_outputs[1])
            assert False

        if not np.allclose(X_outputs[1], X_outputs[0], rtol=5e-2, atol=1e-5):
            for ind in range(len(X_outputs[1][0, :])):
                if not np.allclose(X_outputs[1][:, ind], X_outputs[0][:, ind],
                                   rtol=5e-2, atol=1e-5):
                    print('\nTime history for ' + gas1.species(ind).name +
                          ' mole fraction does not match:')
                    print_time_history_comparison(X_outputs[0][:, ind],
                                                  X_outputs[1][:, ind])
            assert False
        success_counter += 1

    assert success_counter >= 5


def print_time_history_comparison(arr1, arr2):
    " Print a user-friendly comparsion of time histories. "
    assert len(arr1) == len(arr2)
    print('item 1     item 2     Percent difference')
    print('------     ------     ----------------')
    skip_count = 0
    for val1, val2 in zip(arr1, arr2):
        if skip_count == 10:
            skip_count = 0
        else:
            skip_count += 1

        row = '{:<11.4g}{:<11.4g}{:<11.3g}'.format(val1, val2, 100*(val2/val1 - 1))
        if not np.isclose(val1, val2, rtol=5e-2, atol=1e-5):
            print('\x1b[1;31;40m' + row + '\x1b[0m')
        elif skip_count == 0:
            print(row)
