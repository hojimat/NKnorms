import sys
sys.path.append("/home/ravshan/Seafile/research/codes/NKnorms/")
import nkpack as nk
import pytest
import numpy as np

@pytest.fixture
def blueprint():
    output = {'param1': (1,2), 'param2': ('A','B'), 'param3': (789,)}
    return output

@pytest.fixture
def bstrings():
    binary = np.array([1,0,0,0, 1,1,1,1, 1,1,0,1, 0,0,1,1])
    binary2 = np.array([1,0,0,0, 1,0,1,1, 1,0,0,1, 1,0,1,1]) # Hamming = 3
    length = 16
    decimal = 36819
    return {'bin': binary, 'bin2': binary2, 'len': length, 'dec': decimal}

@pytest.fixture
def allocations():
    network =  np.array([[0.,1.,1.,1.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.]])
    return {'net': network}

@pytest.fixture
def social():
    lookup = np.array([[1,0], [0,0], [0,0], [1,1]])
    return {'lookup': lookup}

def test_variate(blueprint):
    assert list(nk.variate(blueprint)) == [
        {'param1': 1, 'param2': 'A', 'param3': 789},
        {'param1': 1, 'param2': 'B', 'param3': 789},
        {'param1': 2, 'param2': 'A', 'param3': 789},
        {'param1': 2, 'param2': 'B', 'param3': 789}]

def test_generate_network(allocations):
    check = (nk.generate_network(4, 2, 1.0, "star") == allocations['net'])
    assert check.all()

def test_bin2dec(bstrings):
    assert nk.bin2dec(bstrings['bin']) == bstrings['dec']

def test_dec2bin(bstrings):
    is_equal = (nk.dec2bin(bstrings['dec'], bstrings['len']) == bstrings['bin'])
    assert is_equal.all()

def test_hamming_distance(bstrings):
    assert nk.hamming_distance(bstrings['bin'], bstrings['bin2']) == 3

def test_calculate_frequency(social):
    assert nk.calculate_frequency([1,1], social['lookup']) == 0.375
    assert nk.calculate_frequency([0,0], social['lookup']) == 0.625
    
    multiple = np.array([[1,1], [0,0]])
    freqs = np.apply_along_axis(lambda bstr: nk.calculate_frequency(bstr, social['lookup']), 1, multiple)
    assert (freqs == np.array([0.375, 0.625])).all()

@pytest.mark.parametrize(
    "goals,score", [
        ([1,1], -0.45),
        ([1,0.6], -0.25),
        ([0.5,0.5], 0),
        ([0.75,0.80], 0),
        ([0.75,0.85], -0.05),
        ])
def test_gp_score(goals, score):
    vals = np.array([0.75, 0.80])
    weights = np.array([1,1])
    assert nk.gp_score(vals, np.array(goals), weights) == pytest.approx(score)
