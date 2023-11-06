import sys
sys.path.append("/home/ravshan/Seafile/research/codes/NKnorms/")
import nkpack as nk
import pytest
import numpy as np

@pytest.fixture
def blueprint():
    output = {'param1': (1,2), 'param2': ('A','B'), 'param3': (789,)}
    return output

def test_bin2dec():
    assert nk.bin2dec(np.array([0,1,1,1,1,0,0,1,1,0])) == 486

def test_variate(blueprint):
    assert list(nk.variate(blueprint)) == [
        {'param1': 1, 'param2': 'A', 'param3': 789},
        {'param1': 1, 'param2': 'B', 'param3': 789},
        {'param1': 2, 'param2': 'A', 'param3': 789},
        {'param1': 2, 'param2': 'B', 'param3': 789}]

def test_generate_network():
    check = nk.generate_network(4, 2, 1.0, "star") == \
            np.array([[0.,1.,1.,1.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.]]).all()
    print(nk.generate_network(4, 2, 1.0, "star"))
    assert check is True