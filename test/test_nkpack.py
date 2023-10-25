import pytest
import numpy as np
import sys
sys.path.append("/home/ravshan/Seafile/research/codes/NKnorms/")
import nkpack as nk

@pytest.fixture
def input_value():
    input = np.array([0,1,1,1,1,0,0,1,1,0])
    return input

def test_bin2dec(input_value):
    assert nk.bin2dec(input_value) == 486