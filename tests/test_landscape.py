import sys
sys.path.append("/home/ravshan/Seafile/research/codes/NKnorms/")
import nkpack as nk
import pytest
import numpy as np
import models

@pytest.fixture
def setup():
    landscape = models.Landscape(5,4,   3,0,0,   0.0,True,True)
    return {'landscape': landscape}

def test_landscape(setup):
    setup['landscape']._generate_interaction_matrix()
    inmat = setup['landscape'].interaction_matrix
    assert all(inmat.sum(axis=0) == inmat.sum(axis=1))