import numpy as np
from benchmarker import benchmark

class Meeting:
    def __init__(self):
        self.proposals = None

    def decide(self, vec):
        self.proposals = vec

    def reset(self):
        self.proposals = None

@benchmark(times=1000)
def using_same_instance(t_):
    meeting = Meeting()
    for t in range(t_):
        meeting.decide(np.random.choice(2,2000))
        meeting.reset()

@benchmark(times=1000)
def using_new_instances(t_):
    for t in range(t_):
        meeting = Meeting()
        meeting.decide(np.random.choice(100,10))

if __name__=='__main__':
    T = 1000
    using_same_instance(T)
    using_new_instances(T)