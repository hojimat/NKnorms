import sys
sys.path.append("/home/ravshan/Seafile/research/codes/NKnorms/")
import models

# set params
ALT=1
PROP=1
COMP=1
COORD=1 # 0: decent, 1: lateral, 2: hierarchical

# set up the world
nature = models.Nature(
    p=5,n=4,kcs=(3,0,0),t=500,rho=0.9,nsoc=4,deg=2,net=3,xi=1.0,tm=50,coord=COORD,
    apc=(ALT,PROP,COMP),wf=1.0,goals=(1.0,1.0),w=1.0,normalize=True,precompute=True)
nature.initialize()
agent = nature.agents[0]
meeting = nature.organization.meeting(n=4,p=5,alt=ALT, prop=PROP, comp=COMP, nature=nature)

# check different alt-prop combinations
print(f"Initial: {agent.current_state[:4]}")
agent.screen(alt=ALT,prop=PROP)


# check different comp combinations
meeting.run()
print(agent.current_state)
print(meeting.outcome)