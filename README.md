# physics-ml
Here I try to use ML in Physics Simulations

## Fluid simulation
First, we re-implement javascript 
[code](https://github.com/matthias-research/pages/blob/master/tenMinutePhysics/17-fluidSim.html) 
from TenMinutesPhysics on python with `fluid_sim.py`

This script will simulate simple fluid dynamics and shows animation of pressure and smoke.
For example, running script with arguments:
```bash
python .\fluid_sim.py --obstacle --pipes 2 --animation ./result.gif --frames 300
```
will generate GIF like this
<p align="center">
    <img src=".\media\simulation.gif" width="60%" height="auto" align="center"/> 
</p>

Here we add randomly moving obstacle and two pipes with horizontal and vertical flows 
which could be removed from argument line parameters. 

*There are also some unused features for now like several obstacles and applying random force.
Maybe it will be finished later.*

Now let us investigate states of simulation which be used to train neural network(s).