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

Now let us investigate states of simulation which be used to train neural network(s) in `Experiments.ipynb`

### Intermediate Observations
Playing with the simulator I've found that resulting pressure is not following a Poison's Equation 
(see `Experiments.ipynb` for more details). Than I've implemented another simulator from this
[video](https://www.youtube.com/watch?v=63Ik8uZMxCQ) in `fluid_sim2.py` and I've showed (also in `Experiments.ipynb`)
that obtained data follows the equation:
$$\frac{\partial^2 p}{\partial x^2} + \frac{\partial^2p}{\partial y^2} = \frac{\rho}{\partial t} (\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} )$$
But in later frames the simulation were collapsed and the picture become white (the fields goes to infinity). *Also rewriting simulator in vector form were helpful end increase speed 100 times!* So, I implemented another simulator from  [numba](https://barbagroup.github.io/essential_skills_RRC/numba/4/) where solving step was more complicated. It had additional term when calculating $b[i,j]$. So the equation had  additipnal tern:
$$\frac{\partial^2 p}{\partial x^2} + \frac{\partial^2p}{\partial y^2} = \frac{\rho}{\partial t} (\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} ) - \rho ( (\frac{\partial u}{\partial x})^2 + 2\frac{\partial u}{\partial y}\frac{\partial v}{\partial x}+ (\frac{\partial v}{\partial y})^2)$$

This additional term is very small (see also in `Experiments.ipynb` part *Simulator 3*), but it was enough to collapse previous simulation. And now if you run animation cell in part *Simulator 3* you could see a plausible animation. 

Finally, we could generate data running
```
python fluid_sim3.py --dst new_sim_10k.npz --fast --frames 10000 --dt 0.01
```
and "validation" data (whis is part of train dataset)
```
python fluid_sim3.py --dst new_sim.npz --fast --frames 1000 --dt 0.01
```

## Training Neural Network
As simple model I take UNet architecture from this [paper](https://arxiv.org/pdf/2109.13076). Code of model, dataset, losses and Pytorch-Lighting module are living in `src` package in respectful files. 

To run test train use script `train.py`
```
python .\train.py --serialize_dir .\results\test --trn_path .\new_sim_10k.npz --val_path .\new_sim.npz --dt 0.01  --force
```

This a test run to ensure that code is working and losses are going down. To check how run was going you could see my wandb [project](https://wandb.ai/sdernal/physics-ml)

So, there are a lot of what to do:
- Write inferece script and look on predictions
- Normalize data
- Add images of pressure field (and etc.) to wandb  during training
- Generate normal train and validation datasets
- Remove some hardcode
- Build a new **NEURAL** simulator
