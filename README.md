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

### Training Neural Network
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

### Experiments results
During a dozens of training UNet I have made some improvements of ML System:
- Made UNet a little bit deeper
- Added callback which draws images of predicted pressure, real pressure, laplacian and network input
- Added RL Scheduler which helped a little bit with losses explosions
- Added Sigmoid activation to the end of UNet to ensure output ve in $[0,1]$
- Added Normalization of data (but simplier than in paper)

I also splitted my "dataset" to train and val just taking a right part for validation:
```python
import numpy as np

sample = np.load('new_sim_10k.npz')
np.savez('trn.npz', p=sample['p'][:9000], v=sample['v'][:9000], u=sample['u'][:9000]) 
np.savez('val.npz', p=sample['p'][9000:], v=sample['v'][9000:], u=sample['u'][9000:]) 
```

And makes a lot of runs, the final was called with
```
python .\train.py --serialize_dir .\results\unet5_to_show --trn_path .\trn.npz --val_path .\val.npz ^
    --dt 0.01 --force --accelerator gpu --patience 50 --max_epochs 200
```

This run could be found at [wandb](https://wandb.ai/sdernal/physics-ml/runs/r7etnbyf) along with the others.
I also used only Dirichlet and Inside losses with this run (Laplacian Loss working strange, but I also logged it). 
If you want add it for backprop, uncomment addition of it in `system.py -> PoissonSolver -> custom_loss`.

The results are not very satisfied: The Laplacian loss is Large(However other losses are quite low).
The picture of predicted pressure is similar to real one, but picture of Laplacian is different. 
There are some possible explanations:

- The Laplacian itself is quite small (~1e-9) end network could get such detail
- The dataset is poor and there are not much difference between frames, so the network just overfitted to medium.

To overcome this problem I need more diverse dataset, but I'm already tired with the 3 previous simulators. 
So I'm planning to write prediction script and finish with this.

To obtain data and checkpoint to inference you could download it from 
[google drive](https://drive.google.com/drive/folders/1A4I0--TqFGXfFi1teHxiW9D1kVKp35-Q?usp=sharing). 
So, the prediction script `neural_sim.py` with calling as
```commandline
python .\neural_sim.py --checkpoint .\unet5_to_show\epoch=134-step=38070.ckpt --trn_path .\trn.npz ^ 
--val_path val.npz --animation bruh4.gif
```
will generated following "simulation" (not valid after normalization fix, but with fix it worse):
<p align="center">
    <img src=".\media\predictions.gif" width="100%" height="auto" align="center"/> 
</p>

Here from left to write: pressure, vertical velocity, horizontal velocity, laplacian of pressure.

For now, it required datasets for calculating normalization constants. But latter it could be refactored.

As result, with used dataset it works not well and probably with more diverse data it will work better.
And there are also a lot of hardcode which should be reworked.

If you run prediction script with mode `--animate_dataset` you will obtain predictions on dataset and differences 
<p align="center">
    <img src=".\media\prediction_diff.gif" width="100%" height="auto" align="center"/> 
</p>
Here abs(p-pred).max() = 0.00043815374 in all frames, the pressure itself is in $[0,1]$

Difference on train while simulation is not converged
<p align="center">
    <img src=".\media\diff_on_train.gif" width="100%" height="auto" align="center"/> 
</p>

The absolute differences per frame are 
```
0.013391733
0.013391733
0.013391733
0.0012540221
0.0005912185
0.00044888258
0.00043940544
0.00043958426
0.00043970346
0.00043958426
0.00043940544
0.00043922663
0.00043910742
0.0004389882
0.0004388094
0.0004388094
0.00043863058
0.00043863058
0.00043851137
```
