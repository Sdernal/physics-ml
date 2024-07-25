from typing import Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from matplotlib import animation

from src.system import PoissonSolver
from src.dataset import PoissonDataset
from src.utils import get_normalization_values, calculate_g


class Scene:
    def __init__(self, system: PoissonSolver, data,
                 p_max = None, g_max = None, h=1., dt=0.01, rho=1., nu=0.5):
        self.system = system
        self.data = data

        # starting point
        self.v = data['v'][0]  # np.ndarray
        self.u = data['u'][0]  # np.ndarray

        self.fig, (self.ax_p, self.ax_v, self.ax_u, self.ax_lp) = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
        self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        self.fig.set_size_inches(2.56, 0.64, True)
        self.ax_p.axis('off')
        self.ax_u.axis('off')
        self.ax_v.axis('off')
        self.ax_lp.axis('off')
        self.frame = []

        self.p_max = p_max
        self.g_max = g_max
        self.h = h
        self.dt = dt
        self.rho = rho
        self.nu = 0.5

    def draw(self, p, v, u):
        while len(self.frame):
            self.frame[-1].remove()
            self.frame.pop()

        self.frame.append(self.ax_p.imshow(np.rot90(p)))
        self.frame.append(self.ax_v.imshow(np.rot90(v)))
        self.frame.append(self.ax_u.imshow(np.rot90(u)))

        dx = dy = self.h
        laplacian_p = np.zeros((64, 64))
        laplacian_p[1:-1, 1:-1] = ((p[2:, 1:-1] - p[1:-1, 1:-1] * 2 + p[:-2, 1:-1]) / dx ** 2
                                    + (p[1:-1, 2:] - p[1:-1, 1:-1] * 2 + p[1:-1, :-2]) / dy ** 2)
        self.frame.append(self.ax_lp.imshow(np.rot90(laplacian_p)))

    def advect(self, p):
        """
        Advection from fluid_sim3.py
        :param p: pressure
        :return:
        """
        new_u = self.u.copy()
        new_v = self.v.copy()
        dx = dy = self.h
        dt = self.dt

        new_u[1:-1, 1:-1] = self.u[1:-1, 1:-1] - (
                self.u[1:-1, 1:-1] * (self.u[1:-1, 1:-1] - self.u[:-2, 1:-1])
                + self.v[1:-1, 1:-1] * (self.u[1:-1, 1:-1] - self.u[1:-1, :-2])
                + (p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * self.rho)
        ) * dt / dx + (
                self.u[2:, 1:-1] + self.u[:-2, 1:-1] + self.u[1:-1, 2:] + self.u[1:-1,:-2] - self.u[1:-1,1:-1] * 4
        ) * self.nu * dt / dx ** 2

        new_v[1:-1, 1:-1] = self.v[1:-1, 1:-1] - (
                self.u[1:-1, 1:-1] * (self.u[1:-1, 1:-1] - self.u[:-2, 1:-1])
                + self.v[1:-1, 1:-1] * (self.v[1:-1, 1:-1] - self.v[1:-1, :-2])
                + (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * self.rho)
        ) * dt / dx + (
                self.v[2:, 1:-1] + self.v[:-2, 1:-1] + self.v[1:-1, 2:] + self.v[1:-1, :-2] - self.v[1:-1, 1:-1] * 4
        ) * self.nu * dt / dx ** 2

        # extrapolation
        new_u[0, :] = new_u[1, :]
        new_u[-1, :] = new_u[-2, :]
        new_v[0, :] = new_v[1, :]
        new_v[-1, :] = new_v[-2, :]

        self.v = new_v
        self.u = new_u

    def animate(self, i):
        # g, p = self.dataset[i]  # type:  torch.FloatTensor
        for _ in range(10):
            # make 10 steps of simulation before drawing
            g = calculate_g(np.expand_dims(self.u, axis=0), np.expand_dims(self.v, axis=0))
            g_batch = torch.FloatTensor(g / self.g_max)  # make batch and normalize
            with torch.no_grad():
                p_pred = self.system.forward(g_batch)
            p_pred.squeeze_(dim=0)  # un-batch
            p = p_pred.numpy() * self.p_max  # de-normalize
            self.advect(p)  # re-calculate velocities
        self.draw(p, self.u, self.v)
        return self.frame


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--checkpoint')
    arg_parser.add_argument('--trn_path')
    arg_parser.add_argument('--val_path')
    arg_parser.add_argument('--animation', type=str, default=None,
                            help="Path to store animation. If None - show in place")
    arg_parser.add_argument('--dt', type=float, default=0.01)
    args = arg_parser.parse_args()

    p_max, g_max = get_normalization_values(args.trn_path)

    system = PoissonSolver(
        trn_path=args.trn_path,
        val_path=args.val_path,
        dt=args.dt,
        batch_size=1,
        p_max=p_max,
        g_max=g_max
    )

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    system.load_state_dict(checkpoint['state_dict'])
    system.eval()

    # dataset = PoissonDataset(args.val_path, dt=args.dt, p_max=p_max, g_max=g_max)
    data = np.load(args.val_path)
    scene = Scene(system, data, p_max, g_max)
    ani = animation.FuncAnimation(scene.fig, scene.animate, frames=len(data['p']) // 10, interval=int(1000 / 10),
                                  blit=True,
                                  repeat=False)

    if args.animation is not None:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=10)
        ani.save(args.animation, writer=writer, dpi=200)
    else:
        plt.show()
