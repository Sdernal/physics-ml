import logging

import numpy as np
from argparse import ArgumentParser

from matplotlib import animation
from tqdm import tqdm
import matplotlib.pyplot as plt

class Fluid:
    def __init__(self):
        self.nt = 200
        self.ngs = 32
        self.nx = 64
        self.ny = 64
        self.h = 1.
        self.rho = 1.
        self.nu = 0.5
        self.u = np.zeros((self.nx, self.ny))
        self.v = np.zeros((self.nx, self.ny))
        self.p = np.zeros((self.nx, self.ny))

        self.p[0, :] = 5
        self.p[-1, :] = 2

    def solve_v(self, dt):
        b = np.zeros((self.nx, self.ny))
        dx = dy = self.h
        for _ in range(self.nt):
            b[1:-1, 1:-1] = (
                (self.u[2:, 1:-1] - self.u[:-2, 1:-1] + self.v[1:-1, 2:] - self.v[1:-1, :-2]) * 2 / dt
                - (self.u[1:-1, 2:] - self.u[1:-1, :-2])*(self.v[2:, 1:-1] - self.v[:-2, 1:-1]) * 2 / dx
                - (self.u[2:, 1:-1] - self.u[:-2, 1:-1])**2 / dx
                - (self.v[1:-1, 2:] - self.v[1:-1, :-2])**2 / dx
            ) * self.rho * dx / 16

            for _ in range(self.ngs):
                self.p[1:-1, 1:-1] = (self.p[2:, 1:-1] + self.p[:-2, 1:-1] + self.p[1:-1, 2:] + self.p[1:-1,:-2]) / 4 - b[1:-1, 1:-1]

                self.p[:, 0] = self.p[:, 1]
                self.p[:, -1] = self.p[:, -2]

    def advect_v(self, dt):
        new_u = self.u.copy()
        new_v = self.v.copy()
        dx = dy = self.h

        new_u[1:-1, 1:-1] = self.u[1:-1, 1:-1] - (
            self.u[1:-1, 1:-1] * (self.u[1:-1, 1:-1] - self.u[:-2, 1:-1])
            + self.v[1:-1, 1:-1] * (self.u[1:-1, 1:-1] - self.u[1:-1, :-2])
            + (self.p[2:, 1:-1] - self.p[:-2, 1:-1]) / (2*self.rho)

        ) * dt / dx + (
                    self.u[2:, 1:-1] + self.u[:-2, 1:-1] + self.u[1:-1, 2:] + self.u[1:-1, :-2] - self.u[1:-1, 1:-1] * 4
        ) * self.nu * dt / dx**2

        new_v[1:-1, 1:-1] = self.v[1:-1, 1:-1] - (
            self.u[1:-1, 1:-1] * (self.u[1:-1, 1:-1] - self.u[:-2, 1:-1])
            + self.v[1:-1, 1:-1] * (self.v[1:-1, 1:-1] - self.v[1:-1, :-2])
            + (self.p[1:-1, 2:] - self.p[1:-1, :-2]) / (2 * self.rho)
        ) * dt / dx + (
            self.v[2:, 1:-1] + self.v[:-2, 1:-1] + self.v[1:-1, 2:] + self.v[1:-1, :-2] - self.v[1:-1, 1:-1] * 4
        ) * self.nu * dt / dx**2

        # extrapolation
        new_u[0, :] = new_u[1, :]
        new_u[-1, :] = new_u[-2, :]
        new_v[0, :] = new_v[1, :]
        new_v[-1, :] = new_v[-2, :]

        self.v = new_v
        self.u = new_u


class Scene:
    def __init__(self, fluid, dt):
        self.fig, ( self.ax_p, self.ax_u, self.ax_v, self.ax_lp) = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
        self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        self.frame = []
        self.fluid = fluid
        self.draw()
        self.dt = dt
        self.left_path = None
        self.right_path = None

    def draw(self):
        while len(self.frame):
            self.frame[-1].remove()
            self.frame.pop()

        self.frame.append(self.ax_p.imshow(np.rot90(self.fluid.p)))
        self.frame.append(self.ax_u.imshow(np.rot90(self.fluid.u)))
        self.frame.append(self.ax_v.imshow(np.rot90(self.fluid.v)))
        p = self.fluid.p
        dx = dy = 1
        laplassian_p = np.zeros((64, 64))
        laplassian_p[1:-1, 1:-1] = ((p[2:, 1:-1] - p[1:-1, 1:-1] * 2 + p[:-2, 1:-1]) / dx ** 2
                                    + (p[1:-1, 2:] - p[1:-1, 1:-1] * 2 + p[1:-1, :-2]) / dy ** 2)
        self.frame.append(self.ax_lp.imshow(np.rot90(laplassian_p)))

    def animate(self, i):
        if i % 100 == 0:
            # every 100 frames start randomly changing pressure on sides
            force = np.random.random(2)*10
            left_target = force[0]
            right_target = force[1]
            self.left_path = np.linspace(self.fluid.p[0, 0], left_target, 100)
            self.right_path = np.linspace(self.fluid.p[-1, 0], right_target, 100)
            print(f'new_target pressure: {self.left_path } {self.right_path}')
        elif self.left_path is not None and self.right_path is not None:
            self.fluid.p[0, :] = self.left_path[i % 100]
            self.fluid.p[-1, :] = self.right_path[i % 100]
        # else don't move
        self.fluid.solve_v(self.dt)
        self.fluid.advect_v(self.dt)
        self.draw()
        return self.frame


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--dst', type=str, default="data.npz",
                            help="Path to store dataset. Should end with .npz")
    arg_parser.add_argument('--frames', type=int, default=100, help="Length of data in frames")
    arg_parser.add_argument('--dt', type=float, default=0.001, help="Time between iterations")
    arg_parser.add_argument('--fast', action="store_true", default=False,
                            help="Apply vector operations to increase speed")
    arg_parser.add_argument('--animate', action="store_true", default=False,
                            help="Animate obtained data during process")

    args = arg_parser.parse_args()

    fluid = Fluid()
    if args.animate:
        scene = Scene(fluid, dt=1/100)
        ani = animation.FuncAnimation(scene.fig, scene.animate, frames=args.frames, interval=int(1000 / 10),
                                      blit=True,
                                      repeat=False)
        plt.show()
    else:
        pt, vt, ut = [], [], []

        # save initial pressure and velocities
        pt.append(fluid.p.copy())
        vt.append(fluid.v.copy())
        ut.append(fluid.u.copy())
        # left_path = None
        # right_path = None
        for frame in tqdm(range(args.frames)):
            # TODO: working very strange
            # Every 100 frames start changing pressure
            # if frame % 100 == 0:
            #     force = np.random.random(2) * 10
            #     left_target = force[0]
            #     right_target = force[1]
            #     left_path = np.linspace(fluid.p[0, 0], left_target, 100)
            #     right_path = np.linspace(fluid.p[-1, 0], right_target, 100)
            # else:
            #     fluid.p[0, :] = left_path[frame % 100]
            #     fluid.p[-1, :] = right_path[frame % 100]
            # make step and save data
            fluid.solve_v(args.dt)
            fluid.advect_v(args.dt)

            pt.append(fluid.p.copy())
            vt.append(fluid.v.copy())
            ut.append(fluid.u.copy())

        # stack features
        pt = np.stack(pt)
        vt = np.stack(vt)
        ut = np.stack(ut)

        logging.info(f"Saving to {args.dst}... With following parameters: "
                     f"rho={fluid.rho}"
                     f"nu={fluid.nu}"
                     f"dt={args.dt}"
                     f"h={fluid.h}")
        np.savez(args.dst, p=pt, v=vt, u=ut)
