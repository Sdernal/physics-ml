import logging

import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm


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

    def solve(self, dt):
        b = np.zeros((self.nx, self.ny))
        dx = dy = self.h
        for _ in range(self.nt):
            for i in range(1, self.nx - 1):
                for j in range(1, self.ny - 1):
                    b[i, j] = self.rho / dt * ((self.u[i + 1, j] - self.u[i - 1, j]) / (2 * dx)
                                               + (self.v[i, j + 1] - self.v[i, j - 1]) / (2 * dy))

            for _ in range(self.ngs):
                for i in range(1, self.nx - 1):
                    for j in range(1, self.ny - 1):
                        self.p[i, j] = (self.p[i + 1, j] + self.p[i - 1, j] + self.p[i, j + 1] + self.p[i, j - 1]
                                        - dx ** 2 * b[i, j]) / 4
                self.p[:, 0] = 0
                self.p[:, -1] = 0

    def solve_v(self, dt):
        b = np.zeros((self.nx, self.ny))
        dx = dy = self.h
        for _ in range(self.nt):
            b[1:-1, 1:-1] = ((self.u[2:, 1:-1] - self.u[:-2, 1:-1]) / (2*dx)
                             + (self.v[1:-1, 2:] - self.v[1:-1, :-2]) / (2*dy)) * self.rho / dt
            for _ in range(self.ngs):
                self.p[1:-1, 1:-1] = (self.p[2:, 1:-1] + self.p[:-2, 1:-1] + self.p[1:-1, 2:] + self.p[1:-1, :-2]
                                        -b[1:-1, 1:-1] * dx**2) / 4

                self.p[:, 0] = self.p[:, 1]
                self.p[:, -1] = self.p[:, -2]

    def advect(self, dt):
        new_u = self.u.copy()
        new_v = self.v.copy()
        dx = dy = self.h
        for i in range(1, self.nx - 1):
            for j in range(1, self.ny - 1):
                new_u[i, j] = self.u[i, j] - dt * (
                        self.u[i, j] * (self.u[i, j] - self.u[i - 1, j]) / dx  # u * du/dx
                        + self.v[i, j] * (self.u[i, j] - self.u[i, j - 1]) / dy  # v * du/dy
                        + (self.p[i + 1, j] - self.p[i - 1, j]) / (2 * self.rho * dx)  # 1/rho * dp / dx
                        + self.nu * (
                                (self.u[i + 1, j] - 2 * self.u[i, j] + self.u[i - 1, j]) / dx ** 2  # d^2u / dx^2
                                + (self.u[i, j + 1] - 2 * self.u[i, j] + self.u[i, j - 1]) / dy ** 2  # d^2u / dy^2
                        )
                )

                new_v[i, j] = self.v[i, j] - dt * (
                    self.u[i, j] * (self.v[i, j] - self.v[i-1, j]) / dx   # u * dv / dx
                    + self.v[i, j] * (self.v[i, j] - self.v[i, j-1]) / dy  # v * dv / dy
                    + (self.p[i, j+1] - self.p[i, j-1]) / (2 * self.rho * dy)  # dp / dy
                    + self.nu * (
                        (self.v[i+1, j] - 2*self.v[i, j] + self.v[i-1, j]) / dx**2  # d^2v / dx^2
                        + (self.v[i, j+1] - 2*self.v[i, j] + self.v[i, j-1]) / dy**2  # d^2v / dy^2
                    )
                )

        # extrapolation
        new_u[0, :] = new_u[1, :]
        new_u[-1, :] = new_u[-2, :]
        new_v[0, :] = new_v[1, :]
        new_v[-1, :] = new_v[-2, :]

        self.v = new_v
        self.u = new_u

    def advect_v(self, dt):
        new_u = self.u.copy()
        new_v = self.v.copy()
        dx = dy = self.h

        new_u[1:-1, 1:-1] = self.u[1:-1, 1:-1] - (
            self.u[1:-1, 1:-1] * (self.u[1:-1, 1:-1] - self.u[:-2, 1:-1]) / dx
            + self.v[1:-1, 1:-1] * (self.u[1:-1, 1:-1] - self.u[1:-1, :-2]) / dy
            + (self.p[2:, 1:-1] - self.p[:-2, 1:-1]) / (2*self.rho*dx)
            + (
                (self.u[2:, 1:-1] - self.u[1:-1, 1:-1] * 2 + self.u[:-2, 1:-1]) / dx**2
                + (self.u[1:-1, 2:] - self.u[1:-1, 1:-1] * 2 + self.u[1:-1, :-2]) / dy**2
            ) * self.nu
        ) * dt

        new_v[1:-1, 1:-1] = self.v[1:-1, 1:-1] - (
            self.u[1:-1, 1:-1] * (self.u[1:-1,1:-1] - self.u[:-2, 1:-1]) / dx
            + self.v[1:-1, 1:-1] * (self.v[1:-1,1:-1] - self.v[1:-1, :-2]) / dy
            + (self.p[1:-1, 2:] - self.p[1:-1, :-2]) / (2* self.rho * dy)
            + (
                (self.v[2:, 1:-1] - self.v[1:-1, 1:-1] * 2 + self.v[:-2, 1:-1]) / dx**2
                + (self.v[1:-1, 2:] - self.v[1:-1, 1:-1] * 2 + self.v[1:-1, :-2]) / dy**2
            ) * self.nu
        ) * dt

        # extrapolation
        new_u[0, :] = new_u[1, :]
        new_u[-1, :] = new_u[-2, :]
        new_v[0, :] = new_v[1, :]
        new_v[-1, :] = new_v[-2, :]

        self.v = new_v
        self.u = new_u


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--dst', type=str, default="data.npz",
                            help="Path to store dataset. Should end with .npz")
    arg_parser.add_argument('--frames', type=int, default=100, help="Length of data in frames")
    arg_parser.add_argument('--dt', type=float, default=0.001, help="Time between iterations")
    arg_parser.add_argument('--fast', action="store_true", default=False,
                            help="Apply vector operations to increase speed")
    args = arg_parser.parse_args()

    fluid = Fluid()

    pt, vt, ut = [], [], []

    # save initial pressure and velocities
    pt.append(fluid.p.copy())
    vt.append(fluid.v.copy())
    ut.append(fluid.u.copy())
    for frame in tqdm(range(args.frames)):
        # make step and save data
        if args.fast:
            fluid.solve_v(args.dt)
            fluid.advect_v(args.dt)
        else:
            fluid.solve(args.dt)
            fluid.advect(args.dt)

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
