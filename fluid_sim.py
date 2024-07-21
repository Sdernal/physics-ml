# Python adaptation of tenMinutePhysics
import random
from typing import List

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from numpy.random import randint
import numpy as np
from math import floor
from argparse import ArgumentParser


class Solid:
    """
    Dataclass for obstacles (if needed)
    """
    x: float
    y: float
    w: float
    h: float

    def __init__(self, x, y, w=1, h=1):
        self.x = x
        self.y = y
        self.w = w
        self.h = h


class Fluid:
    def __init__(self, density, num_x, num_y, h, over_relaxation=1.9):
        self.density = density
        # TODO: removed additional solid cells on walls
        self.num_x = num_x  # +2
        self.num_y = num_y  # +2
        self.h = h  # cell height
        self.p = np.zeros((self.num_x, self.num_y))  # pressure
        self.s = np.zeros((self.num_x, self.num_y))  # states (wall - non-wall)
        self.m = np.ones((self.num_x, self.num_y))  # masses ?
        self.v = np.zeros((self.num_x, self.num_y))  # vertical velocities
        self.u = np.zeros((self.num_x, self.num_y))  # horizontal velocities
        self.over_relaxation = over_relaxation  # TODO: hardcoded
        self.outer_force = [self.num_x // 2, self.num_y // 2]
        self.obstacle_x = None
        self.obstacle_y = None
        self.obstacles = []  # type: List[Solid]  # TODO: for multiple obstacles, empty for now

    def initialize(self, num_pipes: int = 0):
        # tank
        for i in range(self.num_x):
            for j in range(self.num_y):
                s = 1.0  # fluid
                if i == 0 or i == self.num_x - 1 or j == 0:
                    s = 0.0  # solid
                self.s[i, j] = s

        # TODO: let pipes be hardcoded for now
        if num_pipes > 0:
            #  pipe 1
            self.u[1, :] = 10.0
            self.m[0, 30:-30] = 0.0

        if num_pipes > 1:
            # pipe 2
            self.v[:, 1] = 10.0
            self.m[30:-30, 0] = 0.0

    def set_obstacle(self, x, y, dt, r=2.0):
        # print(f"Obstacle position: {x, y}")  # This is for Debugging
        vx = (x - self.obstacle_x) / dt if self.obstacle_x is not None else 0.0
        vy = (y - self.obstacle_y) / dt if self.obstacle_y is not None else 0.0

        self.obstacle_x, self.obstacle_y = x, y

        for i in range(1, self.num_x - 2):
            for j in range(1, self.num_y - 2):
                self.s[i, j] = 1.0

                dx = (i + 0.5) * self.h - x
                dy = (j + 0.5) * self.h - y

                if dx * dx + dy * dy < r * r:
                    self.s[i, j] = 0.0
                    self.m[i, j] = 1.0
                    self.u[i, j] = vx
                    self.u[i + 1, j] = vx
                    self.v[i, j] = vy
                    self.v[i, j + 1] = vy

    def move_obstacle(self, dx, dy, idx, dt):
        """
        TODO: Temporary method to move several objects, not used for now
        """
        vx = dx / dt
        vy = dy / dt
        self.obstacles[idx].x += dx
        self.obstacles[idx].y += dy

        for i in range(1, self.num_x - 2):
            for j in range(1, self.num_y - 2):
                self.s[i, j] = 1.0

                dx = (i + 0.5) * self.h - self.obstacles[idx].x
                dy = (j + 0.5) * self.h - self.obstacles[idx].y

                if abs(dx) < self.obstacles[idx].w and abs(dy) < self.obstacles[idx].h:
                    self.s[i, j] = 0.0
                    self.m[i, j] = 1.0
                    self.u[i, j] = vx
                    self.u[i + 1, j] = vx
                    self.v[i, j] = vy
                    self.v[i, j + 1] = vy

    def simulate(self, dt, gravity, num_iters):
        self.integrate(dt, gravity)
        self.p = np.zeros((self.num_x, self.num_y))
        self.solve_incomressibility(num_iters, dt)
        self.extrapolate()
        self.advect_vel(dt)
        self.advect_smoke(dt)

    def integrate(self, dt, gravity):
        for i in range(1, self.num_x):
            for j in range(1, self.num_y - 1):
                if self.s[i, j] > 0 and self.s[i, j - 1] > 0:
                    self.v[i, j] += gravity * dt

    def apply_random_force(self, dt):
        """
        TODO: Method for applying random force to random set of cells, not used for now
        """
        self.outer_force[0] += randint(0, 3) - 1
        self.outer_force[1] += randint(0, 3) - 1
        i, j = self.outer_force
        random_force = np.random.random(2) * 100 - 50
        for r1 in range(5):
            for r2 in range(5):
                # if i - r > 1 and j - r > 1:
                # self.u[i-r1, j-r2] += random_force[0] * dt
                # self.v[i-r1, j-r2] += random_force[1] * dt
                # if i + r < self.num_x - 1 and j + r < self.num_y - 1 and r > 0:
                self.u[i + r1, j + r2] += random_force[0] * dt
                self.v[i + r1, j + r2] += random_force[1] * dt
        print(f"Applying random force on {i, j}")

    def solve_incomressibility(self, num_iters, dt):
        cp = self.density * self.h / dt
        for _ in range(num_iters):
            for i in range(1, self.num_x - 1):
                for j in range(1, self.num_y - 1):
                    if self.s[i, j] == 0.0:
                        continue

                    s = self.s[i + 1, j] + self.s[i - 1, j] + self.s[i, j + 1] + self.s[i, j - 1]
                    if s == 0:
                        continue

                    d = self.u[i + 1, j] - self.u[i, j] + self.v[i, j + 1] - self.v[i, j]  # divergence

                    ds = (-d / s) * self.over_relaxation
                    self.p[i, j] += ds * cp  # cp = self.density * self.h / dt TODO: very strange results

                    self.u[i, j] -= ds * self.s[i - 1, j]
                    self.u[i + 1, j] += ds * self.s[i + 1, j]
                    self.v[i, j] -= ds * self.s[i, j - 1]
                    self.v[i, j + 1] += ds * self.s[i, j + 1]

    def extrapolate(self):
        self.u[:, 0] = self.u[:, 1]
        self.u[:, -1] = self.u[:, -2]

        self.v[0, :] = self.v[1, :]
        self.v[-1, :] = self.v[-2, :]

    def advect_vel(self, dt):
        new_u = self.u.copy()
        new_v = self.v.copy()

        for i in range(1, self.num_x):
            for j in range(1, self.num_y):
                # u component
                if self.s[i, j] != 0.0 and self.s[i - 1, j] != 0.0 and j < self.num_y - 1:
                    x = i * self.h
                    y = j * self.h + self.h / 2
                    u = self.u[i, j]
                    v = np.mean([
                        self.v[i - 1, j], self.v[i, j], self.v[i - 1, j + 1], self.v[i, j + 1]
                    ])
                    x -= dt * u
                    y -= dt * v
                    u = self.sample_field(x, y, 'u')
                    new_u[i, j] = u
                # v component
                if self.s[i, j] != 0 and self.s[i, j - 1] != 0 and i < self.num_x - 1:
                    x = i * self.h + self.h / 2
                    y = j * self.h
                    u = np.mean([
                        self.u[i, j - 1], self.u[i, j], self.u[i + 1, j - 1], self.u[i + 1, j]
                    ])
                    v = self.v[i, j]
                    x -= - dt * u
                    y -= dt * v
                    v = self.sample_field(x, y, 'v')
                    new_v[i, j] = v

        self.u = new_u
        self.v = new_v

    def advect_smoke(self, dt):
        new_m = self.m.copy()
        for i in range(1, self.num_x - 1):
            for j in range(1, self.num_y - 1):
                if self.s[i, j] != 0:
                    # take velocity in center of a cell
                    u = (self.u[i, j] + self.u[i + 1, j]) / 2
                    v = (self.v[i, j] + self.v[i, j + 1]) / 2
                    # and calculate previous position to this center
                    x = i * self.h + self.h / 2 - dt * u
                    y = j * self.h + self.h / 2 - dt * v
                    # and a "mass" in this point
                    new_m[i, j] = self.sample_field(x, y, 's')
        self.m = new_m

    def sample_field(self, x, y, field):
        # adjust borders
        # if x < self.h:
        #     x = self.h
        # elif x > self.num_x * self.h:
        #     x = self.num_x * self.h
        x = max(self.h, min(x, self.num_x * self.h))
        y = max(self.h, min(y, self.num_y * self.h))

        f = None
        dx, dy = 0.0, 0.0
        if field == 'u':
            f, dy = self.u, self.h / 2
        elif field == 'v':
            f, dx = self.v, self.h / 2
        elif field == 's':
            f, dx, dy = self.m, self.h / 2, self.h / 2
        else:
            assert False, f'Field should be in (u,v,s) not {field}'

        # General Grid Interpolation
        w = np.zeros((2, 2))
        x0 = min(self.num_x - 1, floor((x - dx) / self.h))
        w[0, 1] = ((x - dx) - x0 * self.h) / self.h
        x1 = min(x0 + 1, self.num_x - 1)

        y0 = min(self.num_y - 1, floor((y - dy) / self.h))
        w[1, 1] = ((y - dy) - y0 * self.h) / self.h
        y1 = min(y0 + 1, self.num_y - 1)

        w[0, 0] = 1 - w[0, 1]
        w[1, 0] = 1 - w[1, 1]

        # return value with coefficients from 4 corners around result point
        result = w[0, 0] * w[1, 0] * f[x0, y0] \
                 + w[0, 1] * w[1, 0] * f[x1, y0] \
                 + w[0, 1] * w[1, 1] * f[x1, y1] \
                 + w[0, 0] * w[1, 1] * f[x0, y1]  # maybe there is a mistake in PDF
        return result


class Scene:
    def __init__(self, fluid, axes, add_obstacle=False, fps=30):
        self.fluid = fluid
        self.axes = axes
        self.fps = fps
        self.add_obstacle = add_obstacle
        self.im_ax = []

        if add_obstacle:
            self.fluid.set_obstacle(self.fluid.num_x // 2, self.fluid.num_y // 2, 1 / self.fps)

        # draw first frame
        self.draw()
        # paths for moving obstacle
        self.x_path = None
        self.y_path = None

    def draw(self):
        ax_p, ax_m = self.axes
        if len(self.im_ax) == 0:
            # first frame, need to initialize
            if self.add_obstacle:
                scat_p = ax_p.scatter(self.fluid.num_x // 2 - self.fluid.h / 2,
                                         self.fluid.num_y // 2 - self.fluid.h / 2, s=10)  # TODO: calculate s
                scat_m = ax_m.scatter(self.fluid.num_x // 2 - self.fluid.h / 2,
                                           self.fluid.num_y // 2 - self.fluid.h / 2, s=10)  # TODO: calculate s
                self.im_ax.append(scat_p)
                self.im_ax.append(scat_m)
            self.im_ax.append(ax_p.imshow(np.rot90(self.fluid.p)))
            self.im_ax.append(ax_m.imshow(np.rot90(self.fluid.m)))
        else:
            # later frames we exclude image before drawing another
            self.im_ax[-1].remove()  # ax_m
            self.im_ax.pop()
            self.im_ax[-1].remove()  # ax_p
            self.im_ax.pop()

            self.im_ax.append(ax_p.imshow(np.rot90(self.fluid.p)))
            self.im_ax.append(ax_m.imshow(np.rot90(self.fluid.m)))
            if self.add_obstacle:
                self.im_ax[0].set_offsets((self.fluid.obstacle_x - self.fluid.h / 2,
                                           self.fluid.num_y - self.fluid.obstacle_y - self.fluid.h / 2))
                self.im_ax[1].set_offsets((self.fluid.obstacle_x - self.fluid.h / 2,
                                           self.fluid.num_y - self.fluid.obstacle_y - self.fluid.h / 2))

    def add_objects(self):
        """
        TODO: temporary method to add different obstacles not used for now
        """
        # temporary code for adding multiple objects
        left_obj = Solid(x=32, y=14, w=10)
        center_obj = Solid(x=40, y=32, w=10)
        right_obj = Solid(x=32, y=50, w=10)
        self.fluid.obstacles = [left_obj, right_obj, center_obj]

    def move_obstacle(self, i):
        if i % 20 == 0:
            target_point_x = randint(5, self.fluid.num_x-5) * self.fluid.h
            target_point_y = randint(5, self.fluid.num_y-5) * self.fluid.h
            self.x_path = np.linspace(self.fluid.obstacle_x, target_point_x, 10)
            self.y_path = np.linspace(self.fluid.obstacle_y, target_point_y, 10)
            self.fluid.set_obstacle(self.x_path[i % 10], self.y_path[i % 10], 1 / self.fps)
        elif (i % 20) < 10:
            if self.x_path is None or self.y_path is None:
                self.x_path = np.ones(5) * self.fluid.obstacle_x
                self.y_path = np.ones(5) * self.fluid.obstacle_y
            self.fluid.set_obstacle(self.x_path[i % 10], self.y_path[i % 10], 1 / 30)
        else:
            # not move obstacle
            pass

    def move_obstacles(self, i):
        """
        TODO: Temporary method for move several objects
        """
        # this is for multiple obstacles
        fluid.apply_random_force(1/30)
        phase = i % 10
        fluid.move_obstacle(0, -1 if phase < 5 else 1, 0, 1 / 30)
        fluid.move_obstacle(0, -1 if phase < 5 else 1, 1, 1 / 30)
        fluid.move_obstacle(0, 1 if phase < 5 else -1, 2, 1 / 30)

    def animate(self, i):
        if self.add_obstacle:
            self.move_obstacle(i)

        fluid.simulate(1 / self.fps, -9.81, 40)
        self.draw()

        print(i)

        return self.im_ax

    def init(self):
        return self.im_ax


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--animation', type=str, default=None,
                            help="Path to store animation. If None - show in place")
    arg_parser.add_argument('--obstacle', action="store_true", default=False, help="Flag to add obstacle")
    arg_parser.add_argument('--pipes', type=int, default=0, help="Number of pipes with additional flow")
    arg_parser.add_argument('--density', type=float, default=1., help="Density of fluid")
    arg_parser.add_argument('--num_x', type=int, default=64, help="Number of cells horizontally")
    arg_parser.add_argument("--num_y", type=int, default=64, help="Number of cells vertically")
    arg_parser.add_argument("--h", type=float, default=1., help="Size of cell")
    arg_parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    arg_parser.add_argument("--frames", type=int, default=300, help="Number of frames of simulation")

    args = arg_parser.parse_args()
    fluid = Fluid(args.density, args.num_x, args.num_y, args.h)
    fluid.initialize(num_pipes=args.pipes)

    fig, (ax_p, ax_m) = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    fig.set_size_inches(1.28, 0.64, True)
    ax_p.axis('off')
    ax_m.axis('off')
    scene = Scene(fluid, (ax_p, ax_m), args.obstacle)
    ani = animation.FuncAnimation(fig, scene.animate, frames=args.frames, interval=int(1000 / args.fps), blit=True,
                                  init_func=scene.init,
                                  repeat=False)
    if args.animation is not None:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=args.fps)
        ani.save(args.animation, writer=writer, dpi=200)
    else:
        plt.show()
