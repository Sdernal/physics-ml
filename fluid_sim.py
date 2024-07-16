# Python adaptation of tenMinutePhysics
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from numpy.random import randint
import numpy as np


class Fluid:
    def __init__(self, density, num_x, num_y, h):
        self.density = density
        self.num_x = num_x + 2
        self.num_y = num_y + 2
        self.h = h  # cell height
        self.p = np.zeros((self.num_x, self.num_y))  # pressure
        self.s = np.zeros((self.num_x, self.num_y))  # states (wall - non-wall)
        self.m = np.ones((self.num_x, self.num_y))  # masses
        self.v = np.zeros((self.num_x, self.num_y))  # vertical velocities
        self.u = np.zeros((self.num_x, self.num_y))  # horizontal velocities
        self.over_relaxation = 1.9

    def initialize(self):
        # tank
        for i in range(self.num_x):
            for j in range(self.num_y):
                s = 1.0  # fluid
                if i == 0 or i == self.num_x-1 or j == 0:
                    s = 0.0  # solid
                self.s[i, j] = s

    def simulate(self, dt, gravity, num_iters):
        self.integrate(dt, gravity)
        self.p = np.zeros((self.num_x, self.num_y))
        self.solve_incomressibility(num_iters, dt)
        self.extrapolate()
        self.advect_vel(dt)
        self.advect_smoke(dt)

    def integrate(self, dt, gravity):
        for i in range(1, self.num_x):
            for j in range(1, self.num_y-1):
                if self.s[i, j] > 0 and self.s[i, j-1] > 0:
                    self.v[i, j] += gravity * dt

    def solve_incomressibility(self, num_iters, dt):
        cp = self.density * self.h / dt
        for _ in range(num_iters):
            for i in range(1, self.num_x-1):
                for j in range(1, self.num_y-1):
                    if self.s[i, j] == 0.0:
                        continue

                    s = self.s[i+1, j] + self.s[i-1, j] + self.s[i, j+1] + self.s[i, j-1]
                    if s == 0:
                        continue

                    d = self.u[i+1, j] - self.u[i, j] + self.v[i, j+1] - self.v[i, j]  # divergence

                    ds = (-d/s) * self.over_relaxation
                    self.p[i, j] += ds * cp   # cp = self.density * self.h / dt

                    self.u[i, j] -= ds * self.s[i-1, j]
                    self.u[i+1, j] += ds * self.s[i+1, j]
                    self.v[i, j] -= ds * self.s[i, j-1]
                    self.v[i, j+1] += ds * self.s[s, j+1]

    def extrapolate(self):
        # TODO
        pass


class Scene:
    def __init__(self, fluid, axes):
        self.fluid = fluid
        self.axes = axes

        self.im_ax = []
        self.im_ax.append(self.axes.imshow(randint(0, 256, (64, 64))))

    def get_heatmap(self):
        pass

    def animate(self, i):
        self.im_ax[-1].remove()
        self.im_ax.pop()
        self.im_ax.append(self.axes.imshow(randint(0, 256, (64, 64))))
        return self.im_ax

    def init(self):
        return self.im_ax


if __name__ == '__main__':
    fluid = Fluid(1, 64, 64, 1)

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=10, bitrate=3600)
    # fig = plt.Figure(figsize=(8,8))
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
    # axes = fig.add_subplot()


    scene = Scene(fluid, axes)
    ani = animation.FuncAnimation(fig, scene.animate, frames=60, interval=int(1000/10), blit=True, init_func=scene.init,
                                  repeat=False)

    ani.save('sample.gif', writer=writer)
    # plt.show()
    # print('kek')