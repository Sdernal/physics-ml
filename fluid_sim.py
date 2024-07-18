# Python adaptation of tenMinutePhysics
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from numpy.random import randint
import numpy as np
from math import floor


class Fluid:
    def __init__(self, density, num_x, num_y, h):
        self.density = density
        self.num_x = num_x + 2
        self.num_y = num_y + 2
        self.h = h  # cell height
        self.p = np.zeros((self.num_x, self.num_y))  # pressure
        self.s = np.zeros((self.num_x, self.num_y))  # states (wall - non-wall)
        self.m = np.ones((self.num_x, self.num_y))  # masses ?
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
                    self.v[i, j+1] += ds * self.s[i, j+1]

    def extrapolate(self):
        self.u[:, 0] = self.u[:, 1]
        self.u[:, -1] = self.u[:, -2]

        self.v[0, :] = self.v[1, :]
        self.v[-1, :] = self.v[-2, :]

    def advect_vel(self, dt):
        new_u = self.u.copy()
        new_v = self.v.copy()

        for i in range(1,self.num_x):
            for j in range(1, self.num_y):
                # u component
                if self.s[i, j] != 0.0 and self.s[i-1, j] != 0.0 and j < self.num_y - 1:
                    x = i * self.h
                    y = j * self.h + self.h / 2
                    u = self.u[i, j]
                    v = np.mean([
                        self.v[i-1, j], self.v[i, j], self.v[i-1, j+1], self.v[i, j+1]
                    ])
                    x -= dt*u
                    y -= dt*v
                    u = self.sample_field(x, y, 'u')
                    new_u[i, j] = u
                # v component
                if self.s[i, j] != 0 and self.s[i, j-1] != 0 and i < self.num_x - 1:
                    x = i * self.h + self.h / 2
                    y = j * self.h
                    u = np.mean([
                        self.u[i,j-1], self.u[i,j], self.u[i+1, j-1], self.u[i+1,j]
                    ])
                    v = self.v[i,j]
                    x -= - dt*u
                    y -= dt*v
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
                    u = (self.u[i, j] + self.u[i+1, j]) / 2
                    v = (self.v[i, j] + self.v[i, j+1]) / 2
                    # and calculate previous position to this center
                    x = i*self.h + self.h/2 - dt*u
                    y = j*self.h + self.h/2 - dt*v
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
            f, dx, dy = self.s, self.h / 2, self.h / 2
        else:
            assert False, f'Field should be in (u,v,s) not {field}'

        # General Grid Interpolation
        w = np.zeros((2,2))
        x0 = min(self.num_x-1, floor((x-dx) / self.h))
        w[0,1] = ((x-dx)-x0*self.h)/self.h
        x1 = min(x0+1, self.num_x-1)

        y0 = min(self.num_y-1, floor((y-dy) / self.h))
        w[1,1] = ((y-dy) - y0*self.h)/self.h
        y1 = min(y0+1, self.num_y-1)

        w[0, 0] = 1 - w[0, 1]
        w[1, 0] = 1 - w[1, 1]

        # return value with coefficients from 4 corners around result point
        result = w[0, 0] * w[1, 0] * f[x0, y0] \
            + w[0, 1] * w[1, 0] * f[x1, y0] \
            + w[0, 1] * w[1, 1] * f[x1, y1]  \
            + w[0, 0] * w[1, 1] * f[x0, y1]  # maby there is a mistake in PDF
        return result


class Scene:
    def __init__(self, fluid, axes):
        self.fluid = fluid
        self.axes = axes

        self.im_ax = []
        pressure = fluid.p
        self.im_ax.append(self.axes.imshow(np.rot90(pressure)))

    def get_heatmap(self):
        pass

    def animate(self, i):
        self.im_ax[-1].remove()
        self.im_ax.pop()
        fluid.simulate(1/30, -9.81, 40)
        pressure = fluid.p
        print(i)
        self.im_ax.append(self.axes.imshow(np.rot90(pressure)))
        return self.im_ax

    def init(self):
        return self.im_ax


if __name__ == '__main__':
    fluid = Fluid(1000, 64, 64, 1)
    fluid.initialize()
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=30, bitrate=3600)
    # fig = plt.Figure(figsize=(8,8))
    # plt.gca().set_axis_off()
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
    #                     hspace=0, wspace=0)
    # plt.margins(0, 0)

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    axes.axis('off')
    scene = Scene(fluid, axes)
    ani = animation.FuncAnimation(fig, scene.animate, frames=300, interval=int(1000/30), blit=True, init_func=scene.init,
                                  repeat=False)

    # ani.save('result3.gif', writer=writer)
    plt.show()
    # print('kek')