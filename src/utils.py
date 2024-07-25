import numpy as np


def calculate_g(ut, vt, rho=1., dt=0.01, h=1.):
    dx = dy = h
    g = np.zeros_like(vt)
    g[:, 1:-1, 1:-1] = ((ut[:, 2:, 1:-1] - ut[:, :-2, 1:-1]) / (2 * dx)
                        + (vt[:, 1:-1, 2:] - vt[:, 1:-1, :-2]) / (2 * dy)) * rho / dt

    g2 = np.zeros_like(g)
    g2[:, 1:-1, 1:-1] = (
                                (ut[:, 2:, 1:-1] - ut[:, :-2, 1:-1]) ** 2 / (4 * dx ** 2)
                                + (vt[:, 1:-1, 2:] - vt[:, 1:-1, :-2]) ** 2 / (4 * dy ** 2)
                                + (ut[:, 1:-1, 2:] - ut[:, 1:-1, :-2]) * (vt[:, 2:, 1:-1] - vt[:, :-2, 1:-1]) / (
                                        2 * dx * dy)
                        ) * rho

    g -= g2
    return g


def get_normalization_values(trn_path: str, rho: float = 1, dt: float = 0.01, h: float = 1.0):
    data_archive = np.load(trn_path)
    assert len({'p', 'u', 'v'} & set(data_archive.keys())) == 3

    pt = data_archive['p']
    vt = data_archive['v']
    ut = data_archive['u']
    g = calculate_g(ut, vt, rho, dt, h)

    p_max = abs(pt).max()
    g_max = abs(g).max()
    return p_max, g_max
