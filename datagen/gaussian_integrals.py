"""
Engines for evaluating 1D and 2D Gaussian integrals over activation functions.
"""

import math
import torch
import numpy as np
from numpy.polynomial.hermite import hermgauss


torch.set_default_dtype(torch.float64)


def hermgauss_torch(n: int, device=None, dtype=torch.float64):
    """
    Return Gauss-Hermite nodes/weights as torch tensors.

    These correspond to quadrature for:
        \int exp(-x^2) f(x) dx
    """
    x_np, w_np = hermgauss(n)
    x = torch.tensor(x_np, device=device, dtype=dtype)
    w = torch.tensor(w_np, device=device, dtype=dtype)
    return x, w


class GaussianQuadrature1D:
    """
    1D Gaussian expectations using Gauss-Hermite quadrature.

    For Z ~ N(mean, var):
        E[f(Z)]
    """

    def __init__(self, n: int = 80, device=None, dtype=torch.float64):
        self.n = n
        self.device = device
        self.dtype = dtype
        self.x, self.w = hermgauss_torch(n, device=device, dtype=dtype)

    def standard_normal_expectation(self, f):
        """
        Compute E[f(Z)] for Z ~ N(0,1).
        """
        z = math.sqrt(2.0) * self.x
        vals = f(z)
        return torch.sum(self.w * vals) / math.sqrt(math.pi)

    def normal_expectation(self, f, mean, var):
        """
        Compute E[f(Z)] for Z ~ N(mean, var), var > 0.
        """
        mean = torch.as_tensor(mean, device=self.device, dtype=self.dtype)
        var = torch.as_tensor(var, device=self.device, dtype=self.dtype)
        if torch.any(var <= 0):
            raise ValueError("Variance must be positive.")
        z = mean + torch.sqrt(var) * math.sqrt(2.0) * self.x
        vals = f(z)
        return torch.sum(self.w * vals) / math.sqrt(math.pi)
    
    def E(self, fun1, fun2, q):
        return self.normal_expectation(lambda z: fun1(z) * fun2(z), mean=0.0, var=q)


class GaussianQuadrature2D:
    """
    2D Gaussian expectations using tensor-product Gauss-Hermite quadrature.

    For (U,V) ~ N(0, [[q1, c], [c, q2]]).
    """

    def __init__(self, n: int = 60, device=None, dtype=torch.float64):
        self.n = n
        self.device = device
        self.dtype = dtype

        x, w = hermgauss_torch(n, device=device, dtype=dtype)
        self.x = x
        self.w = w

        X, Y = torch.meshgrid(x, x, indexing="ij")
        WX, WY = torch.meshgrid(w, w, indexing="ij")

        self.X = X
        self.Y = Y
        self.WX = WX
        self.WY = WY

    def standard_bivariate_expectation(self, g, rho):
        """
        Compute E[g(U,V)] where (U,V) are standard normals with Corr(U,V)=rho.
        """
        rho = torch.as_tensor(rho, device=self.device, dtype=self.dtype)
        rho = torch.clamp(rho, -1.0, 1.0)

        z1 = math.sqrt(2.0) * self.X
        z2 = math.sqrt(2.0) * self.Y

        U = z1
        V = rho * z1 + torch.sqrt(1.0 - rho**2) * z2

        vals = g(U, V)
        return torch.sum(self.WX * self.WY * vals) / math.pi

    def general_bivariate_expectation(self, g, q1, q2, c):
        """
        Compute E[g(U,V)] where
            (U,V) ~ N(0, [[q1, c], [c, q2]]).
        """
        q1 = torch.as_tensor(q1, device=self.device, dtype=self.dtype)
        q2 = torch.as_tensor(q2, device=self.device, dtype=self.dtype)
        c = torch.as_tensor(c, device=self.device, dtype=self.dtype)

        if torch.any(q1 <= 0) or torch.any(q2 <= 0):
            raise ValueError("q1 and q2 must be positive.")

        rho = c / torch.sqrt(q1 * q2)
        rho = torch.clamp(rho, -1.0, 1.0)

        z1 = math.sqrt(2.0) * self.X
        z2 = math.sqrt(2.0) * self.Y

        U = torch.sqrt(q1) * z1
        V = torch.sqrt(q2) * (rho * z1 + torch.sqrt(1.0 - rho**2) * z2)

        vals = g(U, V)
        return torch.sum(self.WX * self.WY * vals) / math.pi
    
    def E(self, fun1, fun2, q1, q2, c):
        return self.general_bivariate_expectation(
            lambda u, v: fun1(u) * fun2(v), q1, q2, c
        )