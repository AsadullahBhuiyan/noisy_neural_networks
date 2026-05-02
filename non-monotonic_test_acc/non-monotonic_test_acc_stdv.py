import numpy as np
import math
import matplotlib.pyplot as plt

a = 10.0
C_list = [10]
p_vals = np.linspace(0, 1, 300)

def Phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def mean_acc(p, C, a=1.0, n_gh=80):
    # E_u[ Phi(u + a(1-p))^(C-1) ], u ~ N(0,1)
    xs, ws = np.polynomial.hermite.hermgauss(n_gh)
    shift = a * (1.0 - p)
    s = 0.0
    for x, w in zip(xs, ws):
        u = math.sqrt(2.0) * x
        s += w * Phi(u + shift) ** (C - 1)
    return s / math.sqrt(math.pi)

def std_acc(p, C, a=1.0):
    m = mean_acc(p, C, a)
    return math.sqrt(max(0.0, m * (1.0 - m)))


# Plot
plt.figure(figsize=(8, 6))

for C in C_list:
    y = [std_acc(p, C, a) for p in p_vals]
    plt.plot(p_vals, y, linewidth=5.0, color='b')

plt.xlabel(r"$p$", fontsize=24)
plt.ylabel(r"$\mathrm{Std}[\mathrm{Acc}]$", fontsize=24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(0, 1)
plt.ylim(bottom=0)
plt.tight_layout()
plt.savefig("outputs/non-monotonic_test_acc/toy_model_stdv_test_acc.png", dpi=300)