import numpy as np
import matplotlib.pyplot as plt

LAM, AP, BP = 3.1, 10.0, 16.0          # см


def f_e(theta):
    """F_E(theta) = (1+cos)/2 * sin(u)/u,  u = pi*b_p*sin(theta)/lambda"""
    u = np.pi * BP / LAM * np.sin(theta)
    return np.abs((1 + np.cos(theta)) / 2 * np.sinc(u / np.pi))


def f_h(theta):
    """F_H(theta) = (1+cos)/2 * cos(v)/(1-(2v/pi)^2),  v = pi*a_p*sin(theta)/lambda"""
    v = np.pi * AP / LAM * np.sin(theta)
    den = 1 - (2 * v / np.pi) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.cos(v) / den
    r = np.where(np.abs(den) < 1e-9, np.pi / 4, r)      # границя при 2v/pi = 1
    return np.abs((1 + np.cos(theta)) / 2 * r)


if __name__ == "__main__":
    th = np.radians(np.linspace(0, 90, 9001))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(np.degrees(th), f_e(th), label="Площина E")
    ax.plot(np.degrees(th), f_h(th), label="Площина H")
    ax.axhline(0.707, ls="--", color="gray")
    ax.set_xlabel("θ, град"); ax.set_ylabel("F(θ)"); ax.grid(alpha=.4); ax.legend()
    plt.savefig("horn_pattern.png", dpi=200)
