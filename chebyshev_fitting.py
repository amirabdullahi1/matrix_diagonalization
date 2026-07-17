import numpy as np
from numpy.polynomial import Chebyshev, Polynomial

x = np.linspace(-1, 1, 10000)
y = np.arctan(x)

p = Chebyshev.fit(x, y, 7)

power = p.convert(kind=Polynomial)

for i,c in enumerate(power.coef):
    print(f"x^{i}: {c}")