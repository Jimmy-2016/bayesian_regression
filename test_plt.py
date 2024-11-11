import numpy as np
import matplotlib.pyplot as plt

plt.plot(np.cumsum(np.random.normal(0.01, 1, size=(1000,1))))

plt.show()