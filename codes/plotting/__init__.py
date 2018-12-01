"""Plot utils"""
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
FIG_SIZE = plt.rcParams['figure.figsize']
FIG_SIZE_SUB2 = list(np.array(FIG_SIZE) * np.array([2.5, 1.25]))