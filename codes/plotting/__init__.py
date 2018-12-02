"""Plot utils"""
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
FIG_SIZE_1D = plt.rcParams['figure.figsize']
FIG_SIZE_1D_EXT = np.array(FIG_SIZE_1D) * 1.25
FIG_SIZE_2D = [max(FIG_SIZE_1D)] * 2
FIG_SIZE_2D_SUB2 = list(np.array(FIG_SIZE_2D) * np.array([2.4, 1]))