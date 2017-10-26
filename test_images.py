from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import fitsio
import numpy
import pickle

import matplotlib

import matplotlib.pyplot as plt

data = fitsio.read('HIP86032_57502.37260_97507036_OFF.fits')

n = len(data)
std = np.std(data[int(0.25*n): int(0.75*n)])
min_val = min(data.flatten())
max_val = max(data.flatten())
data = data - min_val
data = data * (255/(max_val-min_val))
data = data.astype(int).astype(float)
print(min(data.flatten()))
print(data)


f, ax = plt.subplots(2, sharex=True)
ax[0].imshow(fitsio.read('HIP86032_57502.37260_97507036_OFF.fits'), aspect='auto')
ax[1].imshow(data, aspect='auto')
plt.show()

