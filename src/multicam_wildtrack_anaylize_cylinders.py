"""Get min, max and overview over cylinder data for min-max-scaling"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wildtrack_globals import MULTICAM_ROOT


cyl = np.load(f"{MULTICAM_ROOT}/cylinder_analysis/cylinder_analysis_array.npy")
df = pd.DataFrame(cyl, columns=["xcenter", "ycenter", "heigth", "radius"])

print(df.describe())

df.hist()
plt.savefig(f"{MULTICAM_ROOT}/cylinder_analysis/cylinder_analysis.png")
