import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Example data for CPT values
cpt = pd.DataFrame({
    'Var1': [0.1, 0.3, 0.6],
    'Var2': [0.7, 0.2, 0.1]
}, index=['State1', 'State2', 'State3'])

# Example delta table (same shape) representing changes in CPT values
delta = pd.DataFrame({
    'Var1': [0.05, -0.05, 0.1],
    'Var2': [-0.1, 0.2, -0.05]
}, index=['State1', 'State2', 'State3'])

plt.figure(figsize=(8, 6))
# The heatmap uses delta to control the background colors.
# The annot parameter overlays the original CPT values.
sns.heatmap(delta, annot=cpt, fmt=".2f", cmap="RdYlGn", center=0)
plt.title("CPT Values with Delta-Based Coloring")
plt.show()
