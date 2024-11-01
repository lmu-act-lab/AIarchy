import matplotlib.pyplot as plt

# Sample data
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

plt.plot(x, y)

# Add parameters on the right side outside the plot
plt.text(0.95, 0.5, 'Slope: 2\nIntercept: 0', fontsize=12, verticalalignment='center', horizontalalignment='right')


plt.show()
