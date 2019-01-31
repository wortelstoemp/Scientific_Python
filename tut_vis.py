# Visualisations
# TODO(tom): http://www.scipy-lectures.org/intro/numpy/array_object.html#basic-visualization
# TODO(tom): http://www.scipy-lectures.org/intro/matplotlib/index.html
# TODO: https://www.oreilly.com/learning/an-illustrated-introduction-to-the-t-sne-algorithm
# ------------------------------------------------------------------------------
# matplotlib
# ------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# Data
x = np.linspace(-np.pi, np.pi, 256, endpoint=True)
c = np.cos(x)
s = np.sin(x)

# Create a figure of size 8x6 inches, 80 dots per inch
# FIGURES ARE NUMBERED STARTING FROM 1 INSTEAD OF 0 (like MATLAB)
plt.figure(num=1, figsize=(8, 6), dpi=80, facecolor="w", edgecolor="w", frameon=True)

# plt.close(1)     # Closes figure 1

# Create a new subplot from a grid of 1x1 of index 1
ax = plt.subplot(1, 1, 1)

# Setting x and y limits
plt.xlim(-4.0, 4.0)
plt.ylim(-1.0, 1.0)
# Ticks are visualized values at the axes, we give them a proper label string
plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
    [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])
plt.yticks([-1, 0, +1],
    [r'$-1$', r'$0$', r'$+1$'])

# Cartesian axes
ax = plt.gca()  # gca stands for 'get current axis'
ax.spines["right"].set_color("none")
ax.spines["top"].set_color("none")
ax.xaxis.set_ticks_position("bottom")
ax.spines["bottom"].set_position(("data", 0))
ax.yaxis.set_ticks_position("left")
ax.spines["left"].set_position(("data", 0))

# Make ticks more visible when they overlap with graphs
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontsize(16)
    label.set_bbox(dict(facecolor='white', edgecolor='None', alpha=0.65))

# Legend
plt.legend(loc="upper left")

# Plot and show
plt.plot(x, c, color="blue", linewidth=2.5, linestyle="-")
plt.plot(x, s, color="green", linewidth=2.5, linestyle="-")
plt.show()

# Save figure using 72 dots per inch
# plt.savefig("exercise_2.png", dpi=72)

# Scatter plots
# plt.scatter(X, Y) # instead of plt.plot(X, Y)
# plt.xticks(())
# plt.yticks(())
# plt.show()

# Bar plots
# plt.bar(X, Y, facecolor="#9999ff", edgecolor="white")
# for x, y in zip(X, Y):
#    plt.text(x, y, '%.2f' % y, ha="center", va="bottom")
# plt.xlim(-1.25, +1.25)
# plt.ylim(-1.25, +1.25)
# plt.xticks(())
# plt.yticks(())
# plt.show()

# Contour plots of a function f(x, y)
# X, Y = np.meshgrid(x, y)
# Z = f(X, Y)
# plt.contourf(X, Y, Z, 8, alpha=0.75, cmap=plt.cm.hot)
# C = plt.contour(X, Y, Z, 8, colors="black", linewidth=0.5)
# plt.clabel(C, inline=1, fontsize=10)
# plt.xticks(())
# plt.yticks(())
# plt.show()

# Imshow of a function f(x, y)
# X, Y = np.meshgrid(x, y)
# Z = f(X, Y)
# plt.imshow(Z, interpolation="nearest", cmap="bone", origin="lower")
# plt.colorbar(shrink=0.92)
# plt.xticks(())
# plt.yticks(())
# plt.show()

# Pie plot
# n = 20
# Z = np.ones(n)
# plt.pie(Z, explode=Z*.05, colors = ['%f' % (i/float(n)) for i in range(n)])
# plt.axis('equal')
# plt.xticks(())
# plt.yticks()
# plt.show()

# Text
# eqs = []
# eqs.append((r"$LaTeX text$"))
# i = 0
# plt.text(x, y, eqs[i], ha="center", va="center", color="#11557c", fontsize=16, clip_on=True)
# plt.xticks(())
# plt.yticks(())
# plt.show()

# Matplotlib tutorials
# https://matplotlib.org/users/pyplot_tutorial.html
# https://matplotlib.org/users/transforms_tutorial.html
# https://matplotlib.org/users/index_text.html
# https://matplotlib.org/users/artists.html