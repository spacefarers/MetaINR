import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# Create figure and axis
fig, ax = plt.subplots(figsize=(1, 8))
# plt.rcParams['font.family'] = 'Times New Roman'

# Create vertical line (red)
ax.plot([0, 0], [0, 598], color="red", linewidth=1)

# Create rectangles (blue with black borders)
y_positions = [
    1,
    11,
    24,
    35,
    48,
    60,
    71,
    84,
    95,
    108,
    120,
    131,
    143,
    155,
    168,
    180,
    191,
    202,
    215,
    227,
    239,
    250,
    262,
    275,
    287,
    299,
    309,
    321,
    334,
    347,
    358,
    369,
    381,
    393,
    406,
    418,
    428,
    440,
    453,
    466,
    477,
    487,
    499,
    512,
    524,
    536,
    546,
    559,
    571,
    583,
]
for y in y_positions:
    rect = Rectangle(
        (-0.05, y - 0.01),
        0.05,
        1,
        facecolor="#00b0f0",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.add_patch(rect)

# # Add text for numbers (vertical orientation)
# ax.text(0.1, 0, '1', fontsize=15, va='bottom', rotation=90)
# ax.text(0.1, 600, '598', fontsize=15, va='top', rotation=90)

# Set axis properties
ax.set_xlim(-0.3, 0.2)
ax.set_ylim(0, 600)
ax.axis("off")

# Adjust layout
plt.tight_layout()
plt.savefig("tsne_bar.png", dpi=300)
