"""Overview plot for the Young's modulus look-up table

The Young's modulus is corrected for pixelation (pixel size 0.34 µm).
The temperature is set to 23°C.
The 20µm plot is used in the dclab documentation. The others
are used in the Shape-Out documentation.
"""
from dclab.features import emodulus
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from matplotlib import cm, colors
from matplotlib.ticker import MultipleLocator

sns.set_style("whitegrid")


lut_name = "LE-2D-FEM-19"
media = ["CellCarrier", "CellCarrier B", "water"]
matrix = {  # channel width: flow rates
    15: [0.016, 0.032, 0.048],
    20: [0.040, 0.080, 0.120],
    30: [0.160, 0.240, 0.320],
    40: [0.320, 0.400, 0.600],
}
deform_max = 0.17
grid_size = 1000
emin = 4e-2
emax = 2e1
norm = colors.LogNorm(vmin=emin, vmax=emax)
cmap = "jet"

for channel_width in matrix:
    flow_rates = matrix[channel_width]
    # setup figure
    fig, axes = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True,
                             gridspec_kw={"left": 0.12,
                                          "right": .85,
                                          "wspace": .1,
                                          "hspace": .15,
                                          },
                             figsize=(7.5, 6.3))
    fig.suptitle("{} µm channel".format(channel_width))
    axes[2][1].set_xlabel("Area [µm²]")
    axes[1][0].set_ylabel("Deformation")
    # column label (flow rate)
    for ii in range(3):
        axes[0][ii].set_title("{} µL/s".format(flow_rates[ii]),
                              fontsize=10, fontweight="bold")
    # row label (medium)
    for jj in range(3):
        pos = axes[jj][0].get_position()
        x = pos.x0 - .11
        y = pos.y0 + pos.height/2
        fig.text(x, y, media[jj], ha='left', va='center', rotation=90,
                 fontsize=10, fontweight="bold")

    # plot the LUTs channel-wise
    lut_base, lut_meta = emodulus.load_lut(lut_name)
    deform = np.linspace(lut_base[:, 1].min(), lut_base[:, 1].max(),
                         grid_size, endpoint=True)
    area_um = np.linspace(lut_base[:, 0].min(), lut_base[:, 0].max(),
                          grid_size, endpoint=True)
    for jj, medium in enumerate(media):  # rows
        for ii, flow_rate in enumerate(matrix[channel_width]):  # columns
            area_um_scaled = emodulus.scale_linear.scale_area_um(
                area_um=area_um,
                channel_width_in=lut_meta["channel_width"],
                channel_width_out=channel_width)
            aa, dd = np.meshgrid(area_um_scaled, deform)
            emod = emodulus.get_emodulus(area_um=aa,
                                         deform=dd,
                                         medium=medium,
                                         channel_width=channel_width,
                                         flow_rate=flow_rate,
                                         px_um=.34,
                                         temperature=23.0,
                                         lut_data=lut_name)
            ax = axes[jj][ii]
            ax.set_aspect("auto")
            pc = ax.pcolormesh(aa, dd, emod, shading="nearest", cmap=cmap,
                               norm=norm, zorder=11)
            ax.grid(zorder=10)
            ax.yaxis.set_major_locator(MultipleLocator(0.05))  # ticks every .5

    axes[2][2].set_ylim(0, axes[2][2].get_ylim()[1] + .03)
    xmin, xmax = axes[2][2].get_xlim()
    axes[2][2].set_xlim(xmin - (xmax-xmin)/30, xmax + (xmax-xmin)/30)
    axes[2][2].grid(True, which="minor", zorder=10)

    # add colorbar
    cax = plt.axes([0.87, 0.1, 0.02, 0.8])

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    plt.colorbar(pc, orientation="vertical", cax=cax, extend="both",
                 label="apparent Young's modulus ({}) [kPa]".format(lut_name))

    plt.savefig("_emodulus_{}um_{}.png".format(channel_width, lut_name),
                dpi=150)
    plt.close()
