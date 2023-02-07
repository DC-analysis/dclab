"""Viscosity models for Young's modulus estimation

This example visualizes the different viscosity models for the
MC-PBS media implemented in dclab. We reproduce the lower left part of
figure 3 in :cite:`Reichel2023` (channel width is 20 µm).

"""
import matplotlib.pylab as plt
import matplotlib.lines as mlines
from matplotlib import cm
import numpy as np

from dclab.features.emodulus import viscosity


visc_res = {}

for medium in ["0.49% MC-PBS", "0.59% MC-PBS"]:
    visc_her = {}
    visc_buy = {}

    kwargs = {
        "medium": medium,
        "channel_width": 20.0,
        "temperature": np.linspace(19, 37, 100, endpoint=True),
    }

    flow_rate = np.arange(0.02, 0.13, 0.02)

    for fr in flow_rate:
        visc_her[fr] = viscosity.get_viscosity_mc_pbs_herold_2017(
            flow_rate=fr, **kwargs)
        visc_buy[fr] = viscosity.get_viscosity_mc_pbs_buyukurganci_2022(
            flow_rate=fr, **kwargs)

    visc_res[medium] = [visc_her, visc_buy]


fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey="all", sharex="all")
colors = [cm.get_cmap("viridis")(x) for x in np.linspace(.8, 0,
                                                         len(flow_rate))]

for ii, medium in enumerate(visc_res):
    visc_her, visc_buy = visc_res[medium]
    ax = axes.flatten()[ii]
    ax.set_title(medium)

    for jj, fr in enumerate(flow_rate):
        ax.plot(kwargs["temperature"], visc_her[fr], color=colors[jj], ls="--")
        ax.plot(kwargs["temperature"], visc_buy[fr], color=colors[jj], ls="-")

    ax.set_xlabel("Temperature [°C]")
    ax.set_ylabel("Viscosity [mPa·s]")
    ax.grid()
    ax.set_ylim(2, 12)

handles = []
for jj, fr in enumerate(flow_rate):
    handles.append(
        mlines.Line2D([], [], color=colors[jj], label=f'{fr:.4g} µL/s'))
handles.append(
    mlines.Line2D([], [], color='gray', label='Büyükurgancı 2022'))
handles.append(
    mlines.Line2D([], [], color='gray', ls="--", label='Herold 2017'))
axes[0].legend(handles=handles)


plt.tight_layout()
plt.show()
