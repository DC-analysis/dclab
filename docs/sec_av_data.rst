===============
Data processing
===============

Visualization
=============
For data visualization, dclab comes with predefined 
:ref:`kernel density estimators (KDEs) <sec_ref_kde_methods>` and
an :ref:`event downsampling <sec_ref_downsampling>` module.
The functionalities of both modules are made available directly via the
:class:`RTDCBase <dclab.rtdc_dataset.RTDCBase>` class.

**Example**: basic KDE scatter plot
    The KDE of the events in a 2D scatter plot can be used to
    colorize events according to event density.

    .. plot::

        import matplotlib.pylab as plt
        import dclab
        ds = dclab.new_dataset("data/example.rtdc")
        kde = ds.get_kde_scatter(xax="area_um", yax="deform")
        
        ax = plt.subplot(111, title="{} events".format(len(kde)))
        sc = ax.scatter(ds["area_um"], ds["deform"], c=kde, marker=".")
        ax.set_xlabel(dclab.dfn.feature_name2label["area_um"])
        ax.set_ylabel(dclab.dfn.feature_name2label["deform"])
        plt.colorbar(sc, label="kernel density estimate [a.u]")
        plt.show()

**Example**: scatter plot with data-density-based downsampling
    To reduce the complexity of the plot (e.g. when exporting to
    scalable vector graphics (.svg)), the plotted events can be
    downsampled by removing events from high-event-density regions. 
    The number of events plotted is reduced but the resulting
    visualization is identical (compare to example above).

    .. plot::

        import matplotlib.pylab as plt
        import dclab
        ds = dclab.new_dataset("data/example.rtdc")
        xsamp, ysamp = ds.get_downsampled_scatter(xax="area_um", yax="deform", downsample=2000)
        kde = ds.get_kde_scatter(xax="area_um", yax="deform", positions=(xsamp, ysamp))

        ax = plt.subplot(111, title="{} events".format(len(kde)))
        sc = ax.scatter(xsamp, ysamp, c=kde, marker=".")
        ax.set_xlabel(dclab.dfn.feature_name2label["area_um"])
        ax.set_ylabel(dclab.dfn.feature_name2label["deform"])
        plt.colorbar(sc, label="kernel density estimate [a.u]")
        plt.show()


In addition, dclab comes with predefined isoelasticity lines that
are commonly used to identify events with similar elastic moduli.
Isoelasticity lines are available via the
:ref:`isoelastics <sec_ref_isoelastics>` module.


**Example**: Adding isoelastics to a scatter plot

    .. plot::

        import matplotlib.pylab as plt
        import dclab
        ds = dclab.new_dataset("data/example.rtdc")
        kde = ds.get_kde_scatter(xax="area_um", yax="deform")

        isodef = dclab.isoelastics.get_default()
        iso = isodef.get_with_rtdcbase(method="numerical",
                                       col1="area_um",
                                       col2="deform",
                                       dataset=ds)

        ax = plt.subplot(111, title="{} events".format(len(kde)))
        for ss in iso:
            ax.plot(ss[:, 0], ss[:, 1], color="gray", zorder=1)
        sc = ax.scatter(ds["area_um"], ds["deform"], c=kde, marker=".", zorder=2)
        ax.set_xlabel(dclab.dfn.feature_name2label["area_um"])
        ax.set_ylabel(dclab.dfn.feature_name2label["deform"])
        ax.set_ylim(0, 0.2)
        plt.colorbar(sc, label="kernel density estimate [a.u]")
        plt.show()


Statistics
==========
The :ref:`sec_ref_statistics` module comes with a predefined set of
methods to compute simple feature statistics. 


Export
======
The :class:`RTDC_Base` class has the attribute :class:`RTDC_Base.export`
which allows to export event data to several data file formats. See
:ref:`sec_ref_rtdc_export` for more information.


ShapeOut
========
Keep in mind that in some cases, it might still be useful to make use
of ShapeOut. For instance, you can create and export polygon filters
in ShapeOut and then import them in dclab.

