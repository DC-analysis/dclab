from __future__ import annotations

import logging
from typing import Callable
import warnings
import weakref


logger = logging.getLogger(__name__)


class IgnoringPerishableBasinTTL(UserWarning):
    """Used when refreshing a basin does not support TTL"""


class PerishableRecord:
    """A class containing information about perishable basins

    Perishable basins are basins that may discontinue to work after
    e.g. a specific amount of time (e.g. presigned S3 URLs). With the
    `PerishableRecord`, these basins may be "refreshed" (made
    available again).
    """
    def __init__(self,
                 basin,
                 expiration_func: Callable | None = None,
                 expiration_kwargs: dict | None = None,
                 refresh_func: Callable | None = None,
                 refresh_kwargs: dict | None = None,
                 ):
        """
        Parameters
        ----------
        basin: Basin
            Instance of the perishable basin
        expiration_func: callable
            A function that determines whether the basin has perished.
            It must accept `basin` as the first argument. Calling this
            function should be fast, as it is called every time a feature
            is accessed.
            Note that if you are implementing this in the time domain, then
            you should use `time.time()` (TSE), because you need an absolute
            time measure. `time.monotonic()` for instance does not count up
            when the system goes to sleep. However, keep in mind that if
            a remote machine dictates the expiration time, then that
            remote machine should also transmit the creation time (in case
            there are time offsets).
        expiration_kwargs: dict
            Additional kwargs for `expiration_func`.
        refresh_func: callable
            The function used to refresh the `basin`. It must accept
            `basin` as the first argument.
        refresh_kwargs: dict
            Additional kwargs for `refresh_func`
        """
        if not isinstance(basin, weakref.ProxyType):
            basin = weakref.proxy(basin)
        self.basin = basin
        self.expiration_func = expiration_func
        self.expiration_kwargs = expiration_kwargs or {}
        self.refresh_func = refresh_func
        self.refresh_kwargs = refresh_kwargs or {}

    def __repr__(self):
        state = "perished" if self.perished() else "valid"
        return f"<PerishableRecord ({state}) at {hex(id(self))}>"

    def perished(self) -> bool | None:
        """Determine whether the basin has perished

        Returns
        -------
        state: bool or None
            True means the basin has perished, False means the basin
            has not perished, and `None` means we don't know
        """
        if self.expiration_func is None:
            return None
        else:
            return self.expiration_func(self.basin, **self.expiration_kwargs)

    def refresh(self, extend_by: float | None = None) -> None:
        """Extend the lifetime of the associated perishable basin

        Parameters
        ----------
        extend_by: float
            Custom argument for extending the life of the basin.
            Normally, this would be a lifetime.

        Returns
        -------
        basin: dict | None
            Dictionary for instantiating a new basin
        """
        if self.refresh_func is None:
            # The basin is a perishable basin, but we have no way of
            # refreshing it.
            logger.error(f"Cannot refresh basin '{self.basin}'")
            return

        if extend_by and "extend_by" not in self.refresh_kwargs:
            warnings.warn(
                "Parameter 'extend_by' ignored, because the basin "
                "source does not support it",
                IgnoringPerishableBasinTTL)
            extend_by = None

        rkw = {}
        rkw.update(self.refresh_kwargs)

        if extend_by is not None:
            rkw["extend_by"] = extend_by

        self.refresh_func(self.basin, **rkw)
        logger.info(f"Refreshed basin '{self.basin}'")

        # If everything went well, reset the current dataset of the basin
        if self.basin._ds is not None:
            self.basin._ds.close()
            self.basin._ds = None
