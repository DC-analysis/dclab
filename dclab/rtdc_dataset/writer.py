import pathlib

import h5py


class RTDCWriter:
    def __init__(self, path_or_h5file, mode="append", compression="gzip"):
        """RT-DC data writer classe

        Parameters
        ----------
        path_or_h5file: str or pathlib.Path or h5py.Group
            Path to an HDF5 file or an HDF5 file opened in write mode
        mode: str
            Defines how the data are stored:
            - "append": append new feature data to existing h5py Datasets
            - "replace": replace existing h5py Datasets with new features
                         (used for ancillary feature storage)
            - "reset": do not keep any previous data
        compression: str
            Compression method used for data storage;
            one of [None, "lzf", "gzip", "szip"].
        """
        assert mode in ["append", "replace", "reset"]
        self.mode = mode
        self.compression = compression
        if isinstance(path_or_h5file, h5py.Group):
            self.path = pathlib.Path(path_or_h5file.file.filename)
            self._h5 = path_or_h5file
        else:
            self.path = pathlib.Path(path_or_h5file)
            self._h5 = h5py.File(path_or_h5file,
                                 mode=("w" if mode == "reset" else "a"))

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        # close the HDF5 file
        self._h5.close()

    def store_log(self, name, lines):
        """Write log data

        Parameters
        ----------
        name: str
            name of the log entry
        lines: list of str
            the text lines of the log
        """
        log_group = self._h5.require_group("logs")
        self.write_text(group=log_group, name=name, lines=lines)

    def write_text(self, group, name, lines):
        """Write text to an HDF5 dataset

        Text data are written as as fixed-length string dataset.

        Parameters
        ----------
        group: h5py.Group
            parent group
        name: str
            name of the dataset containing the text
        lines: list of str
            the text, line by line
        """
        # replace text?
        if name in group and self.mode == "replace":
            del group[name]

        # handle strings
        if isinstance(lines, (str, bytes)):
            lines = [lines]

        lnum = len(lines)
        # Determine the maximum line length and use fixed-length strings,
        # because compression and fletcher32 filters won't work with
        # variable length strings.
        # https://github.com/h5py/h5py/issues/1948
        # 100 is the recommended maximum and the default, because if
        # `mode` is e.g. "append", then this line may not be the longest.
        max_length = 100
        lines_as_bytes = []
        for line in lines:
            # convert lines to bytes
            if not isinstance(line, bytes):
                lbytes = line.encode("UTF-8")
            else:
                lbytes = line
            max_length = max(max_length, len(lbytes))
            lines_as_bytes.append(lbytes)

        if name not in group:
            # Create the dataset
            txt_dset = group.create_dataset(
                name,
                (lnum,),
                dtype=f"S{max_length}",
                maxshape=(None,),
                chunks=True,
                fletcher32=True,
                compression=self.compression)
            line_offset = 0
        else:
            # TODO: test whether fixed length is long enough!
            # Resize the dataset
            txt_dset = group[name]
            line_offset = txt_dset.shape[0]
            txt_dset.resize(line_offset + lnum, axis=0)

        # Write the text data line-by-line
        for ii, lbytes in enumerate(lines_as_bytes):
            txt_dset[line_offset + ii] = lbytes
