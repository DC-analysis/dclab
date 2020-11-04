"""Reading and writing trained machine learning models for dclab"""
import atexit
import hashlib
import json
import pathlib
import shutil
import tempfile
import time
import traceback as tb
import warnings

from . import models

#: Supported file formats (including respective model classes).
SUPPORTED_FORMATS = {}
for _md in [models.TensorflowModel]:

    for _fmt in _md.supported_formats():
        SUPPORTED_FORMATS[_fmt["name"]] = {
            "requirements": _fmt["requirements"],
            "suffix": _fmt["suffix"],
            "class": _md}


class ModelFormatExportFailedWarning(UserWarning):
    pass


def export_model(path, model, enforce_formats=None):
    """Export an ML model to all possible formats

    The model must be exportable with at least one method
    listed in :const:`SUPPORTED_FORMATS`.

    Parameters
    ----------
    path: str or pathlib.Path
        Directory where the model is stored to. For each supported
        model, a new subdirectory or file is created.
    model: An instance of an ML model, NOT dclab.ml.models.BaseModel
        Trained model instance
    enforce_formats: list of str
        Enforced file formats for export. If the export for one
        of these file formats fails, a ValueError is raised.
    """
    if enforce_formats is None:
        enforce_formats = []
    path = pathlib.Path(path)
    if not path.is_dir():
        raise ValueError(
            "Output `path` should be a directory: '{}'".format(path))
    if len(sorted(path.rglob("*"))) != 0:
        raise ValueError(
            "Model output directory should be empty: '{}'".format(path))
    if enforce_formats:
        for e_fmt in enforce_formats:
            if e_fmt not in SUPPORTED_FORMATS:
                raise ValueError(
                    "Unsupported format '{}', expected ".format(e_fmt)
                    + "one of {}!".format(", ".join(SUPPORTED_FORMATS.keys())))
    exported_formats = {}
    for fmt in SUPPORTED_FORMATS:
        tmp = tempfile.mkdtemp(prefix="dclab_ml_{}".format(fmt))
        try:
            suffix = SUPPORTED_FORMATS[fmt]["suffix"]
            tmp_out = pathlib.Path(tmp) / (fmt + suffix)
            cls = SUPPORTED_FORMATS[fmt]["class"]
            cls.save_bare_model(tmp_out, model, save_format=fmt)
            # attempt to load the model to see if it worked
            cls.load_bare_model(tmp_out)
        except BaseException:
            warnings.warn("Could not export to '{}': {}".format(
                fmt, tb.format_exc(limit=1)),
                ModelFormatExportFailedWarning)
            if fmt in enforce_formats:
                raise
        else:
            pout = path / tmp_out.name
            pathlib.Path(tmp_out).rename(pout)
            exported_formats[fmt] = tmp_out.name
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    if not exported_formats:
        raise ValueError("Export failed for all model formats!")

    # Now compute the hash and get the destination filename
    m_hash = hash_path(path)
    m_dict = {"sha256": m_hash,
              "formats": exported_formats,
              "date": time.strftime("%Y-%m-%d %H:%M"),
              }
    return m_dict


def hash_path(path):
    """Create a SHA256 hash of a file or all files in a directory

    The files are sorted before hashing for reproducibility.
    """
    path = pathlib.Path(path)
    assert path.is_dir()
    hasher = hashlib.sha256()
    if path.is_dir():
        paths = sorted(path.rglob("*"))
    else:
        paths = [path]
    for pp in paths:
        hasher.update(pp.name.encode())
        if pp.is_dir():
            continue
        with pp.open("rb") as fd:
            while True:
                chunk = fd.read(65536)
                if chunk:
                    hasher.update(chunk)
                else:
                    break
    return hasher.hexdigest()


def load_modc(path, from_format=None):
    """Load models from a .modc file for inference

    Parameters
    ----------
    path: str or path-like
        Path to a .modc file
    from_format: str
        If set to None, the first available format in
        :const:`SUPPORTED_FORMATS` is used. If set to
        a key in :const:`SUPPORTED_FORMATS`, then this
        format will take precedence and an error will
        be raised if loading with this format fails.

    Returns
    -------
    model: dclab.ml.models.BaseModel
        Model that can be used for inference via `model.predict`
    """
    # unpack everything
    t_dir = pathlib.Path(tempfile.mkdtemp(prefix="modc_load_"))
    cleanup = atexit.register(lambda: shutil.rmtree(t_dir, ignore_errors=True))
    shutil.unpack_archive(path, t_dir, format="zip")

    # Get the metadata
    with (t_dir / "index.json").open("r") as fd:
        meta = json.load(fd)

    assert meta["model count"] == len(meta["models"])
    for model_dict in meta["models"]:
        mpath = t_dir / model_dict["path"]

        formats = list(model_dict["formats"].keys())
        if from_format:
            formats = [from_format] + formats

        for fmt in formats:
            if fmt in SUPPORTED_FORMATS:
                cls = SUPPORTED_FORMATS[fmt]["class"]
                load = cls.load_bare_model
                try:
                    bare_model = load(mpath / model_dict["formats"][fmt])
                except BaseException:
                    if from_format and fmt == from_format:
                        # user requested this format explicitly
                        raise
                else:
                    # load `bare_model` into BaseModel
                    model = cls(bare_model=bare_model,
                                inputs=model_dict["input features"],
                                outputs=model_dict["output features"],
                                output_labels=model_dict["output labels"],
                                model_name=model_dict["name"]
                                )
                    break
            elif from_format and fmt == from_format:
                raise ValueError("The format specified via `from_format` "
                                 + " '{}' is not supported!".format(fmt))
        else:
            raise ValueError("No compatible model file format found!")

    # We are nice and do the cleanup before exit
    cleanup()
    atexit.unregister(cleanup)

    return model


def save_modc(path, dc_models):
    """Save ML models to a .modc file

    Parameters
    ----------
    path: str, pathlib.Path
        Output .modc path
    dc_models: list of dclab.ml.models.BaseModel or dclab.ml.models.BaseModel
        Models to save

    Returns
    -------
    meta: dict
        Dictionary written to index.json in the .modc file
    """
    if not isinstance(dc_models, list):
        dc_models = [dc_models]

    # save all models to a temporary directory
    t_dir = pathlib.Path(tempfile.mkdtemp(prefix="modc_save_"))
    cleanup = atexit.register(lambda: shutil.rmtree(t_dir, ignore_errors=True))

    model_data = []
    for ii, mm in enumerate(dc_models):
        p_mod = t_dir / "model_{}".format(ii)
        p_mod.mkdir()
        m_dict = export_model(p_mod, mm.bare_model)
        m_dict["index"] = ii
        m_dict["input features"] = mm.inputs
        m_dict["output features"] = mm.outputs
        m_dict["output labels"] = mm.output_labels
        m_dict["name"] = mm.name
        m_dict["path"] = p_mod.name
        model_data.append(m_dict)

    meta = {
        "model count": len(dc_models),
        "models": model_data,
    }

    # save metadata
    with (t_dir / "index.json").open("w") as fd:
        json.dump(meta, fp=fd, indent=2, separators=(',', ': '),
                  sort_keys=True)

    aout = shutil.make_archive(base_name=path, format="zip", root_dir=t_dir,
                               base_dir=".")
    pathlib.Path(aout).rename(path)

    # We are nice and do the cleanup before exit
    cleanup()
    atexit.unregister(cleanup)

    return meta
