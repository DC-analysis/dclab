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

from .mllibs import tensorflow as tf


#: Supported file formats (including instructions on how to open
#: and save them).
SUPPORTED_FORMATS = {
    "tensorflow-SavedModel": {
        "requirements": ["tensorflow"],
        "suffix": ".tf",
        "func:load": lambda path: tf.saved_model.load(str(path)),
        "func:save": lambda path, model: tf.saved_model.save(
            obj=model, export_dir=str(path)),
    }
}


def export_model(path, model, enforce_formats=[]):
    """Export an ML model

    The model must be exportable with at least one method
    listed in :const:`SUPPORTED_FORMATS`.

    Parameters
    ----------
    path: str or pathlib.Path
        Directory where the model is stored to. For each supported
        model, a new subdirectory or file is created.
    model: An instance of an ML model
        Trained model instance
    enforce_formats: list of str
        Enforced file formats for export. If the export for one
        of these file formats fails, a ValueError is raised.
    """
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
        try:
            tmp = tempfile.mkdtemp(prefix="dclab_ml_{}".format(fmt))
            suffix = SUPPORTED_FORMATS[fmt]["suffix"]
            tmp_out = pathlib.Path(tmp) / (fmt + suffix)
            save = SUPPORTED_FORMATS[fmt]["func:save"]
            save(tmp_out, model)
            # attempt to load the model to see if it worked
            load = SUPPORTED_FORMATS[fmt]["func:load"]
            load(tmp_out)
        except BaseException:
            warnings.warn("Could not export to '{}': {}".format(
                fmt, tb.format_exc(limit=1)))
            if fmt in enforce_formats:
                raise
        else:
            pout = path / tmp_out.name
            pathlib.Path(tmp_out).rename(pout)
            exported_formats[fmt] = tmp_out.name
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


def load_modc(path):
    """Load models from a .modc file for inference

    The first available format from :const:`SUPPORTED_FORMATS`
    will be used.
    """
    # unpack everything
    t_dir = pathlib.Path(tempfile.mkdtemp(prefix="modc_load_"))
    cleanup = atexit.register(lambda: shutil.rmtree(t_dir, ignore_errors=True))
    shutil.unpack_archive(path, t_dir, format="zip")

    # Get the metadata
    with (t_dir / "index.json").open("r") as fd:
        meta = json.load(fd)

    assert meta["model count"] == len(meta["models"])
    for model in meta["models"]:
        mpath = t_dir / model["path"]

        for fmt in model["formats"]:
            if fmt in SUPPORTED_FORMATS:
                load = SUPPORTED_FORMATS[fmt]["func:load"]
                try:
                    model = load(mpath / model["formats"][fmt])
                except BaseException:
                    pass
                else:
                    break
        else:
            raise ValueError("No compatible model file format found!")

    # We are nice and do the cleanup before exit
    cleanup()
    atexit.unregister(cleanup)

    return model, meta


def save_modc(path, models, inputs, outputs, model_names=None,
              output_labels=None):
    """Save an ML model to a .modc file

    Parameters
    ----------
    path: str, pathlib.Path
        Output .modc path
    models: list of ML model instances
        Model(s) to save, e.g. ``[tf.keras.Model, tf.keras.Model]``
    inputs: list of str
        List of features for each model in that order, e.g.
        ``[["image", "deform"], ["area_um"]]``
    outputs: list of str
        List of output features the model provides in that order, e.g.
        ``[["ml_score_rbc"], ["ml_score_rt1", "ml_score_tfe"]]``
    model_names: str or None
        The names of the models
    output_labels: list of str
        List of more descriptive labels for the features, e.g.
        ``["red blood cell"]``.

    Returns
    -------
    meta: dict
        Dictionary written to index.json in the .modc file
    """
    if not isinstance(models, list):
        models = [models]
        inputs = [inputs]
        outputs = [outputs]
        if output_labels:
            output_labels = [output_labels]
        if model_names:
            model_names = [model_names]

    # save all models to a temporary directory
    t_dir = pathlib.Path(tempfile.mkdtemp(prefix="modc_save_"))
    cleanup = atexit.register(lambda: shutil.rmtree(t_dir, ignore_errors=True))

    model_data = []
    for ii, mm in enumerate(models):
        p_mod = t_dir / "model_{}".format(ii)
        p_mod.mkdir()
        m_dict = export_model(p_mod, mm)
        m_dict["index"] = ii
        m_dict["input features"] = inputs[ii]
        m_dict["output features"] = outputs[ii]
        if output_labels:
            m_dict["output labels"] = output_labels[ii]
        else:
            m_dict["output labels"] = outputs[ii]
        name = model_names[ii] if model_names else m_dict["sha256"][:6]
        m_dict["name"] = name
        m_dict["path"] = p_mod.name
        model_data.append(m_dict)

    meta = {
        "model count": len(models),
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
