import logging
import os
import pathlib
import shutil
import subprocess as sp

logger = logging.getLogger(__name__)

_has_lme4 = None
_has_r = None


class CommandFailedError(BaseException):
    """Used when `run_command` encounters an error"""
    pass


class RNotFoundError(BaseException):
    pass


def get_r_path():
    """Return the path of the R executable"""
    # Maybe the user set the executable already?
    r_exec = os.environ.get("R_EXEC")
    if r_exec is not None:
        r_exec = pathlib.Path(r_exec)
        if r_exec.is_file():
            return r_exec

    # Try to get the executable using which
    r_exec = shutil.which("R")
    if r_exec is not None:
        r_exec = pathlib.Path(r_exec)
        return r_exec

    # Try to determine the path to the executable from R_HOME
    r_home = os.environ.get("R_HOME")
    if r_home and not pathlib.Path(r_home).is_dir():
        logger.warning(f"R_HOME Directory does not exist: {r_home}")
        r_home = None

    if r_home is None:
        raise RNotFoundError(
            "Cannot find R, please set the `R_HOME` environment variable "
            "or use `set_r_path`.")

    r_home = pathlib.Path(r_home)

    # search for the R executable
    for rr in [
        r_home / "bin" / "R",
        r_home / "bin" / "x64" / "R",
    ]:
        if rr.is_file():
            return rr
        rr_win = rr.with_name("R.exe")
        if rr_win.is_file():
            return rr_win
    else:
        raise RNotFoundError(
            f"Could not find R binary in '{r_home}'")


def get_r_script_path():
    """Return the path to the Rscript executable"""
    return get_r_path().with_name("Rscript")


def get_r_version():
    """Return the full R version string"""
    require_r()
    cmd = (str(get_r_path()), "--version")
    logger.debug(f"Looking for R version with: {' '.join(cmd)}")
    r_version = run_command(
        cmd,
        env={"R_LIBS_USER": os.environ.get("R_LIBS_USER", "")},
    )
    r_version = r_version.split(os.linesep)
    if r_version[0].startswith("WARNING"):
        r_version = r_version[1]
    else:
        r_version = r_version[0]
    logger.info(f"R version found: {r_version}")
    # get the actual version string
    if r_version.startswith("R version "):
        r_version = r_version.split(" ", 2)[2]
    return r_version.strip()


def has_lme4():
    """Return True if the lme4 package is installed"""
    global _has_lme4
    if _has_lme4:
        return True
    require_r()
    for pkg in ["lme4", "statmod", "nloptr"]:
        res = run_command(
            (str(get_r_path()), "-q", "-e", f"system.file(package='{pkg}')"),
            env={"R_LIBS_USER": os.environ.get("R_LIBS_USER", "")},
            )
        if not res.split("[1]")[1].count(pkg):
            avail = False
            break
    else:
        avail = _has_lme4 = True
    return avail


def has_r():
    """Return True if R is available"""
    global _has_r
    if _has_r:
        return True
    try:
        hasr = get_r_path().is_file()
    except RNotFoundError:
        hasr = False
    if hasr:
        _has_r = True
    return hasr


def require_lme4():
    """Install the lme4 package (if not already installed)

    Besides ``lme4``, this also installs ``nloptr`` and ``statmod``.
    The packages are installed to the user data directory
    given in :const:`lib_path` from the http://cran.rstudio.org mirror.
    """
    install_command = ("install.packages("
                       "c('statmod','nloptr','lme4'),"
                       "repos='http://cran.rstudio.org'"
                       ")"
                       )
    require_r()
    if not has_lme4():
        run_command(cmd=(get_r_path(), "-e", install_command),
                    env={"R_LIBS_USER": os.environ.get("R_LIBS_USER", "")},
                    )


def require_r():
    """Make sure R is installed an R HOME is set"""
    if not has_r():
        raise RNotFoundError("Cannot find R, please set its path with the "
                             "`set_r_path` function or set the `RHOME` "
                             "environment variable.")


def run_command(cmd, **kwargs):
    """Run a command via subprocess"""
    if hasattr(sp, "STARTUPINFO"):
        # On Windows, subprocess calls will pop up a command window by
        # default when run from Pyinstaller with the ``--noconsole``
        # option. Avoid this distraction.
        si = sp.STARTUPINFO()
        si.dwFlags |= sp.STARTF_USESHOWWINDOW
        # Windows doesn't search the path by default. Pass it an
        # environment so it will.
        env = os.environ
    else:
        si = None
        env = None

    kwargs.setdefault("text", True)
    kwargs.setdefault("stderr", sp.STDOUT)
    if env is not None:
        if "env" in kwargs:
            env.update(kwargs.pop("env"))
        kwargs["env"] = env
    kwargs["startupinfo"] = si

    # Convert paths to strings
    cmd = [str(cc) for cc in cmd]

    try:
        tmp = sp.check_output(cmd, **kwargs)
    except sp.CalledProcessError as e:
        raise CommandFailedError(f"The command '{' '.join(cmd)}' failed with "
                                 f"exit code {e.returncode}: {e.output}")

    return tmp.strip()


def set_r_lib_path(r_lib_path):
    """Add given directory to the R_LIBS_USER environment variable"""
    paths = os.environ.get("R_LIBS_USER", "").split(os.pathsep)
    paths = [p for p in paths if p]
    paths.append(str(r_lib_path).strip())
    os.environ["R_LIBS_USER"] = os.pathsep.join(list(set(paths)))


def set_r_path(r_path):
    """Set the path of the R executable/binary"""
    tmp = run_command((str(r_path), "RHOME"))

    r_home = tmp.split(os.linesep)
    if r_home[0].startswith("WARNING"):
        res = r_home[1]
    else:
        res = r_home[0].strip()
    os.environ["R_HOME"] = res
    os.environ["R_EXEC"] = str(pathlib.Path(r_path).resolve())
