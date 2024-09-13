import logging
import os
import pathlib
import subprocess as sp
import sys

logger = logging.getLogger(__name__)

_has_lme4 = None
_has_r = None


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

    # Try to determine the path to the executable from R_HOME
    r_home = os.environ.get("R_HOME")
    if r_home and not pathlib.Path(r_home).is_dir():
        logger.warning(f"R_HOME Directory does not exist: {r_home}")
        r_home = None
    if r_home is None:
        cmd = ("R", "RHOME")
        try:
            tmp = sp.check_output(cmd, universal_newlines=True)
            # may raise FileNotFoundError, WindowsError, etc
            r_home = tmp.split(os.linesep)
        except BaseException:
            pass
        else:
            if r_home[0].startswith("WARNING"):
                r_home = r_home[1].strip()
            else:
                r_home = r_home[0].strip()
    if r_home is None:
        raise RNotFoundError(
            "Cannot find R, please set the `R_HOME` environment variable "
            "or use `set_r_path`.")

    r_home = pathlib.Path(r_home)

    if sys.platform == "win32" and "64 bit" in sys.version:
        r_exec = r_home / "bin" / "x64" / "R.exe"
    else:
        r_exec = r_home / "bin" / "R"
    if not r_exec.is_file():
        raise RNotFoundError(
            f"Expected R binary at '{r_exec}' does not exist!")
    logger.info(f"R path: {r_exec}")
    return r_exec


def get_r_script_path():
    """Return the path to the Rscript executable"""
    return get_r_path().with_name("Rscript")


def get_r_version():
    """Return the full R version string"""
    require_r()
    cmd = ("R", "--version")
    logger.debug(f"Looking for R version with: {cmd}")
    tmp = sp.check_output(cmd, stderr=sp.STDOUT)
    r_version = tmp.decode("ascii", "ignore").split(os.linesep)
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
        res = run_command(("R", "-q", "-e", f"system.file(package='{pkg}')"))
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
        hasr = bool(get_r_path())
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
    require_r()
    if not has_lme4():
        run_command((
            "R", "-e", "install.packages(c('statmod','nloptr','lme4'),"
                       "repos='http://cran.rstudio.org')"))


def require_r():
    """Make sure R is installed an R HOME is set"""
    if not has_r():
        raise RNotFoundError("Cannot find R, please set its path with the "
                             "`set_r_path` function or set the `RHOME` "
                             "environment variable.")


def run_command(cmd):
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

    # Convert paths to strings
    cmd = [str(cc) for cc in cmd]

    tmp = sp.check_output(cmd,
                          startupinfo=si,
                          env=env,
                          stderr=sp.STDOUT,
                          text=True,
                          )
    return tmp


def set_r_path(r_path):
    """Set the path of the R executable/binary"""
    tmp = run_command((r_path, "RHOME"))

    r_home = tmp.split(os.linesep)
    if r_home[0].startswith("WARNING"):
        res = r_home[1]
    else:
        res = r_home[0].strip()
    os.environ["R_HOME"] = res
    os.environ["R_EXEC"] = str(pathlib.Path(r_path).resolve())
