import logging
import os
import subprocess as sp

from .rlibs import rpy2, rpy2_is_version_3, import_r_submodules

# Disable rpy2 logger because of unnecessary prints to stdout
logging.getLogger("rpy2.rinterface_lib.callbacks").disabled = True


class RNotFoundError(BaseException):
    pass


class AutoRConsole(object):
    """Helper class for catching R console output"""
    lock = False
    perform_lock = rpy2_is_version_3

    def __init__(self):
        """
        By default, this console always returns "yes" when asked a
        question. If you need something different, you can subclass
        and override `consoleread` fucntion. The console stream is
        recorded in `self.stream`.
        """
        self.stream = [["init", "Starting RConsole class\n"]]
        if AutoRConsole.perform_lock:
            if AutoRConsole.lock:
                raise ValueError("Only one RConsole instance allowed!")
            AutoRConsole.lock = True
            self.original_funcs = {
                "consoleread": rpy2.rinterface_lib.callbacks.consoleread,
                "consolewrite_print":
                    rpy2.rinterface_lib.callbacks.consolewrite_print,
                "consolewrite_warnerror":
                    rpy2.rinterface_lib.callbacks.consolewrite_warnerror,
            }
            rpy2.rinterface_lib.callbacks.consoleread = self.consoleread
            rpy2.rinterface_lib.callbacks.consolewrite_print = \
                self.consolewrite_print
            rpy2.rinterface_lib.callbacks.showmessage = \
                self.consolewrite_print

            rpy2.rinterface_lib.callbacks.consolewrite_warnerror = \
                self.consolewrite_warnerror
        # Set locale (to get always English messages)
        rpy2.robjects.r('Sys.setlocale("LC_MESSAGES", "C")')
        rpy2.robjects.r('Sys.setlocale("LC_CTYPE", "C")')

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if AutoRConsole.perform_lock:
            AutoRConsole.lock = False
            rpy2.rinterface_lib.callbacks.consoleread = \
                self.original_funcs["consoleread"]
            rpy2.rinterface_lib.callbacks.consolewrite_print = \
                self.original_funcs["consolewrite_print"]
            rpy2.rinterface_lib.callbacks.consolewrite_warnerror = \
                self.original_funcs["consolewrite_warnerror"]

    def close(self):
        """Remove the rpy2 monkeypatches"""
        self.__exit__()

    def consoleread(self, prompt):
        """Read user input, returns "yes" by default"""
        self.write_to_stream("consoleread", prompt + "YES")
        return "yes"

    def consolewrite_print(self, s):
        self.write_to_stream("consolewrite_print", s)

    def consolewrite_warnerror(self, s):
        self.write_to_stream("consolewrite_warnerror", s)

    def write_to_stream(self, topic, s):
        prev_topic = self.stream[-1][0]
        same_topic = prev_topic == topic
        unfinished_line = self.stream[-1][1][-1] not in ["\n", "\r"]
        if same_topic and unfinished_line:
            # append to previous line
            self.stream[-1][1] += s
        else:
            self.stream.append([topic, s])

    def get_prints(self):
        prints = []
        for line in self.stream:
            if line[0] == "consolewrite_print":
                prints.append(line[1].strip())
        return prints

    def get_warnerrors(self):
        warnerrors = []
        for line in self.stream:
            if line[0] == "consolewrite_warnerror":
                warnerrors.append(line[1].strip())
        return warnerrors


def check_r():
    """Make sure R is installed an R HOME is set"""
    if not has_r():
        raise RNotFoundError("Cannot find R, please set its path with the "
                             + "`set_r_path` function.")


def get_r_path():
    """Get the path of the R executable/binary from rpy2"""
    r_home = rpy2.situation.get_r_home()
    return rpy2.situation.get_r_exec(r_home)


def get_r_version():
    check_r()
    return rpy2.situation.r_version_from_subprocess()


def has_lme4():
    """Return True if the lme4 package is installed"""
    check_r()
    lme4_there = rpy2.robjects.packages.isinstalled("lme4")
    statmod_there = rpy2.robjects.packages.isinstalled("statmod")
    return lme4_there and statmod_there


def has_r():
    """Return True if R is available"""
    return rpy2.situation.get_r_home() is not None


def import_lme4():
    check_r()
    if has_lme4():
        lme4pkg = rpy2.robjects.packages.importr("lme4")
    else:
        raise ValueError(
            "The R package 'lme4' is not installed, please install it via "
            + "`dclab.lme4.rsetup.install_lme4()` or by executing "
            + "in a shell: R -e " + '"install.packages(' + "'lme4', "
            + "repos='http://cran.r-project.org')" + '"')
    return lme4pkg


def install_lme4():
    """Install the lme4 package (if not already installed)

    The packages are installed to the user data directory
    given in :const:`lib_path`.
    """
    check_r()
    if not has_lme4():
        # import R's utility package
        utils = rpy2.robjects.packages.importr('utils')
        # select the first mirror in the list
        utils.chooseCRANmirror(ind=1)
        # install lme4 to user data directory (say yes to user dir install)
        with AutoRConsole() as rc:
            # install statmod first (Doesn't R have package dependencies?!)
            utils.install_packages(
                rpy2.robjects.vectors.StrVector(["statmod", "lme4"]))
        return rc


def set_r_path(r_path):
    """Set the path of the R executable/binary for rpy2"""
    tmp = sp.check_output((r_path, 'RHOME'), universal_newlines=True)
    r_home = tmp.split(os.linesep)
    if r_home[0].startswith('WARNING'):
        res = r_home[1]
    else:
        res = r_home[0].strip()
    os.environ["R_HOME"] = res
    import_r_submodules()
