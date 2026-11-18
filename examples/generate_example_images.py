import os
import os.path as op
import pathlib
import sys

import matplotlib.pylab as plt

thisdir = op.dirname(op.abspath(__file__))
sys.path.insert(0, op.dirname(thisdir))

DPI = 80


if __name__ == "__main__":
    # Do not display example plots
    plt.show = lambda: None
    files = os.listdir(thisdir)
    files = [f for f in files if f.endswith(".py")]
    files = [f for f in files if f != op.basename(__file__)]
    files = sorted([op.join(thisdir, f) for f in files])

    for f in files:
        fname = f[:-3] + ".jpg"
        if not op.exists(fname):
            exec_str = pathlib.Path(f).read_text()
            if exec_str.count("plt.show()"):
                # run the example script to create the image
                exec(exec_str.replace("plt.show()", ""))  # noqa: S102
                plt.savefig(fname, dpi=DPI)
                print(f"Image created: '{fname}'")
            else:
                print(f"No image: '{fname}'")
        else:
            print(f"Image skipped (already exists): '{fname}'")
        plt.close()
