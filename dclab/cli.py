"""Command line interface"""
import argparse
import pathlib

from .rtdc_dataset import load


def verify_dataset():
    """Perform checks on experimental data sets"""
    parser = argparse.ArgumentParser(description='dclab dataset checker.')
    parser.add_argument('path', metavar='path', type=str,
                        help='path to dataset')
    args = parser.parse_args()
    path_in = pathlib.Path(args.path).resolve()
    load.check_dataset(path_in)
