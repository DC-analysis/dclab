# python 2+3 compatibility functions
import io
import sys


if sys.version_info[0] == 2:
    str_types = (str, unicode)  # noqa: F821
    hdf5_str = unicode  # noqa: F821

    def is_file_obj(obj):
        return isinstance(obj, file)  # noqa: F821

    def path_to_str(path):
        """This is a heuristic function"""
        try:
            string = str(path)
        except UnicodeDecodeError:
            try:
                string = unicode(path)  # noqa: F821
            except BaseException:
                try:
                    string = unicode(path).encode("utf-8")  # noqa: F821
                except BaseException:
                    string = str(path).decode("utf-8")
        return string

else:
    str_types = str
    hdf5_str = str

    def is_file_obj(obj):
        return isinstance(obj, io.IOBase)

    def path_to_str(path):
        return str(path)
