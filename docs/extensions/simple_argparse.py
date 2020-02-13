"""Very basic docstrings for argparse.ArgumentParser

Usage
-----
To document a command, simply pass the location of a function that
returns an argparse.ArgumentParser instance.

   .. simple_argparse::
       :module: module.submodule
       :func: returns_argument_parser
       :prog: command-line-tool-name

Features:
- prints parameters as table instead of adding a new heading
- no additional modules needed (e.g. CommonMark)

Downsides:
- not as many functionalities as 'sphinxarg.ext'
- makes use of private function argparse.ArgumentParser._actions

Changelog:
0.3 (2020-02-14)
 - change "positional" to "required" arguments
 - check the "required" property to include keyword arguements
   that are required
0.2 (2019-07-12)
 - remove tables, use bullet points (improve readability for long
   docstrings)
 - correct usage of "metavar" instead of "dest"
0.1
 - good for short docstrings
"""
import importlib

from docutils.statemachine import ViewList
from docutils.parsers.rst import Directive, directives
from sphinx.util.nodes import nested_parse_with_titles
from docutils import nodes


class ArgparseDirective(Directive):
    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = True

    option_spec = {
        'module': directives.unchanged,
        'func': directives.unchanged,
        'prog': directives.unchanged,
    }

    def generate_rst(self):
        rst = []

        # get parser instance
        parser = self.get_parser()

        # description
        rst.append(parser.description)
        rst.append("")

        # usage
        usage = parser.format_usage()

        rst.append(".. code::")
        rst.append("")
        rst.append("    {}".format(usage))
        rst.append("")

        # parameters
        pos, opt = self.get_actions()

        if pos:
            rst.append("")
            rst.append("**required arguments:**")
            rst.append("")
            for item in pos:
                rst.append("- ``{}``".format(item[0]))
                rst.append("  {}".format(item[1]))
            rst.append("")

        if opt:
            rst.append("")
            rst.append("**optional arguments:**")
            rst.append("")
            for item in opt:
                if isinstance(item[3], bool):
                    if item[3]:
                        default = " *(enabled by default)*"
                    else:
                        default = " *(disabled by default)*"
                elif item[3] != "":
                    default = " *(default: {})*".format(item[3])
                else:
                    default = ""
                rst.append("- ``{}`` {} ".format(item[1], default))
                rst.append("  {}".format(item[2]))
            rst.append("")

        return rst

    def get_parser(self):
        """This will import the module and get the parser"""
        mod = importlib.import_module(self.options["module"])
        func = getattr(mod, self.options["func"])
        parser = func()
        parser.prog = self.options["prog"]
        return parser

    def get_actions(self):
        parser = self.get_parser()
        actions = parser._actions[1:]  # ignore help action
        positional = []
        optional = []
        for ac in actions:
            if ac.option_strings and not ac.required:
                optional.append([
                    ac.metavar,
                    ", ".join(ac.option_strings),
                    ac.help,
                    ac.default if ac.default is not None else "",
                ])
            else:
                positional.append([
                    ac.metavar,
                    ac.help,
                ])
        return positional, optional

    def run(self):
        rst = self.generate_rst()

        vl = ViewList(rst, "fakefile.rst")
        # Create a node.
        node = nodes.section()
        node.document = self.state.document
        # Parse the rst.
        nested_parse_with_titles(self.state, vl, node)
        return node.children


def setup(app):
    app.add_directive('simple_argparse', ArgparseDirective)
    return {'version': '0.3'}   # identifies the version of our extension
