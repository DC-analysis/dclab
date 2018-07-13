"""visualization of dclab definitions

Usage
-----
Directives:

List of features with arguments "all", "scalar", or "non-scalar"

   .. dclab_features::

List of configuration keys with arguments "analysis", "metadata", or a
configuration key section (e.g. "experiment").

   .. dclab_config::

"""
from docutils.statemachine import ViewList
from docutils.parsers.rst import Directive
from sphinx.util.nodes import nested_parse_with_titles
from docutils import nodes

from dclab import definitions as dfn


class Base(Directive):
    required_arguments = 1
    optional_arguments = 0

    def generate_rst(self):
        pass        

    def run(self):
        rst = self.generate_rst()

        vl = ViewList(rst, "fakefile.rst")
        # Create a node.
        node = nodes.section()
        node.document = self.state.document
        # Parse the rst.
        nested_parse_with_titles(self.state, vl, node)
        return node.children


class Config(Base):
    def generate_rst(self):
        which = self.arguments[0]
        rst = []

        if which == "analysis":
            cfg = dfn.CFG_ANALYSIS
        elif which == "metadata":
            cfg = dfn.CFG_METADATA
        else:
            cfg = dfn.CFG_ANALYSIS.copy()
            cfg.update(dfn.CFG_METADATA)
            cfg = {which: cfg[which]}

        for key in sorted(cfg.keys()):
            rst.append("")
            rst.append(".. csv-table::")
            rst.append("    :header: {}, parsed, description [units]".format(key))
            rst.append("    :widths: 30, 10, 60")
            rst.append("    :delim: tab")
            rst.append("")

            
            for item in sorted(cfg[key]):
                if item[1] is str:
                    ref = ":class:`str`"
                elif item[1] is float:
                    ref = ":class:`float`"
                else:
                    ref = ":func:`{f} <dclab.parse_funcs.{f}>`".format(f=item[1].__name__)
                rst.append("    {}\t {}\t {}".format(item[0],
                                                     ref,
                                                     item[2]))
            
            rst.append("")

        return rst


class Features(Base):
    def generate_rst(self):
        which = self.arguments[0]
        rst = []

        if which == "all":
            feats = sorted(dfn.FEATURES_SCALAR + dfn.FEATURES_NON_SCALAR)
        elif which == "scalar":
            feats = sorted(dfn.FEATURES_SCALAR)
        elif which == "non-scalar":
            feats = sorted(dfn.FEATURES_NON_SCALAR)    

        rst.append(".. csv-table::")
        rst.append("    :header: {} features, description [units]".format(which))
        rst.append("    :widths: 2, 7")
        rst.append("    :delim: tab")
        rst.append("")

        for item in feats:
            rst.append("    {}\t {}".format(item[0], item[1]))
        
        rst.append("")

        return rst


def setup(app):
    app.add_directive('dclab_features', Features)
    app.add_directive('dclab_config', Config)
    return {'version': '0.1'}   # identifies the version of our extension
