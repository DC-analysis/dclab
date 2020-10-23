

class PipelineWarning(UserWarning):
    """Super-class for warnings relevant to data analysis

    There are those types of warnings in dclab that are
    important to the user, because they suggest that the
    user may not use the correct model (e.g. Young's modulus
    computation) in his analysis pipeline. All of these
    warnings should be subclassed from PipelineWarning
    to allow identifying them in higher-level software
    such as Shape-Out and to present them correctly to the
    user.
    """
    pass
