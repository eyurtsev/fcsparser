import os

from .api import parse

test_sample_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
    'tests', 'data', 'FlowCytometers', 'HTS_BD_LSR-II', 'HTS_BD_LSR_II_Mixed_Specimen_001_D6_D06.fcs')


from . import _version
__version__ = _version.get_versions()['version']
