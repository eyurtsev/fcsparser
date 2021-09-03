from __future__ import print_function

import os
import timeit
import unittest

import numpy
import pytest
from numpy import array

from .. import parse as parse_fcs
from ..api import FCSParser

HERE = os.path.abspath(os.path.dirname(__file__))
BASE_PATH = os.path.join(HERE, 'data', 'FlowCytometers')

# Used for checking data segments in fcs files generated by different machines.
FILE_IDENTIFIER_TO_PATH = {
    'mq fcs 2.0': os.path.join(BASE_PATH, 'MiltenyiBiotec', 'FCS2.0',
                               'EY_2013-07-19_PBS_FCS_2.0_Custom_Without_Add_Well_A1.001.fcs'),
    'mq fcs 3.0': os.path.join(BASE_PATH, 'MiltenyiBiotec', 'FCS3.0',
                               'FCS3.0_Custom_Compatible.fcs'),
    'mq fcs 3.1': os.path.join(BASE_PATH, 'MiltenyiBiotec', 'FCS3.1',
                               'EY_2013-07-19_PBS_FCS_3.1_Well_A1.001.fcs'),
    'LSR II fcs 3.0': os.path.join(BASE_PATH, 'HTS_BD_LSR-II',
                                   'HTS_BD_LSR_II_Mixed_Specimen_001_D6_D06.fcs'),
    'Fortessa fcs 3.0': os.path.join(BASE_PATH, 'Fortessa',
                                     'FCS_3.0_Fortessa_PBS_Specimen_001_A1_A01.fcs'),
    'large fake fcs': os.path.join(BASE_PATH, 'fake_large_fcs', 'fake_large_fcs.fcs'),
    'cyflow cube 8': os.path.join(BASE_PATH, 'cyflow_cube_8', 'cyflow_cube_8.fcs'),
    'fake bitmask error': os.path.join(BASE_PATH, 'fake_bitmask_error', 'fcs1_cleaned.lmd'),
    'Cytek xP5': os.path.join(BASE_PATH, 'Cytek_xP5', 'Cytek_xP5.fcs'),
}

# The group of files below is used for checking behavior other than reading data.
ADDITIONAL_FILE_MAPPING = {
    'duplicate_names': os.path.join(BASE_PATH, 'MiltenyiBiotec', 'FCS3.1',
                                    'SG_2014-09-26_Duplicate_Names.fcs'),
    'corrupted_file': os.path.join(BASE_PATH, 'corrupted', 'corrupted.fcs'),
    'test_channel_naming': os.path.join(BASE_PATH, 'MiltenyiBiotec', 'FCS3.1',
                                        'EY_2013-07-19_PBS_FCS_3.1_Well_A1.001.fcs')
}


def check_data_segment(file_name, expected_array_values):
    """Check that the data segmented extracted from the file corresponds to the expected values."""
    file_path = FILE_IDENTIFIER_TO_PATH[file_name]
    meta, df = parse_fcs(file_path)
    diff = numpy.abs(expected_array_values - df.values[0:4, :])
    return numpy.all(diff < 10 ** -8)  # Is this the proper way to do the test?


class TestFCSReader(unittest.TestCase):
    def test_mq_FCS_2_0_text_segment(self):
        """Test TEXT segment parsed from FCS (2.0 format) file from a MACSQuant flow cytometer."""
        fname = FILE_IDENTIFIER_TO_PATH['mq fcs 2.0']
        meta = parse_fcs(fname, meta_data_only=True)
        self.assertEqual('EY_2013-07-19_PBS_FCS_2.0_Custom_Without_Add_Well_A1.001.fcs',
                         meta['$FIL'])
        self.assertEqual('MACSQuant', meta['$CYT'])

    def test_mq_FCS_3_0_text_segment(self):
        """Test TEXT segment parsed from FCS (3.0 format) file from a MACSQuant flow cytometer."""
        fname = FILE_IDENTIFIER_TO_PATH['mq fcs 3.0']
        meta = parse_fcs(fname, meta_data_only=True)

        expected_fname = 'EY_2013-07-19_PID_101_MG1655_Transformants_D01_Well_A4.001.fcs'
        self.assertEqual(expected_fname, meta['$FIL'])
        self.assertEqual('MACSQuant', meta['$CYT'])

    def test_mq_FCS_3_1_text_segment(self):
        """Test TEXT segment parsed from FCS (3.1 format) file from a MACSQuant flow cytometer."""
        fname = FILE_IDENTIFIER_TO_PATH['mq fcs 3.1']
        meta = parse_fcs(fname, meta_data_only=True)
        self.assertEqual('MACSQuant', meta['$CYT'])

    def test_repeated_delimiter_text_segment(self):
        parser = FCSParser()
        raw_text = '/flow_speed/3 m//s/x/a///y/b/////'
        text = parser._extract_text_dict(raw_text)
        self.assertDictEqual(text, {'flow_speed': '3 m/s', 'x': 'a/', 'y': 'b//'})

    def test_mq_FCS_2_0_data_segment(self):
        """Test DATA segment parsed from FCS (2.0 format) file from a MACSQuant flow cytometer"""
        values = array([[1.60764902830123901367e-03, 1.46554875373840332031e+00,
                         2.03116130828857421875e+00, 3.60766235351562500000e+02,
                         1.57965099811553955078e+00, 1.90790867805480957031e+00,
                         4.13974487304687500000e+02, -3.39318931102752685547e-01,
                         7.84086048603057861328e-01, -2.16378631591796875000e+02,
                         2.23477900028228759766e-01, 5.51754534244537353516e-01,
                         2.02515670776367187500e+02, -2.45075762271881103516e-01,
                         7.51648128032684326172e-01, -1.64115936279296875000e+02],
                        [1.73421576619148254395e-03, 2.02872133255004882812e+00,
                         1.45744156837463378906e+00, 6.95987243652343750000e+02,
                         2.53075432777404785156e+00, 2.87364006042480468750e+00,
                         4.40339477539062500000e+02, -4.55530643463134765625e-01,
                         5.64715683460235595703e-01, -4.03327392578125000000e+02,
                         4.40459489822387695312e-01, 6.26768946647644042969e-01,
                         3.51373107910156250000e+02, 9.75054726004600524902e-02,
                         8.04706454277038574219e-01, 5.41634635925292968750e+01],
                        [2.33519822359085083008e-03, 1.64427363872528076172e+00,
                         1.39029204845428466797e+00, 5.91341064453125000000e+02,
                         3.27396273612976074219e+00, 3.31171727180480957031e+00,
                         4.94299865722656250000e+02, 6.40093326568603515625e-01,
                         8.14603626728057861328e-01, 3.92886383056640625000e+02,
                         -5.28994854539632797241e-03, 2.77096331119537353516e-01,
                         -9.54532432556152343750e+00, 9.36367154121398925781e-01,
                         1.11724865436553955078e+00, 4.18674407958984375000e+02],
                        [3.37790674529969692230e-03, 1.01516211032867431641e+00,
                         1.84805572032928466797e+00, 2.74656768798828125000e+02,
                         1.72263097763061523438e+00, 1.75532078742980957031e+00,
                         4.90688354492187500000e+02, -2.24935024976730346680e-01,
                         6.62015736103057861328e-01, -1.69886459350585937500e+02,
                         5.19020110368728637695e-02, 3.99166643619537353516e-01,
                         6.50129623413085937500e+01, 3.03109139204025268555e-01,
                         8.76159846782684326172e-01, 1.67871810913085937500e+02]])

        self.assertTrue(check_data_segment('mq fcs 2.0', values))

    def test_mq_FCS_3_0_data_segment(self):
        """Test DATA segment parsed from FCS (3.0 format) file from a MACSQuant flow cytometer"""
        values = array([[4.99655876159667968750e+01, -1.78884857177734375000e+02,
                         3.53545867919921875000e+02, -6.63189687500000000000e+04,
                         1.53373974609375000000e+03, 1.71934411621093750000e+03,
                         1.16922687500000000000e+05, -1.85308218002319335938e+00,
                         1.55241485595703125000e+02, -1.56457653808593750000e+03,
                         7.68178787231445312500e+01, 1.61987319946289062500e+02,
                         6.21571679687500000000e+04, 3.74284629821777343750e+01,
                         1.23476585388183593750e+02, 3.87178945312500000000e+04],
                        [7.65274276733398437500e+01, 8.52657958984375000000e+02,
                         8.35975280761718750000e+02, 1.33687671875000000000e+05,
                         2.55060937500000000000e+03, 2.26837988281250000000e+03,
                         1.47379843750000000000e+05, 4.53825866699218750000e+02,
                         4.73883544921875000000e+02, 1.25524226562500000000e+05,
                         -8.64693832397460937500e+00, 9.53993377685546875000e+01,
                         -1.18802871093750000000e+04, 8.17031021118164062500e+01,
                         2.03511352539062500000e+02, 5.58651601562500000000e+04],
                        [8.48738250732421875000e+01, 1.49076705932617187500e+02,
                         2.49545867919921875000e+02, 7.83013671875000000000e+04,
                         6.49180541992187500000e+02, 5.83344177246093750000e+02,
                         1.45864812500000000000e+05, 1.76183197021484375000e+02,
                         2.59241485595703125000e+02, 8.90778906250000000000e+04,
                         1.03054801940917968750e+02, 1.69987319946289062500e+02,
                         7.94623906250000000000e+04, -6.35836944580078125000e+01,
                         1.13396583557128906250e+02, -6.63863593750000000000e+04],
                        [1.02074172973632812500e+02, 1.37832305908203125000e+02,
                         3.85545867919921875000e+02, 4.68581250000000000000e+04,
                         1.88981237792968750000e+03, 1.81534411621093750000e+03,
                         1.36448781250000000000e+05, 3.93574859619140625000e+02,
                         4.83241485595703125000e+02, 1.06751273437500000000e+05,
                         6.74475479125976562500e+01, 1.77987319946289062500e+02,
                         4.96691835937500000000e+04, -3.04502067565917968750e+01,
                         2.20916580200195312500e+02, -1.28346718750000000000e+04]])

        self.assertTrue(check_data_segment('mq fcs 3.0', values))

    def test_BD_LSR_II(self):
        """Test DATA segment parsed from FCS (3.0 format) file from HTS BD LSR-II flow cytometer"""
        values = array([[-2.85312500000000000000e+04, 1.00000000000000000000e+01,
                         0.00000000000000000000e+00, 7.00149963378906250000e+02,
                         1.65600000000000000000e+03, 2.77083515625000000000e+04,
                         9.87999954223632812500e+01, 5.41499977111816406250e+01,
                         1.64220001220703125000e+02, 1.20360000610351562500e+02,
                         2.00000002980232238770e-01],
                        [-4.94148789062500000000e+04, 8.00000000000000000000e+00,
                         0.00000000000000000000e+00, 1.27584997558593750000e+03,
                         2.27800000000000000000e+03, 3.67050507812500000000e+04,
                         1.55800003051757812500e+02, 1.33000001907348632812e+01,
                         1.61840011596679687500e+02, 9.48600006103515625000e+01,
                         4.00000005960464477539e-01],
                        [-5.86843203125000000000e+04, 1.40000000000000000000e+01,
                         0.00000000000000000000e+00, -5.12049987792968750000e+02,
                         4.72000000000000000000e+02, 0.00000000000000000000e+00,
                         2.27999992370605468750e+01, 8.55000019073486328125e+00,
                         1.72550003051757812500e+02, 8.56800003051757812500e+01,
                         5.00000000000000000000e-01],
                        [-3.85783984375000000000e+03, 4.32000000000000000000e+02,
                         0.00000000000000000000e+00, 2.76449981689453125000e+02,
                         1.33900000000000000000e+03, 1.35305644531250000000e+04,
                         -4.93999977111816406250e+01, 3.42000007629394531250e+01,
                         1.57080001831054687500e+02, 8.97599945068359375000e+01,
                         6.99999988079071044922e-01]])
        self.assertTrue(check_data_segment('LSR II fcs 3.0', values))

    def test_Fortessa_data_segment(self):
        """Test DATA segment parsed from FCS (3.0 format) file from a Fortessa flow cytometer."""
        values = array([[1.31284997558593750000e+03, 5.60000000000000000000e+02,
                         1.53640968750000000000e+05, 1.47263989257812500000e+03,
                         1.42400000000000000000e+03, 6.77745312500000000000e+04,
                         1.79399986267089843750e+01, 8.57999992370605468750e+00,
                         1.37059997558593750000e+02, -3.67200012207031250000e+01,
                         0.00000000000000000000e+00],
                        [9.15529968261718750000e+02, 2.97000000000000000000e+02,
                         2.02020781250000000000e+05, 3.24479980468750000000e+02,
                         3.49000000000000000000e+02, 6.09315781250000000000e+04,
                         2.33999996185302734375e+01, 7.79999971389770507812e+00,
                         1.65550003051757812500e+02, 2.01599998474121093750e+01,
                         0.00000000000000000000e+00],
                        [2.27150000000000000000e+03, 5.49000000000000000000e+02,
                         2.62143000000000000000e+05, 8.54099975585937500000e+02,
                         8.65000000000000000000e+02, 6.47101640625000000000e+04,
                         2.49599990844726562500e+01, 2.65199985504150390625e+01,
                         5.77500000000000000000e+01, 1.36800003051757812500e+01,
                         1.00000001490116119385e-01],
                        [2.33232983398437500000e+03, 5.83000000000000000000e+02,
                         2.62143000000000000000e+05, 6.24000000000000000000e+02,
                         6.27000000000000000000e+02, 6.52224296875000000000e+04,
                         2.49599990844726562500e+01, 4.91399993896484375000e+01,
                         7.23799972534179687500e+01, -1.29600009918212890625e+01,
                         1.00000001490116119385e-01]])
        self.assertTrue(check_data_segment('Fortessa fcs 3.0', values))

    def test_mq_FCS_3_1_data_segment(self):
        """Test DATA segment parsed from FCS (3.1 format) file from a MACSQuant flow cytometer"""
        fname = FILE_IDENTIFIER_TO_PATH['mq fcs 3.1']
        meta, df = parse_fcs(fname)

    def test_Cytek_xP5(self):
        """Test the 3-byte-wide DATA segment from a Cytek xP5"""
        fname = FILE_IDENTIFIER_TO_PATH['Cytek xP5']
        meta, df = parse_fcs(fname)

    def test_fcs_reader_API(self):
        """Make sure that the API remains consistent."""
        for fname in FILE_IDENTIFIER_TO_PATH.values():
            print('Running file {0}'.format(fname))
            # Invoke the parser in multiple ways to make sure that all invocations run successfully.
            # This is a shallow test that only verifies consistency.
            meta = parse_fcs(fname, meta_data_only=True)
            meta, data_pandas = parse_fcs(fname, meta_data_only=False)
            meta, data_pandas = parse_fcs(fname, meta_data_only=False, reformat_meta=True)
            self.assertIsInstance(meta['_channel_names_'], tuple)
            self.assertGreater(len(meta['_channel_names_']), 0)

    def test_channel_naming_manual(self):
        """Check that channel names correspond to manual setting."""
        pnn_names = ['Time', 'HDR-CE', 'HDR-SE', 'HDR-V', 'FSC-A', 'FSC-H', 'FSC-W',
                     'SSC-A', 'SSC-H', 'SSC-W', 'FL2-A', 'FL2-H', 'FL2-W', 'FL4-A',
                     'FL4-H', 'FL4-W', 'FL7-A', 'FL7-H', 'FL7-W']

        path = ADDITIONAL_FILE_MAPPING['test_channel_naming']
        meta = parse_fcs(path, meta_data_only=True,
                         reformat_meta=True, channel_naming='$PnN')
        channel_names = list(meta['_channel_names_'])

        self.assertListEqual(channel_names, pnn_names)

        # ---  Test with data

        meta, data = parse_fcs(path, meta_data_only=False,
                               reformat_meta=True, channel_naming='$PnN')
        channel_names = list(meta['_channel_names_'])

        self.assertListEqual(channel_names, pnn_names)
        self.assertListEqual(list(data.columns.values), pnn_names)

        # ---  Test with data

        pns_names = ['HDR-T', 'HDR-CE', 'HDR-SE', 'HDR-V', 'FSC-A', 'FSC-H', 'FSC-W',
                     'SSC-A', 'SSC-H', 'SSC-W', 'V2-A', 'V2-H', 'V2-W', 'Y2-A',
                     'Y2-H', 'Y2-W', 'B1-A', 'B1-H', 'B1-W']

        meta, data = parse_fcs(path, meta_data_only=False,
                               reformat_meta=True, channel_naming='$PnS')
        channel_names = list(meta['_channel_names_'])

        self.assertListEqual(channel_names, pns_names)
        self.assertListEqual(list(data.columns.values), pns_names)

    def test_speed_of_reading_fcs_files(self):
        """Test the speed of loading a FCS files"""
        file_path = FILE_IDENTIFIER_TO_PATH['mq fcs 3.1']
        number = 1000

        time = timeit.timeit(
            lambda: parse_fcs(file_path, meta_data_only=True, reformat_meta=False), number=number)

        print('Loading fcs file {0} times with meta_data only without reformatting of '
              'meta takes {1} per loop'.format(time / number, number))

        time = timeit.timeit(
            lambda: parse_fcs(file_path, meta_data_only=True, reformat_meta=True), number=number)
        print('Loading fcs file {0} times with meta_data only with reformatting of '
              'meta takes {1} per loop'.format(time / number, number))

        time = timeit.timeit(
            lambda: parse_fcs(file_path, meta_data_only=False, reformat_meta=False), number=number)

        print('Loading fcs file {0} times both meta and data but without reformatting of '
              'meta takes {1} per loop'.format(time / number, number))

    def test_reading_corrupted_fcs_file(self):
        """Raise exception when reading a corrupted fcs file."""
        file_path = ADDITIONAL_FILE_MAPPING['corrupted_file']

        # This is raising the wrong error unfortunately. (Should raise a ParseError.)
        # Will require major version bump to fix API.
        with self.assertRaises(ValueError):
            parse_fcs(file_path)

    def test_reading_large_fcs_file(self):
        """Find data segment of a large FCS file."""
        values = array([[1.31284998e+03, 5.60000000e+02, 1.53640969e+05,
                         1.47263989e+03, 1.42400000e+03, 6.77745312e+04,
                         1.79399986e+01, 8.57999992e+00, 1.37059998e+02,
                         -3.67200012e+01, 0.00000000e+00],
                        [9.15529968e+02, 2.97000000e+02, 2.02020781e+05,
                         3.24479980e+02, 3.49000000e+02, 6.09315781e+04,
                         2.33999996e+01, 7.79999971e+00, 1.65550003e+02,
                         2.01599998e+01, 0.00000000e+00],
                        [2.27150000e+03, 5.49000000e+02, 2.62143000e+05,
                         8.54099976e+02, 8.65000000e+02, 6.47101641e+04,
                         2.49599991e+01, 2.65199986e+01, 5.77500000e+01,
                         1.36800003e+01, 1.00000001e-01],
                        [2.33232983e+03, 5.83000000e+02, 2.62143000e+05,
                         6.24000000e+02, 6.27000000e+02, 6.52224297e+04,
                         2.49599991e+01, 4.91399994e+01, 7.23799973e+01,
                         -1.29600010e+01, 1.00000001e-01]], dtype=numpy.float32)
        self.assertTrue(check_data_segment('large fake fcs', values))

    def test_reading_bitmask_error(self):
        """Check correct bitmasking of partial Integer fields in data segment."""
        values = array([[5.28000000e+2, 5.28000000e+2, 5.28000000e+2,
                         5.28000000e+2, 5.28000000e+2, 5.28000000e+2,
                         5.28000000e+2],
                        [5.28000000e+2, 5.28000000e+2, 5.28000000e+2,
                         5.28000000e+2, 5.28000000e+2, 5.28000000e+2,
                         5.28000000e+2],
                        [5.28000000e+2, 5.28000000e+2, 5.28000000e+2,
                         5.28000000e+2, 5.28000000e+2, 5.28000000e+2,
                         5.28000000e+2],
                        [5.28000000e+2, 5.28000000e+2, 5.28000000e+2,
                         5.28000000e+2, 5.28000000e+2, 5.28000000e+2,
                         5.28000000e+2]], dtype=numpy.float32)
        self.assertTrue(check_data_segment('fake bitmask error', values))

    def test_reading_in_memory_fcs_file(self):
        """Find data segment of an in-memory FCS file."""
        values = array([[4.99655876159667968750e+01, -1.78884857177734375000e+02,
                         3.53545867919921875000e+02, -6.63189687500000000000e+04,
                         1.53373974609375000000e+03, 1.71934411621093750000e+03,
                         1.16922687500000000000e+05, -1.85308218002319335938e+00,
                         1.55241485595703125000e+02, -1.56457653808593750000e+03,
                         7.68178787231445312500e+01, 1.61987319946289062500e+02,
                         6.21571679687500000000e+04, 3.74284629821777343750e+01,
                         1.23476585388183593750e+02, 3.87178945312500000000e+04],
                        [7.65274276733398437500e+01, 8.52657958984375000000e+02,
                         8.35975280761718750000e+02, 1.33687671875000000000e+05,
                         2.55060937500000000000e+03, 2.26837988281250000000e+03,
                         1.47379843750000000000e+05, 4.53825866699218750000e+02,
                         4.73883544921875000000e+02, 1.25524226562500000000e+05,
                         -8.64693832397460937500e+00, 9.53993377685546875000e+01,
                         -1.18802871093750000000e+04, 8.17031021118164062500e+01,
                         2.03511352539062500000e+02, 5.58651601562500000000e+04],
                        [8.48738250732421875000e+01, 1.49076705932617187500e+02,
                         2.49545867919921875000e+02, 7.83013671875000000000e+04,
                         6.49180541992187500000e+02, 5.83344177246093750000e+02,
                         1.45864812500000000000e+05, 1.76183197021484375000e+02,
                         2.59241485595703125000e+02, 8.90778906250000000000e+04,
                         1.03054801940917968750e+02, 1.69987319946289062500e+02,
                         7.94623906250000000000e+04, -6.35836944580078125000e+01,
                         1.13396583557128906250e+02, -6.63863593750000000000e+04],
                        [1.02074172973632812500e+02, 1.37832305908203125000e+02,
                         3.85545867919921875000e+02, 4.68581250000000000000e+04,
                         1.88981237792968750000e+03, 1.81534411621093750000e+03,
                         1.36448781250000000000e+05, 3.93574859619140625000e+02,
                         4.83241485595703125000e+02, 1.06751273437500000000e+05,
                         6.74475479125976562500e+01, 1.77987319946289062500e+02,
                         4.96691835937500000000e+04, -3.04502067565917968750e+01,
                         2.20916580200195312500e+02, -1.28346718750000000000e+04]])
        fname = FILE_IDENTIFIER_TO_PATH['mq fcs 3.0']

        with open(fname, 'rb') as f:
            data = f.read()

        matrix = FCSParser.from_data(data).data
        diff = numpy.abs(values - matrix[0:4, :])
        self.assertTrue(numpy.all(diff < 10 ** -8))


# FCS file that contains only the header.
CYTEK_NL_2000_sample_header = os.path.join(BASE_PATH, 'cytek-nl-2000', 'sample_header.fcs')


class TestHeaderParsing(unittest.TestCase):
    def test_parsing_headers(self):
        """Attempt to parse a corrupted header from a cytek-nl-2000 flow cytometer."""
        # Validate that a user warning is raised.
        with pytest.warns(UserWarning):
            fcs_parser = FCSParser(read_data=False, path=CYTEK_NL_2000_sample_header)

        # Check that some of the header has been parsed
        self.assertIsInstance(fcs_parser.annotation, dict)
        # Verify number of keys
        self.assertEqual(200, len(fcs_parser.annotation))
        # Check one hard-coded key
        self.assertEqual(fcs_parser.annotation['$P9B'], 32)
