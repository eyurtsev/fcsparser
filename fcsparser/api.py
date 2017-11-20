#!/usr/bin/env python
"""
Parser for FCS 2.0, 3.0, 3.1 files. Python 2/3 compatible.
`
Distributed under the MIT License.

Useful documentation for dtypes in numpy
http://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.byteswap.html?highlight=byteswap#numpy.ndarray.byteswap  # noqa
http://docs.scipy.org/doc/numpy/user/basics.types.html
http://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html
"""
from __future__ import division

from io import BytesIO
import string
import sys
import warnings
import contextlib
import logging

import numpy

try:
    import pandas as pd
except ImportError:
    pd = None
    warnings.warn(u'pandas is not installed, so the parse_fcs function can only be used together '
                  u'with numpy.')

logger = logging.getLogger(__name__)


def fromfile(file, dtype, count, *args, **kwargs):
    """Wrapper around np.fromfile to support any file-like object."""
    try:
        return numpy.fromfile(file, dtype=dtype, count=count, *args, **kwargs)
    except (TypeError, IOError):
        return numpy.frombuffer(file.read(count * numpy.dtype(dtype).itemsize),
                                dtype=dtype, count=count, *args, **kwargs)


class ParserFeatureNotImplementedError(Exception):
    """Raise when encountering fcs files for which parsing hasn't been implemented."""
    pass


class FCSParser(object):
    def __init__(self, path=None, read_data=True, channel_naming='$PnS', data_set=0):
        """Parse FCS files.

        Compatible with most FCS 2.0, 3.0, 3.1 files.

        self.annotation: a dictionary holding the parsed content of the TEXT segment
                         In addition, a key called __header__ has been added to this dictionary
                         It specifies the information parsed from the FCS file HEADER segment.
                         (This won't be necessary for most users.)

        self.data holds the parsed DATA segment
        self.analysis holds the ANALYSIS segment as read from the file.

        After the data segment is read:
            self.channel_names holds the chosen names of the channels
            self.channel_names_alternate holds the alternate names of the channels

        Args:
            path : str
                Path of .fcs file
            read_data : bool
                If True, reads the data immediately.
                Otherwise, use read_data method to read in the data from the fcs file.
            channel_naming: '$PnS' | '$PnN'
                Determines which meta data field is used for naming the channels.

                The default should be $PnS (even though it is not guaranteed to be unique)

                $PnN stands for the short name (guaranteed to be unique).
                    Will look like 'FL1-H'
                $PnS stands for the actual name (not guaranteed to be unique).
                    Will look like 'FSC-H' (Forward scatter)

                The chosen field will be used to population self.channels.

                If the chosen field does not exist in the fcs file.
                The program attempts to use the alternative field by default.

                Note: These names are not flipped in the implementation.
                It looks like they were swapped for some reason in the official FCS specification.
            data_set: int
                Index of retrieved data set in the fcs file.
                This value specifies the data set being retrieved from an fcs file with multple data sets.
        """
        self._data = None
        self._channel_naming = channel_naming
        self.channel_names_s = []
        self.channel_names_n = []

        # Attributes parsed from fcs file
        self._data_start = -1
        self._data_end = -1
        self.channel_numbers = []
        self._analysis = None
        self._file_size = 0

        if channel_naming not in ('$PnN', '$PnS'):
            raise ValueError(u'channel_naming must be either "$PnN" or "$PnS"')

        self.annotation = {}

        self.path = path

        if path:
            with open(path, 'rb') as f:
                self.load_file(f, data_set, read_data)

    def load_file(self, file_handle, data_set=0, read_data=True):
        """"""
        file_handle.seek(0, 2)
        self._file_size = file_handle.tell()
        file_handle.seek(0)
        data_segments = 0
        # seek the correct data set in fcs
        nextdata_offset = 0
        while data_segments <= data_set:
            self.read_header(file_handle, nextdata_offset)
            self.read_text(file_handle)
            if '$NEXTDATA' in self.annotation:
                data_segments += 1
                nextdata_offset = self.annotation['$NEXTDATA']
                file_handle.seek(nextdata_offset)
                if nextdata_offset == 0 and data_segments < data_set:
                    warnings.warn("File does not contain the number of data sets.")
                    break
            else:
                if data_segments != 0:
                    warnings.warn("File does not contain $NEXTDATA information.")
                break
        if read_data:
            self.read_data(file_handle)

    @classmethod
    def from_data(cls, data):
        """Load an FCS file from a bytes-like object.

        Args:
            data: buffer containing contents of an FCS file.

        Returns:
            FCSParser instance with data loaded
        """
        obj = cls()
        with contextlib.closing(BytesIO(data)) as file_handle:
            obj.load_file(file_handle)
        return obj

    def read_header(self, file_handle, nextdata_offset=0):
        """Read the header of the FCS file.

        The header specifies where the annotation, data and analysis are located inside the binary
        file.

        Args:
            file_handle: buffer containing FCS file.
            nextdata_offset: byte offset of a set header from file start specified by $NEXTDATA
        """
        header = {'FCS format': file_handle.read(6)}

        file_handle.read(4)  # 4 space characters after the FCS format

        for field in ['text start', 'text end', 'data start', 'data end', 'analysis start',
                      'analysis end']:
            s = file_handle.read(8)
            try:
                field_value = int(s)
            except ValueError:
                field_value = 0
            header[field] = field_value + nextdata_offset

        # Checking that the location of the TEXT segment is specified
        for k in ['text start', 'text end']:
            if header[k] == 0:
                raise ValueError(u'The FCS file "{}" seems corrupted. (Parser cannot locate '
                                 u'information about the "{}" segment.)'.format(self.path, k))
            elif header[k] > self._file_size:
                raise ValueError(u'The FCS file "{}" is corrupted. "{}" segment '
                                 u'is larger than file size'.format(self.path, k))
            else:
                # All OK
                pass

        self._data_start = header['data start']
        self._data_end = header['data start']

        if header['analysis end'] - header['analysis start'] != 0:
            warnings.warn(u'There appears to be some information in the ANALYSIS segment of file '
                          u'{0}. However, it might not be read correctly.'.format(self.path))

        self.annotation['__header__'] = header

    def read_text(self, file_handle):
        """Parse the TEXT segment of the FCS file.

        The TEXT segment contains meta data associated with the FCS file.
        Converting all meta keywords to lower case.
        """
        header = self.annotation['__header__']  # For convenience

        #####
        # Read in the TEXT segment of the FCS file
        # There are some differences in how the
        file_handle.seek(header['text start'], 0)
        raw_text = file_handle.read(header['text end'] - header['text start'] + 1)
        try:
            raw_text = raw_text.decode('utf-8')
        except UnicodeDecodeError as e:
            logger.warning(u'Encountered an illegal utf-8 byte in the header.\n Illegal utf-8 '
                           u'characters will be ignored.\n The illegal byte was {} at '
                           u'position {}.'.format(repr(e.object[e.start]), e.start))
            raw_text = raw_text.decode('utf-8', 'ignore')

        #####
        # Parse the TEXT segment of the FCS file into a python dictionary
        delimiter = raw_text[0]

        if raw_text[-1] != delimiter:
            raw_text = raw_text.strip()
            if raw_text[-1] != delimiter:
                msg = (u'The first two characters were:\n {}. The last two characters were: {}\n'
                       u'Parser expects the same delimiter character in beginning '
                       u'and end of TEXT segment'.format(raw_text[:2], raw_text[-2:]))
                raise ParserFeatureNotImplementedError(msg)

        # Below 1:-1 used to remove first and last characters which should be reserved for delimiter
        raw_text_segments = raw_text[1:-1].split(delimiter)
        keys, values = raw_text_segments[0::2], raw_text_segments[1::2]
        text = {key: value for key, value in zip(keys, values)}  # build dictionary

        ##
        # Extract channel names and convert some of the channel properties
        # and other fields into numeric data types (from string)
        # Note: do not use regular expressions for manipulations here.
        # Regular expressions are too heavy in terms of computation time.
        pars = int(text['$PAR'])
        if '$P0B' in keys:  # Checking whether channel number count starts from 0 or from 1
            self.channel_numbers = range(0, pars)  # Channel number count starts from 0
        else:
            self.channel_numbers = range(1, pars + 1)  # Channel numbers start from 1

        # Extract parameter names
        try:
            names_n = tuple([text['$P{0}N'.format(i)] for i in self.channel_numbers])
        except KeyError:
            names_n = []

        try:
            names_s = tuple([text['$P{0}S'.format(i)] for i in self.channel_numbers])
        except KeyError:
            names_s = []

        self.channel_names_s = names_s
        self.channel_names_n = names_n

        # Convert some of the fields into integer values
        keys_encoding_bits = ['$P{0}B'.format(i) for i in self.channel_numbers]

        add_keys_to_convert_to_int = ['$NEXTDATA', '$PAR', '$TOT']

        keys_to_convert_to_int = keys_encoding_bits + add_keys_to_convert_to_int

        for key in keys_to_convert_to_int:
            value = text[key]
            text[key] = int(value)

        self.annotation.update(text)

        # Update data start segments if needed

        if self._data_start == 0:
            self._data_start = int(text['$BEGINDATA'])
        if self._data_end == 0:
            self._data_end = int(text['$ENDDATA'])

    def read_analysis(self, file_handle):
        """Read the ANALYSIS segment of the FCS file and store it in self.analysis.

        Warning: This has never been tested with an actual fcs file that contains an
        analysis segment.

        Args:
            file_handle: buffer containing FCS data
        """
        start = self.annotation['__header__']['analysis start']
        end = self.annotation['__header__']['analysis end']
        if start != 0 and end != 0:
            file_handle.seek(start, 0)
            self._analysis = file_handle.read(end - start)
        else:
            self._analysis = None

    def _verify_assumptions(self):
        """Verify that all assumptions made by the parser hold."""
        text = self.annotation
        keys = text.keys()

        if '$MODE' not in text or text['$MODE'] != 'L':
            raise ParserFeatureNotImplementedError(u'Mode not implemented')

        if '$P0B' in keys:
            raise ParserFeatureNotImplementedError(u'Not expecting a parameter starting at 0')

        if text['$BYTEORD'] not in ["1,2,3,4", "4,3,2,1", "1,2", "2,1"]:
            raise ParserFeatureNotImplementedError(u'$BYTEORD {} '
                                                   u'not implemented'.format(text['$BYTEORD']))

    def get_channel_names(self):
        """Get list of channel names. Raises a warning if the names are not unique."""
        names_s, names_n = self.channel_names_s, self.channel_names_n

        # Figure out which channel names to use
        if self._channel_naming == '$PnS':
            channel_names, channel_names_alternate = names_s, names_n
        else:
            channel_names, channel_names_alternate = names_n, names_s

        if len(channel_names) == 0:
            channel_names = channel_names_alternate

        if len(set(channel_names)) != len(channel_names):
            msg = (u'The default channel names (defined by the {} '
                   u'parameter in the FCS file) were not unique. To avoid '
                   u'problems in downstream analysis, the channel names '
                   u'have been switched to the alternate channel names '
                   u'defined in the FCS file. To avoid '
                   u'seeing this warning message, explicitly instruct '
                   u'the FCS parser to use the alternate channel names by '
                   u'specifying the channel_naming parameter.')
            msg = msg.format(self._channel_naming)
            warnings.warn(msg)
            channel_names = channel_names_alternate

        return channel_names

    def read_data(self, file_handle):
        """Read the DATA segment of the FCS file."""
        self._verify_assumptions()
        text = self.annotation

        if (self._data_start > self._file_size) or (self._data_end > self._file_size):
            raise ValueError(u'The FCS file "{}" is corrupted. Part of the data segment '
                             u'is missing.'.format(self.path))

        num_events = text['$TOT']  # Number of events recorded
        num_pars = text['$PAR']  # Number of parameters recorded

        if text['$BYTEORD'].strip() == '1,2,3,4' or text['$BYTEORD'].strip() == '1,2':
            endian = '<'
        elif text['$BYTEORD'].strip() == '4,3,2,1' or text['$BYTEORD'].strip() == '2,1':
            endian = '>'
        else:
            msg = 'Unrecognized byte order ({})'.format(text['$BYTEORD'])
            raise ParserFeatureNotImplementedError(msg)

        # dictionary to convert from FCS format to numpy convention
        conversion_dict = {'F': 'f', 'D': 'f', 'I': 'u'}

        if text['$DATATYPE'] not in conversion_dict.keys():
            raise ParserFeatureNotImplementedError('$DATATYPE = {0} is not yet '
                                                   'supported.'.format(text['$DATATYPE']))

        # Calculations to figure out data types of each of parameters
        # $PnB specifies the number of bits reserved for a measurement of parameter n
        bytes_per_par_list = [int(text['$P{0}B'.format(i)] / 8) for i in self.channel_numbers]

        par_numeric_type_list = [
            '{endian}{type}{size}'.format(endian=endian,
                                          type=conversion_dict[text['$DATATYPE']],
                                          size=bytes_per_par)
            for bytes_per_par in bytes_per_par_list
        ]

        # Parser for list mode. Here, the order is a list of tuples.
        # Each tuple stores event related information
        file_handle.seek(self._data_start, 0)  # Go to the part of the file where data starts

        ##
        # Read in the data
        if len(set(par_numeric_type_list)) > 1:
            # values saved in mixed data formats
            dtype = ','.join(par_numeric_type_list)
            data = fromfile(file_handle, dtype, num_events)
            names = self.get_channel_names()
            data.dtype.names = tuple([name.encode('ascii', errors='replace') for name in names])
        else:
            # values saved in a single data format
            dtype = par_numeric_type_list[0]
            data = fromfile(file_handle, dtype, num_events * num_pars)
            data = data.reshape((num_events, num_pars))
        ##
        # Convert to native byte order
        # This is needed for working with pandas data structures
        native_code = '<' if (sys.byteorder == 'little') else '>'
        if endian != native_code:
            # swaps the actual bytes and also the endianness
            data = data.byteswap().newbyteorder()

        self._data = data

    @property
    def data(self):
        """Get parsed DATA segment of the FCS file."""
        if self._data is None:
            with open(self.path, 'rb') as f:
                self.read_data(f)
        return self._data

    @property
    def analysis(self):
        """Get ANALYSIS segment of the FCS file."""
        if self._analysis is None:
            with open(self.path, 'rb') as f:
                self.read_analysis(f)
        return self._analysis

    def reformat_meta(self):
        """Collect the meta data information in a more user friendly format.

        Function looks through the meta data, collecting the channel related information into a
        dataframe and moving it into the _channels_ key.
        """
        meta = self.annotation  # For shorthand (passed by reference)
        channel_properties = []

        for key, value in meta.items():
            if key[:3] == '$P1':
                if key[3] not in string.digits:
                    channel_properties.append(key[3:])

        # Capture all the channel information in a list of lists -- used to create a data frame
        channel_matrix = [
            [meta.get('$P{0}{1}'.format(ch, p)) for p in channel_properties]
            for ch in self.channel_numbers
        ]

        # Remove this information from the dictionary
        for ch in self.channel_numbers:
            for p in channel_properties:
                key = '$P{0}{1}'.format(ch, p)
                if key in meta:
                    meta.pop(key)

        num_channels = meta['$PAR']
        column_names = ['$Pn{0}'.format(p) for p in channel_properties]

        df = pd.DataFrame(channel_matrix, columns=column_names,
                          index=(1 + numpy.arange(num_channels)))

        if '$PnE' in column_names:
            df['$PnE'] = df['$PnE'].apply(lambda x: x.split(','))

        df.index.name = 'Channel Number'
        meta['_channels_'] = df
        meta['_channel_names_'] = self.get_channel_names()

    @property
    def dataframe(self):
        """Construct Pandas dataframe."""
        data = self.data
        channel_names = self.get_channel_names()
        return pd.DataFrame(data, columns=channel_names)


def parse(path, meta_data_only=False, output_format='DataFrame', compensate=False,
          channel_naming='$PnS', reformat_meta=False, data_set=0):
    """Parse an fcs file at the location specified by the path.

    Parameters
    ----------
    path: str
        Path of .fcs file
    meta_data_only: bool
        If True, the parse_fcs only returns the meta_data (the TEXT segment of the FCS file)
    output_format: 'DataFrame' | 'ndarray'
        If set to 'DataFrame' the returned
    channel_naming: '$PnS' | '$PnN'
        Determines which meta data field is used for naming the channels.
        The default should be $PnS (even though it is not guaranteed to be unique)

        $PnN stands for the short name (guaranteed to be unique).
            Will look like 'FL1-H'
        $PnS stands for the actual name (not guaranteed to be unique).
            Will look like 'FSC-H' (Forward scatter)

        The chosen field will be used to population self.channels

        Note: These names are not flipped in the implementation.
        It looks like they were swapped for some reason in the official FCS specification.
    reformat_meta: bool
        If true, the meta data is reformatted with the channel information organized
        into a DataFrame and moved into the '_channels_' key
    data_set: int
        Index of retrieved data set in the fcs file.
        This value specifies the data set being retrieved from an fcs file with multple data sets.

    Returns
    -------
    if meta_data_only is True:
        meta_data: dict
            Contains a dictionary with the meta data information
    Otherwise:
        a 2-tuple with
            the first element the meta_data (dictionary)
            the second element the data (in either DataFrame or numpy format)

    Examples
    --------
    fname = '../tests/data/EY_2013-05-03_EID_214_PID_1120_Piperacillin_Well_B7.001.fcs'
    meta = parse_fcs(fname, meta_data_only=True)
    meta, data_pandas = parse_fcs(fname, meta_data_only=False, output_format='DataFrame')
    meta, data_numpy  = parse_fcs(fname, meta_data_only=False, output_format='ndarray')
    """
    if compensate:
        raise ParserFeatureNotImplementedError(u'Compensation has not been implemented yet.')

    if reformat_meta or (output_format == 'DataFrame'):
        if pd is None:
            raise ImportError(u'You do not have pandas installed.')

    read_data = not meta_data_only

    parsed_fcs = FCSParser(path, read_data=read_data, channel_naming=channel_naming, data_set=data_set)

    if reformat_meta:
        parsed_fcs.reformat_meta()

    meta = parsed_fcs.annotation

    if meta_data_only:
        return meta
    elif output_format == 'DataFrame':
        return meta, parsed_fcs.dataframe
    elif output_format == 'ndarray':
        # Constructs numpy matrix
        return meta, parsed_fcs.data
    else:
        raise ValueError(u'The output_format must be either "ndarray" or "DataFrame".')
