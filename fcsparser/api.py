#!/usr/bin/env python
# Eugene Yurtsev 07/20/2013
# Distributed under the MIT License

# Thanks to:
# - Ben Roth : adding a fix for Accuri C6 fcs files.

##
# Useful documentation for dtypes in numpy
# http://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.byteswap.html?highlight=byteswap#numpy.ndarray.byteswap
# http://docs.scipy.org/doc/numpy/user/basics.types.html
# http://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html
from __future__ import division

import sys
import warnings
import string
import os
import logging

import numpy

try:
    import pandas as pd
except ImportError:
    pd = None
    warnings.warn("pandas is not installed, so the parse_fcs function "
                  "can only be used together with numpy.")


class ParserFeatureNotImplementedError(Exception):
    """Raised when encountering fcs files with an unfamiliar feature"""
    pass


# def raise_parser_feature_not_implemented(message):
#     print(("Some of the parser features have not yet been implemented.\n"
#            "If you would like to see this feature implemented, please send a sample FCS file\n"
#            " to the developers.\n"
#            "The following problem was encountered with your FCS file:\n"
#            " {0} ").format(message))
#     raise NotImplementedError(message)


class FCSParser(object):
    """A Parser for .fcs files.
    Should work for many FCS 2.0, 3.0, 3.1

    self.annotation: a dictionary holding the parsed content of the TEXT segment
                     In addition, a key called __header__ has been added to this dictionary
                     It specifies the information parsed from the FCS file HEADER segment.
                     (This won't be necessary for most users.)

    self.data holds the parsed DATA segment
    self.analysis holds the ANALYSIS segment as read from the file.

    After the data segment is read:
        self.channel_names holds the chosen names of the channels
        self.channel_names_alternate holds the alternate names of the channels
    """

    def __init__(self, path, dataset_start=0, read_data=True, channel_naming='$PnS'):
        """
        Parameters
        ----------
        path : str
            Path of .fcs file
        dataset_start : integer
            Start of the dataset
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
        """
        self._data = None
        self._channel_naming = channel_naming
        self.channel_names_s = []
        self.channel_names_n = []
        self._dataset_start = dataset_start;

        # Attributes parsed from fcs file
        self._data_start = -1
        self._data_end = -1
        self._text_start = -1
        self._text_end = -1
        self.channel_numbers = []
        self._analysis = ''

        self._file_size = os.path.getsize(path)

        if channel_naming not in ('$PnN', '$PnS'):
            raise ValueError("channel_naming must be either '$PnN' or '$PnS")

        self.annotation = {}
        self.path = path

        with open(path, 'rb') as f:
            f.seek(dataset_start,0)
            self.read_header(f)
            if(self._dataset_start):
                self._data_start=self._data_start+self._dataset_start;
                self._data_end=self._data_end+self._dataset_start;
                self._text_start=self._text_start+self._dataset_start;
                self._text_end=self._text_end+self._dataset_start;
            self.read_text(f)
            if read_data:
                self.read_data(f)

    def read_header(self, file_handle):
        """
        Reads the header of the FCS file.
        The header specifies where the annotation,
        data and analysis are located inside the binary file.
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
            header[field] = field_value

        # Checking that the location of the TEXT segment is specified
        for k in ['text start', 'text end']:
            if header[k] == 0:
                raise ValueError(
                    "The FCS file '{}' seems corrupted. (Parser cannot locate information "
                    "about the '{}' segment.)".format(self.path, k))
            elif header[k] > self._file_size:
                raise ValueError("The FCS file '{}' is corrupted. '{}' segment "
                                 "is larger than file size".format(self.path, k))

        self._data_start = header['data start']
        self._data_end = header['data start']
        self._text_start = header['text start']
        self._text_end = header['text end']

        if header['analysis start'] != 0:
            warnings.warn(
                "There appears to be some information in the ANALYSIS segment of file {0}. "
                "However, it might not be read correctly.".format(
                    self.path))

        self.annotation['__header__'] = header

    def read_text(self, file_handle):
        """
        Reads the TEXT segment of the FCS file.
        This is the meta data associated with the FCS file.
        Converting all meta keywords to lower case.
        """
        header = self.annotation['__header__']  # For convenience

        #####
        # Read in the TEXT segment of the FCS file
        # There are some differences in how the 
        file_handle.seek(self._text_start, 0)
        raw_text = file_handle.read(self._text_end - self._text_start + 1)
        raw_text = raw_text.decode('utf-8')

        #####
        # Parse the TEXT segment of the FCS file into a python dictionary
        delimiter = raw_text[0]

        if raw_text[-1] != delimiter:
            raw_text = raw_text.strip()
            if raw_text[-1] != delimiter:
                msg = 'The first two characters were:\n {}. The last two characters were: {}\n' \
                      'Parser expects the same delimiter character in beginning ' \
                      'and end of TEXT segment'.format(repr(raw_text[:2]), (repr(raw_text[-2:])))
                raise ParserFeatureNotImplementedError(msg)

        # Below 1:-1 used to remove first and last characters which should be reserved for delimiter
        raw_text_segments = raw_text[1:-1].split(delimiter)
        keys, values = raw_text_segments[0::2], raw_text_segments[1::2]
        text = {key: value for key, value in zip(keys, values)}  # build dictionary

        ####
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

        # unused currently but keep for debugging
        # keys_encoding_range = ['$P{0}R'.format(i) for i in self.channel_numbers]

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

            ##
            # Keep for debugging
            # key_list = self.header['text'].keys()
            # for key in sorted(text.keys()):
            # print key, text[key]
            # raise Exception('here')

    def read_analysis(self, file_handle):
        """
        Reads the ANALYSIS segment of the FCS file and stores it in self.analysis
        Warning: This has never been tested with an actual
        fcs file that contains an analysis segment.
        """
        start = self.annotation['__header__']['analysis start']
        end = self.annotation['__header__']['analysis end']
        if start != 0 and end != 0:
            file_handle.seek(start+self._dataset_start, 0)
            self._analysis = file_handle.read(end - start)
        else:
            self._analysis = ''

    def _check_assumptions(self):
        """
        Checks the FCS file to make sure that some of the assumptions made by the parser are met.
        """
        text = self.annotation
        keys = text.keys()

        if '$NEXTDATA' in text and text['$NEXTDATA'] != 0:
            if '$ENDDATA' in text:
                nextdata = text['$NEXTDATA']
                enddata = int(text['$ENDDATA'])
                if nextdata != enddata+1:
                    logging.error('This file has multiple datasets, $NEXTDATA must equal to $ENDDATA+1')
                else:
                    logging.info('This file has multiple datasets')
            else:
                logging.error('This file has multiple datasets, $ENDDATA not defined')


        if '$MODE' not in text or text['$MODE'] != 'L':
            raise ParserFeatureNotImplementedError('Mode not implemented')

        if '$P0B' in keys:
            raise ParserFeatureNotImplementedError('Not expecting a parameter starting at 0')

        if text['$BYTEORD'] not in ["1,2,3,4", "4,3,2,1", "1,2", "2,1"]:
            raise ParserFeatureNotImplementedError(
                '$BYTEORD {} not implemented'.format(text['$BYTEORD']))

    def get_channel_names(self):
        """
        Figures out which the channel names to use.
        Raises a warning if the names are not unique.
        """
        names_s, names_n = self.channel_names_s, self.channel_names_n

        # Figure out which channel names to use
        if self._channel_naming == '$PnS':
            channel_names, channel_names_alternate = names_s, names_n
        else:
            channel_names, channel_names_alternate = names_n, names_s

        if len(channel_names) == 0:
            channel_names = channel_names_alternate

        if len(set(channel_names)) != len(channel_names):
            msg = ('The default channel names (defined by the {} ' +
                   'parameter in the FCS file) were not unique. To avoid ' +
                   'problems in downstream analysis, the channel names ' +
                   'have been switched to the alternate channel names ' +
                   'defined in the FCS file. To avoid ' +
                   'seeing this warning message, explicitly instruct ' +
                   'the FCS parser to use the alternate channel names by ' +
                   'specifying the channel_naming parameter.')
            msg = msg.format(self._channel_naming)
            warnings.warn(msg)
            channel_names = channel_names_alternate

        return channel_names

    def read_data(self, file_handle):
        """ Reads the DATA segment of the FCS file. """
        self._check_assumptions()
        text = self.annotation

        if (self._data_start > self._file_size) or (self._data_end > self._file_size):
            raise ValueError(
                "The FCS file '{}' is corrupted. Part of the data segment is missing.".format(
                    self.path))

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
            raise ParserFeatureNotImplementedError(
                '$DATATYPE = {0} is not yet supported.'.format(text['$DATATYPE']))

        # Calculations to figure out data types of each of parameters
        # $PnB specifies the number of bits reserved for a measurement of parameter n
        bytes_per_par_list = [int(text['$P{0}B'.format(i)] / 8) for i in self.channel_numbers]

        par_numeric_type_list = [
            '{endian}{type}{size}'.format(endian=endian, type=conversion_dict[text['$DATATYPE']],
                                          size=bytes_per_par)
            for bytes_per_par in bytes_per_par_list]

        # unused currently, but keep for debugging
        # bytes_per_event = sum(bytes_per_par_list)
        # total_bytes = bytes_per_event * num_events

        # Parser for list mode. Here, the order is a list of tuples.
        # Each tuple stores event related information
        file_handle.seek(self._data_start, 0)  # Go to the part of the file where data starts

        ##
        # Read in the data
        if len(set(par_numeric_type_list)) > 1:
            # values saved in mixed data formats
            dtype = ','.join(par_numeric_type_list)
            data = numpy.fromfile(file_handle, dtype=dtype, count=num_events)
            names = self.get_channel_names()
            data.dtype.names = tuple([name.encode('ascii', errors='replace') for name in names])
        else:
            # values saved in a single data format
            dtype = par_numeric_type_list[0]
            data = numpy.fromfile(file_handle, dtype=dtype, count=num_events * num_pars)
            data = data.reshape((num_events, num_pars))
        ##
        # Convert to native byte order 
        # This is needed for working with pandas datastructures
        native_code = '<' if (sys.byteorder == 'little') else '>'
        if endian != native_code:
            # swaps the actual bytes and also the endianness
            data = data.byteswap().newbyteorder()

        self._data = data

    @property
    def data(self):
        """ Holds the parsed DATA segment of the FCS file. """
        if self._data is None:
            with open(self.path, 'rb') as f:
                self.read_data(f)
        return self._data

    @property
    def analysis(self):
        """ Holds the parsed ANALYSIS segment of the FCS file. """
        if self._analysis == '':
            with open(self.path, 'rb') as f:
                self.read_analysis(f)
        return self._analysis

    def reformat_meta(self):
        """ Collects the meta data information in a more user friendly format.
        Function looks through the meta data, collecting the channel related information
        into a dataframe and moving it into the _channels_ key
        """
        meta = self.annotation  # For shorthand (passed by reference)
        channel_properties = []

        for key, value in meta.items():
            if key[:3] == '$P1':
                if key[3] not in string.digits:
                    channel_properties.append(key[3:])

        # Capture all the channel information in a list of lists -- used to create a data frame
        channel_matrix = [[meta.get('$P{0}{1}'.format(ch, p)) for p in channel_properties] for ch in
                          self.channel_numbers]

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


def parse(path, dataset_start=0, meta_data_only=False, output_format='DataFrame', compensate=False,
              channel_naming='$PnS',
              reformat_meta=False):
    """
    Parse an fcs file at the location specified by the path.

    Parameters
    ----------
    path : str
        Path of .fcs file
    dataset_start : integer
        Start of dataset in the file (multi-dataset)
    meta_data_only : bool
        If True, the parse_fcs only returns the meta_data (the TEXT segment of the FCS file)
    output_format : 'DataFrame' | 'ndarray'
        If set to 'DataFrame' the returned
    channel_naming : '$PnS' | '$PnN'
        Determines which meta data field is used for naming the channels.
        The default should be $PnS (even though it is not guaranteed to be unique)

        $PnN stands for the short name (guaranteed to be unique).
            Will look like 'FL1-H'
        $PnS stands for the actual name (not guaranteed to be unique).
            Will look like 'FSC-H' (Forward scatter)

        The chosen field will be used to population self.channels

        Note: These names are not flipped in the implementation.
        It looks like they were swapped for some reason in the official FCS specification.

    reformat_meta : bool
        If true, the meta data is reformatted with the channel information organized
        into a DataFrame and moved into the '_channels_' key

    Returns
    -------
    if meta_data_only is True:
        meta_data : dict
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
        raise ParserFeatureNotImplementedError("Compensation has not been implemented yet.")

    if reformat_meta or (output_format == 'DataFrame'):
        if pd is None:
            raise ImportError('You do not have pandas installed.')

    read_data = not meta_data_only

    parsed_fcs = FCSParser(path, dataset_start=dataset_start, read_data=read_data, channel_naming=channel_naming)

    if reformat_meta:
        parsed_fcs.reformat_meta()

    meta = parsed_fcs.annotation

    if meta_data_only:
        return meta
    elif output_format == 'DataFrame':
        # Constructs pandas DF object
        data = parsed_fcs.data
        channel_names = parsed_fcs.get_channel_names()
        data = pd.DataFrame(data, columns=channel_names)
        return meta, data
    elif output_format == 'ndarray':
        # Constructs numpy matrix
        return meta, parsed_fcs.data
    else:
        raise ValueError("The output_format must be either 'ndarray' or 'DataFrame'")
