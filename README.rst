FCSParser
=================


fcsparser is a python package for reading fcs files. 

.. image:: https://github.com/eyurtsev/kor/actions/workflows/test.yml/badge.svg?branch=main&event=push   
   :target: https://github.com/eyurtsev/kor/actions/workflows/test.yml
   :alt: Unit Tests

Install
==================

    $ pip install fcsparser
    
or
    
    $ conda install -c bioconda fcsparser

Using
==================

    >>> import fcsparser
    >>> path = fcsparser.test_sample_path
    >>> meta, data = fcsparser.parse(path, reformat_meta=True)

A more detailed example can be found here: https://github.com/eyurtsev/fcsparser/blob/master/doc/fcsparser_example.ipynb

Features
===================

- **python**: 3.8, 3.9, 3.10, 3.11
- **FCS Formats**: Supports FCS 2.0, 3.0, and 3.1
- **FCS Machines**: BD FACSCalibur, BD LSRFortessa, BD LSR-II, MiltenyiBiotec MACSQuant VYB, Sony SH800

Contributing
=================

Pull requests are greatly appreciated. Missing features include:

1. the ability to apply compensation.
2. a set of transformations (hlog, logicle, etc.) that can be applied.

Also fcs files from more devices and more formats are greatly appreciated, especially if the parser fails for them!

Resources
==================

- **Documentation:** https://github.com/eyurtsev/fcsparser
- **Source Repository:** https://github.com/eyurtsev/fcsparser
- **Comments or questions:** https://github.com/eyurtsev/fcsparser/issues

LICENSE
===================

The MIT License (MIT)

Copyright (c) 2013-2023 Eugene Yurtsev

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
