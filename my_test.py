#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 11:37:26 2024

@author: marty
"""
import tempfile
from pathlib import Path

import pandas as pd

from hybf import CompressedWriter, CompressedReader
from tests.data.generators import DataGenerator

#df1 = DataGenerator.create_compressed_dataset()
df1 = DataGenerator.create_edge_cases_dataset()

test_file = Path(tempfile.mkdtemp()) /'test.hybf'

with open(test_file, 'wb') as f:
    writer = CompressedWriter()
    writer.write(df1, f)

with open(test_file, 'rb') as f:
    reader = CompressedReader()
    df2 = reader.read(f)
    
assert list(df1.columns) == list(df2.columns)

for col in df1.columns:
    print(col)
    assert df1[col].dtype == df2[col].dtype
    pd.testing.assert_series_equal(df1[col], df2[col])
                