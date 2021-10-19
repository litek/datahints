import pandas as pd
import numpy as np
from unittest import TestCase
from typing import Any, Optional, Union
from datetime import date, datetime
from datahints import DataType, DataFrame, Series


class A(DataFrame):
    col1: Series[int]
    col2: Optional[Series[float]]


class B(A):
    col3: Series[str]


class DataTypeTests(TestCase):
    def test_get_type(self):
        self.assertEqual(DataType.get_type(pd.Series), Any)
        self.assertEqual(DataType.get_type(Series), Any)
        self.assertEqual(DataType.get_type(Series[int]), int)
        self.assertEqual(DataType.get_type(Series[Union[int, float]]), Union[int, float])
        self.assertEqual(DataType.get_type(Optional[Series[str]]), Union[str, None])
        self.assertEqual(DataType.get_type(Optional[Series[Union[int, float]]]), Union[int, float, None])

    def test_get_dtype(self):
        self.assertEqual(DataType.get_dtype(str), (np.str_, np.object_))
        self.assertEqual(DataType.get_dtype(int), (np.int32, np.int64))
        self.assertEqual(DataType.get_dtype(float), (np.float32, np.float64))
        self.assertEqual(DataType.get_dtype(bool), (np.bool_, ))
        self.assertEqual(DataType.get_dtype(date), (np.dtype("datetime64[ns]"),))
        self.assertEqual(DataType.get_dtype(datetime), (np.dtype("datetime64[ns]"),))
        self.assertEqual(DataType.get_dtype(Union[str, int]), (np.str_, np.object_, np.int32, np.int64))
        self.assertEqual(DataType.get_dtype(Union[str, Union[int, bool]]), (np.str_, np.object_, np.int32, np.int64, np.bool_))

    def test_is_series(self):
        self.assertTrue(DataType.is_series(pd.Series))
        self.assertTrue(DataType.is_series(Series))
        self.assertTrue(DataType.is_series(Series[int]))
        self.assertTrue(DataType.is_series(Optional[Series[int]]))
        self.assertFalse(DataType.is_series(int))

    def test_check(self):
        dt = DataType(Series[Union[int, float]])
        self.assertTrue(dt.check(np.int64))
        self.assertTrue(dt.check(np.float64))
        self.assertFalse(dt.check(np.object_))


class DataFrameTestCase(TestCase):
    def test_create(self):
        df = A.create([(1, 2.0)])
        self.assertEqual(df.columns.to_list(), [A.col1, A.col2])
        self.assertRaises(ValueError, lambda: A.create([(1, 2.0, 3)]))

    def test_validate(self):
        df = pd.DataFrame([(1, 2.0, 3)], columns=B.get_type_hints().keys())
        self.assertRaises(TypeError, lambda: B.validate(df))

        df.col3 = df.col3.astype(str)
        self.assertTrue(B.validate(df) is df)

    def test_getattr(self):
        self.assertEqual(A.col1, "col1")
        self.assertEqual(B.col2, "col2")
        self.assertRaises(AttributeError, lambda: A.col3)

        with self.assertWarns(UserWarning):
            class Foo(DataFrame):
                merge: Series
