import typing
import pandas as pd
import numpy as np
from inspect import isclass
from datetime import date

T = typing.TypeVar("T")


class DataType:
    def __init__(self, tp):
        self.type = self.get_type(tp)
        self.dtypes = self.get_dtype(self.type)
        self.optional = type(None) in self.dtypes

    @staticmethod
    def get_type(tp):
        """
        Extract any generic type hint from series definition

        :param tp:
        :return:
        """
        if isclass(tp):
            return typing.Any
        else:
            origin = typing.get_origin(tp)
            args = typing.get_args(tp)

            if isclass(origin):
                return args[0]
            else:
                typ = next((DataType.get_type(a) for a in args if DataType.is_series(a)), typing.Any)
                opt = type(None) in args
                return typing.Union[typ, None] if opt else typ

    @staticmethod
    def get_dtype(tp) -> tuple:
        """
        Get associated numpy types from the primitive type

        :param tp:
        :return:
        """
        if typing.get_origin(tp) is typing.Union:
            args = typing.get_args(tp)
            return sum((DataType.get_dtype(a) for a in args), tuple())
        elif tp is str:
            return np.str_, np.object_
        elif tp is int:
            return np.int32, np.int64
        elif tp is float:
            return np.float32, np.float64
        elif tp is bool:
            return np.bool_,
        elif isclass(tp) and issubclass(tp, date):
            return np.dtype("datetime64[ns]"),
        else:
            return tp,

    @staticmethod
    def is_series(tp) -> bool:
        """
        Check whether type definition includes a pandas series

        :param tp:
        :return:
        """
        if isclass(tp):
            return issubclass(tp, pd.Series)
        else:
            origin = typing.get_origin(tp)
            if isclass(origin):
                return issubclass(origin, pd.Series)
            else:
                args = typing.get_args(tp)
                return next((True for a in args if DataType.is_series(a)), False)

    def check(self, dtype) -> bool:
        """
        Evaluate typing definition against dtype

        :param dtype:
        :return:
        """
        for t in self.dtypes:
            if t is typing.Any or np.issubdtype(dtype, t):
                return True

        return False

    def __str__(self):
        """
        Represent as a comma separated list of dtypes

        :return:
        """
        output = [t.__name__ if hasattr(t, "__name__") else type(t).__name__ for t in self.dtypes]
        return ", ".join(output)


class DataFrameMeta(type):
    __dtypes: typing.Dict[str, DataType]

    def __init__(cls, name, bases, attrs):
        """
        Create dataframe class extracting type definitions

        :param name:
        :param bases:
        :param attrs:
        """
        super().__init__(name, bases, attrs)
        cls.__dtypes = {}

        # read annotations manually, ref. https://github.com/pandas-dev/pandas/issues/43912
        for base in bases:
            if type(base) is DataFrameMeta:
                cls.__dtypes.update(base.get_type_hints())

        for key, tp in cls.__annotations__.items():
            if DataType.is_series(tp):
                cls.__dtypes[key] = DataType(tp)

    def get_type_hints(cls) -> typing.Dict[str, DataType]:
        """
        Retrieve a map of extracted type definitions

        :return:
        """
        return cls.__dtypes

    def create(cls: typing.Type[T], data, validate=True) -> T:
        """
        Create new dataframe and validate

        :param data:
        :param validate:
        :return:
        """
        df = pd.DataFrame(data)

        if isinstance(df.columns, pd.RangeIndex):
            if len(df.columns) != len(cls.__dtypes):
                raise ValueError("Unnamed data needs to equal pre-defined column length")
            else:
                df.columns = pd.Index(cls.__dtypes.keys())

        if validate:
            cls.validate(df)

        return df

    def validate(cls: typing.Type[T], df: pd.DataFrame) -> T:
        """
        Validate dataframe against definitions throwing on first encountered error

        :param df:
        :return:
        """
        for key, dt in cls.__dtypes.items():
            if key in df.columns:
                dtype = df.dtypes[key]
                if not dt.check(dtype):
                    raise TypeError(f"Invalid data type '{dtype}' for column '{key}' expected '{dt}'")
            elif not dt.optional:
                raise TypeError(f"Missing required column '{key}'")

        return df

    def __getattr__(cls, item):
        """
        Return defined series attributes as a string for static column referencing

        :param item:
        :return:
        """
        if item in cls.__dtypes:
            return item
        else:
            raise AttributeError(item)


class DataFrame(pd.DataFrame, metaclass=DataFrameMeta):
    ...


class Series(typing.Generic[T], pd.Series):
    ...
