# datahints

Validate pandas dataframes using type information.
Enables IDE autocompletion and refactoring of column names
as well as explicitly documenting expected columns and their data types.

All dataframes are treated as standard pandas dataframes,
no instancing of the subclasses is required.

## Installation
`pip install datahints`

## Example
```python
import pandas as pd
import numpy as np
from datahints import DataFrame, Series
from typing import Optional, Union

class MyDataFrame(DataFrame):
    col1: Series[int]
    col2: Series[Union[int, float]]
    col3: Optional[Series[np.float64]]

# validate any dataframe
df = pd.DataFrame({"col1": [1, 2], "col2": [3.0, 4.0]})
MyDataFrame.validate(df)

# validate and annotate type for IDE support
df = MyDataFrame.validate(df)

# use statically named columns 
df[MyDataFrame.col3] = [5.0, 6.0]

# constructor simply returns a dataframe instance
df = MyDataFrame({MyDataFrame.col1: [1, 2], MyDataFrame.col2: [3.0, 4.0]})
MyDataFrame.validate(df)

# create and validate directly
df = MyDataFrame.create({
    MyDataFrame.col1: [1, 2],
    MyDataFrame.col2: [3.0, 4.0]
})

# infer column names automatically (if all columns are present)
df = MyDataFrame.create([
    (1, 3.0, 5.0),
    (2, 4.0, 6.0)
])
```

## Type inference
Native types are automatically assumed to represent the following numpy equivalents:

|type|dtype|
|----|-----|
|`str`|`np.str_`, `np.object_`|
|`int`|`np.int32`, `np.int64`|
|`float`|`np.float32`, `np.float64`|
|`bool`|`np.bool_`|
|`date`, `datetime`|`"datetime64[ns]"`|
