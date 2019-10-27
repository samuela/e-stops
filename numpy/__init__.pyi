"""See

* https://github.com/facebook/pyre-check/issues/47
* https://github.com/numpy/numpy-stubs
"""

# pylint: disable=line-too-long
# See https://stackoverflow.com/questions/33533148/how-do-i-specify-that-the-return-type-of-a-method-is-the-same-as-the-class-itsel
from __future__ import annotations

from typing import Any, List, Optional, Tuple, TypeVar, Union, Iterable

# pylint: disable=unused-argument, redefined-builtin, inherit-non-class

pi: float
newaxis: None

Shape = Tuple[int, ...]

class ndarray(Iterable):
  @property
  def shape(self) -> Shape:
    ...

  @property
  def T(self) -> ndarray:
    ...

  def __int__(self) -> int:
    ...

  def __float__(self) -> float:
    ...

  def __getitem__(self, key) -> Any:
    ...

  def __add__(self, other) -> ndarray:
    ...

  def __radd__(self, other) -> ndarray:
    ...

  def __sub__(self, other) -> ndarray:
    ...

  def __rsub__(self, other) -> ndarray:
    ...

  def __mul__(self, other) -> ndarray:
    ...

  def __rmul__(self, other) -> ndarray:
    ...

  def __div__(self, other) -> ndarray:
    ...

  def __neg__(self) -> ndarray:
    ...

  def __matmul__(self, other) -> ndarray:
    ...

  def __truediv__(self, other) -> ndarray:
    ...

  def __le__(self, other) -> ndarray:
    ...

ArrayLike = TypeVar("ArrayLike", int, float, ndarray, List[int], List[float])

def abs(x: ArrayLike) -> ArrayLike:
  ...

def amax(a: ndarray, axis: Optional[int] = None) -> ndarray:
  ...

def amin(a: ndarray, axis: Optional[int] = None) -> ndarray:
  ...

def arange(start: int) -> ndarray:
  ...

def argmax(a: ArrayLike, axis: Optional[int] = None) -> ArrayLike:
  ...

def array(object: Any) -> ndarray:
  ...

def broadcast_arrays(*args: ArrayLike) -> List[ndarray]:
  ...

def broadcast_to(arr: ndarray, shape: Shape) -> ndarray:
  ...

def clip(a: ArrayLike, a_min: Any, a_max: Any) -> ArrayLike:
  ...

def cos(x: ArrayLike) -> ArrayLike:
  ...

def cumsum(a: ArrayLike, axis: Optional[int]) -> ArrayLike:
  ...

def diag(v: ndarray, k: int = 0) -> ndarray:
  ...

def dot(a: ndarray, b: ndarray) -> ndarray:
  ...

def eye(n: int) -> ndarray:
  ...

def inner(a: ndarray, b: ndarray) -> ndarray:
  ...

def interp(x: ndarray, xp: ndarray, fp: ndarray,
           right: Optional[Any]) -> ndarray:
  ...

def isfinite(x: ArrayLike) -> ArrayLike:
  ...

def linspace(start: ArrayLike, stop: ArrayLike,
             num: Optional[int] = None) -> ndarray:
  ...

def log(x: ArrayLike) -> ArrayLike:
  ...

def max(a: ArrayLike, axis: Optional[int] = None) -> ArrayLike:
  ...

def min(a: ArrayLike, axis: Optional[int] = None) -> ArrayLike:
  ...

def ones(shape: Shape, dtype: Optional[Any] = None) -> ndarray:
  ...

def reshape(a: ArrayLike, newshape: Shape) -> ArrayLike:
  ...

def sin(x: ArrayLike) -> ArrayLike:
  ...

def sum(a: ndarray,
        axis: Optional[Union[int, Tuple[int, ...]]] = None) -> ndarray:
  ...

def sqrt(x: ArrayLike) -> ArrayLike:
  ...

def where(condition: Any, x: Optional[Any] = None,
          y: Optional[Any] = None) -> ndarray:
  ...

def zeros(shape: Shape, dtype: Optional[Any] = None) -> ndarray:
  ...

def zeros_like(x: ndarray, dtype: Optional[Any] = None) -> ndarray:
  ...
