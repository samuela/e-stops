"""This is numpy only (no JAX stuff)."""

from numpy import ndarray

# pylint: disable=unused-argument, redefined-builtin

def seed(n: int):
  ...

def rand(*shape: int) -> ndarray:
  ...
