import importlib


def test_preprocessing_module_imports():
    assert importlib.import_module("src.preprocessing")
