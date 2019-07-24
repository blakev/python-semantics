#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# https://github.com/blakev/python-semantics
#
# >>
#
#   python-semantics, 2019
#
# <<

import unittest
from unittest import TestCase

from semantics import *


class TestVersionSemantics(TestCase):
    def test_null_version(self):
        assert Version.parse(None) == Version(0, 0, 0)
        assert Version(0, 0, 0) == None
        assert Version(0, 0, 0) is not None
        assert int(Version(0, 0, 0)) is 0

    def test_module_version(self):
        assert VERSION.numeric_str == __version__

    def test_repr(self):
        v = repr(Version(1, 0, 0))
        assert 'Version' in v
        assert 'value=' in v
        assert 'pre_release=' in v
        assert 'metadata=' in v

    def test_str_default(self):
        cases = [
            (Version(1, 0, 0), '1.0.0'),
            (Version(1, 0, 0, pre_release=['a']), '1.0.0-a'),
            (Version(1, 0, 0, metadata=['build']), '1.0.0+build'),
            (Version(1, 2, 3, metadata=['build'], pre_release=['alpha']), '1.2.3-alpha+build'),
            (Version(1, pre_release=['alpha', '1']), '1.0.0-alpha.1')
        ]

        for a, b in cases:
            assert str(a) == b, str(a)
            assert a == Version.parse(b)

    def test_exports(self):
        cases = [
            ('1.0.0', ((1, 0, 0), (1, 0, 0, [])))
        ]

        for val, checks in cases:
            v = Version.parse(val)
            assert v.numeric_str == val
            assert v.numeric == checks[0]
            assert v.as_cmp() == checks[1]

    def test_export_as_dict(self):
        pass

    def test_terminology_attrs(self):
        pass

    def test_evaluate(self):
        pass

    def test_extract(self):
        pass

    def test_parse(self):
        pass


if __name__ == '__main__':
    unittest.main()
