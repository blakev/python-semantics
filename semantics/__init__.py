#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# https://github.com/blakev/python-semantics
#
# >>
#
#   Semantic Versioning 2.0.0
#   Summary
#
#   Given a version number MAJOR.MINOR.PATCH, increment the:
#
#       MAJOR version when you make incompatible API changes,
#       MINOR version when you add functionality in a backwards-compatible manner, and
#       PATCH version when you make backwards-compatible bug fixes.
#
#   Additional labels for pre-release and build metadata are available
#    as extensions to the MAJOR.MINOR.PATCH format.
#
#   python-semantics, 2019
#
# <<

import re
import string
from enum import Enum
from collections import deque
from collections.abc import Mapping
from itertools import islice, tee, zip_longest
from functools import total_ordering
from typing import Any, Dict, Iterable, List, Tuple, Union

__author__ = 'Blake VandeMerwe'
__version__ = "0.1.0"
__all__ = [
    '__author__',
    '__version__',
    'Version',
    'VERSION'
]

EXTRA_CHARS = '-+.'
PRE_CHARS = string.ascii_letters + string.digits + EXTRA_CHARS
Regex = type(re.compile(r''))


def affix(orig: Any, ch: str) -> str:
    """Ensure a given character is prepended to some original value."""
    if isinstance(orig, (list, tuple,)):
        orig = '.'.join(map(str, orig))
    return '%s%s' % (ch, str(orig).lstrip(ch))


def sliding_window(n, seq) -> Iterable:
    """Iterate over a group of elements over a given sequence."""
    # taken from toolz.itertoolz.sliding_window
    #  https://toolz.readthedocs.io/en/latest/_modules/toolz/itertoolz.html#sliding_window
    return zip(*(deque(islice(it, i), 0) or it for i, it in enumerate(tee(seq, n))))


class Strategy(Enum):
    """Define different regular expression strategies for extracting
    semantic version information for various strings.

    ``STRICT`` adheres to SemVer spec. 2.0.0

    ``LOOSE`` is more forgiving with pre-release and metadata sections.
    """

    STRICT = re.compile(
        r'^'
        r'([1-9]\d*|0)\.(\d+)\.(\d+)'
        r'(\-[0-9a-z][0-9a-z-.]*)?'
        r'(\+[0-9a-z-.]*)?'
        r'$', re.I)

    LOOSE = re.compile(
        r'([1-9]\d*|0)\.(\d+)\.(\d+)'
        r'([.-][0-9a-z][0-9a-z-.]*)?'
        r'([-+][0-9a-z-.]*)?', re.I)


class SemanticVersionError(ValueError):
    """Library base exception class for all parsing and representation errors."""


@total_ordering
class Version:
    """Main representation of a parsed semantic version.

    Args:
        major (int):
        minor (int):
        patch (int):
        pre_release (List[str]):
        metadata (List[str]):
    """

    __slots__ = (
        '_major', '_minor', '_patch', '_pre_release', '_metadata'
    )

    def __init__(self,
                 major: int = 0,
                 minor: int = 0,
                 patch: int = 0,
                 *,
                 pre_release: List[str] = None,
                 metadata: List[str] = None):
        self._major = major
        self._minor = minor
        self._patch = patch
        self._pre_release = pre_release or []
        self._metadata = metadata or []

    def __repr__(self) -> str:
        return "<Version(value=%s, pre_release=[%s], metadata=[%s])>" % (
            self.numeric_str, ','.join(self.pre_release), ','.join(self.metadata))

    def __str__(self) -> str:
        return "%d.%d.%d%s%s" % (
            self.major,
            self.minor,
            self.patch,
            '-' + '.'.join(map(str, self.pre_release)) if self.pre_release else '',
            '+' + '.'.join(map(str, self.metadata)) if self.metadata else '')

    def __int__(self):
        return self.major

    def __bool__(self) -> bool:
        return self.stable

    def __eq__(self, other) -> bool:
        other = self.parse(other)
        return self.value == other.value and self.pre_release == other.pre_release

    def __gt__(self, other) -> bool:
        # evaluating two semantic version instances respecting SemVer v2.0.0 item 11
        #   https://semver.org/#spec-item-11
        other = self.parse(other)

        if self.value > other.value:
            return True
        elif self.value < other.value:
            return False

        a = self.pre_release
        b = other.pre_release

        if a and not b:
            return False
        elif b and not a:
            return True

        for x, y in zip_longest(a, b, fillvalue=None):
            if x is None:
                return False
            if y is None:
                return True

            x_num, y_num = x.isnumeric(), y.isnumeric()

            if x_num and y_num:
                x = int(x)
                y = int(y)

            if not x_num and y_num:
                # self.str > other.number
                return True

            if x_num and not y_num:
                # self.number < other.str
                return False

            # comparisons work for str and int
            if x == y:
                continue
            if x > y:
                return True
            else:
                return False
        raise RuntimeError('unknown error A: %s, B: %s' % (self, other))

    # <---- Representation attributes

    @property
    def major(self) -> int:
        return self._major

    @property
    def minor(self) -> int:
        return self._minor

    @property
    def patch(self) -> int:
        return self._patch

    @property
    def pre_release(self) -> List[str]:
        return self._pre_release

    @property
    def metadata(self) -> List[str]:
        return self._metadata

    @property
    def value(self) -> Tuple[int, ...]:
        """Return a customized representation of a parsed semantic version.

        This property is supposed to be overwritten and can be used for custom comparisons.
        """
        return self.numeric

    # <---- Export functionality

    @property
    def numeric(self) -> Tuple[int, int, int]:
        """The numeric N.X.Y portion of the version string."""
        return self.major, self.minor, self.patch

    @property
    def numeric_str(self) -> str:
        """Return the numeric portion N.X.Y as a string."""
        return '%d.%d.%d' % self.numeric

    def as_cmp(self) -> Tuple[int, int, int, List[str]]:
        """Tuple representation used for comparison; omits build metadata."""
        return self.major, self.minor, self.patch, self.pre_release

    def as_dict(self) -> Dict[str, Union[int, str]]:
        """Dictionary representation of the underlying data, including String."""
        return {
            '_': str(self),
            'major': self.major,
            'minor': self.minor,
            'patch': self.patch,
            'pre_release': self.pre_release,
            'metadata': self.metadata}

    # <---- Spec. defined terminology

    @property
    def stable(self) -> bool:
        """Any version greater than 0.X.Y with no pre-release data is considered "stable"."""
        return self.major > 0 and not self.unstable

    @property
    def public(self) -> bool:
        """Any version greater than 1.0.0 is considered "public" with an established API."""
        return self.major >= 1

    @property
    def unstable(self) -> bool:
        """Any version with pre-release data is considered "unstable" in SemVer."""
        return bool(self.pre_release)

    # ----- END

    @classmethod
    def evaluate(cls,
                 value: str,
                 strategy: Strategy = Strategy.LOOSE,
                 debug: bool = False) -> bool:
        """Attempt to parse a string of versions to determine if they evaluate to True.

        Notes:
            Available operators are: ``< <= == != >= >``.

        Example:

            >>> Version.evaluate("2.0.0-alpha < 2.0.0")
            True
            >>> Version.evaluate("1.0.0-alpha < 1.0.0-beta < 1.0.0-beta.1 < 1.0.0")
            True
            >>> Version.evaluate("1.0.0 > 1.0.1")
            False

        Args:
            value (str): A string of semantic versions separated by comparison operators.
            strategy (Strategy): Enum token that maps to a Regex instance. The default
                LOOSE strategy is to assist in arbitrary strings being correctly parsed
                better.
            debug (bool): When ``True`` the parsing steps are printed via stdout.

        Returns:
            bool: True if the greater expression evaluates correctly and is valid.
        """
        fn_map = {
            '<': '__lt__',
            '<=': '__le__',
            '==': '__eq__',
            '!=': '__ne__',
            '>=': '__gt__',
            '>': '__gt__'}
        values = re.sub(r'[ ]+', r' ', str(value))
        values = re.sub(r'[ ]*([<=!>]+)[ ]*', r' \1 ', values)
        values = values.strip().split(' ')

        if len(values) == 1:
            return True

        if len(values) % 2 == 0:
            raise SemanticVersionError('cannot parse this expression')

        operators = [fn_map.get(o, None) for o in islice(values, 1, None, 2)]
        versions = [cls.parse(v, strategy) for v in islice(values, 0, None, 2)]

        if None in operators:
            raise SemanticVersionError('unknown or missing comparison operator in expression')

        result = True

        for idx, a_b in enumerate(sliding_window(2, versions)):
            fn = getattr(a_b[0], operators[idx], lambda _: False)
            result = fn(a_b[1])

            if debug:
                print(idx, a_b[0].as_cmp(), operators[idx], a_b[1].as_cmp(), '=', result)
            else:
                if not result:
                    return False
        return result

    @classmethod
    def extract(cls,
                value: str,
                excess: Union[str, int, Regex] = None,
                strategy: Strategy = Strategy.LOOSE) -> 'Version':
        """Attempt to extract a semantic Version instance from an arbitrary string.

        If unsuccessful a Version placeholder is returned instead of throwing an exception.

        Args:
            value (str):
            excess (Union[int, str]):
            strategy (Strategy):

        Returns:
            Version: if successful, otherwise invalid Version instance that evaluates to 0.
        """

        if excess:
            if isinstance(excess, Regex):
                value = excess.sub('', value)

            if isinstance(excess, str):
                if value.endswith(excess):
                    excess = len(excess)

            if isinstance(excess, int):
                value = value[:-excess]

        value = value.strip(EXTRA_CHARS)
        match = strategy.value.search(value)

        if not match:
            return null_version()

        try:
            version = cls.parse(match.group(), strategy=strategy)
        except SemanticVersionError:
            return null_version()
        return version

    @classmethod
    def parse(cls,
              value: Any,
              strategy: Strategy = Strategy.STRICT) -> 'Version':
        """Takes an abstract ``value`` and converts it to an instance of Version.

        Args:
            value (Any):
            strategy (Strategy):

        Raises:
            SemanticVersionError:

        Returns:
            Version:
        """
        if isinstance(value, Version):
            return value

        if value is None:
            return null_version()

        elif isinstance(value, int):
            return cls.parse((0, value, 0))

        elif isinstance(value, (tuple, list)):
            value = list(value)
            if len(value) < 3:
                value.extend([0] * (3 - len(value)))
            if len(value) > 3:
                value[3] = affix(value[3], '-')
            if len(value) > 4:
                value[4] = affix(value[4], '+')
            if len(value) > 5:
                value[4] = '.'.join(map(str, value[4:]))
            value = [0 if v is None else v for v in value]
            value = '.'.join(map(str, value[:3])) + ''.join(map(str, value[3:5]))
            return cls.parse(value)

        elif isinstance(value, Mapping):
            pre_release = value.get('pre_release', [])
            metadata = value.get('metadata', [])
            return cls.parse('%d.%d.%d%s%s' % (
                value.get('major', 0),
                value.get('minor', 1),
                value.get('patch', 0),
                '-' + '.'.join(map(str, pre_release)) if pre_release else '',
                '+' + '.'.join(map(str, metadata)) if metadata else ''))

        if not isinstance(value, str):
            raise SemanticVersionError('unsupported value type %s' % type(value))

        if not value[0].isnumeric():
            raise SemanticVersionError('semantic version must start with a digit')

        if value[-1] in EXTRA_CHARS:
            raise SemanticVersionError('semantic version cannot end in a split token')

        if '..' in value:
            raise SemanticVersionError('empty extra identifiers is not allowed')

        matched = strategy.value.match(value)

        if not matched:
            raise SemanticVersionError('invalid version string format, %s' % value)

        major, minor, patch, pre, meta = matched.groups()

        # correct the value type
        major, minor, patch = int(major), int(minor), int(patch)

        if major == minor == 0:
            raise SemanticVersionError('minimum valid version is 0.1.0, %s' % value)

        pre = pre.strip('-.').split('.') if pre else []
        meta = meta.strip('+.').split('.') if meta else []

        # returned the parsed abstract value
        return Version(major, minor, patch, pre_release=pre, metadata=meta)


def null_version() -> Version:
    # Version factory for placeholder instances
    return Version(0, 0, 0)


VERSION = Version.parse(__version__)
# library alias

if __name__ == '__main__':  # pragma: nocoverage
    import sys

    if len(sys.argv) == 1:
        sys.exit(0)

    try:
        res = Version.evaluate(sys.argv[1])
    except SemanticVersionError:
        res = False
    except Exception as e:
        raise
    print(res)
    sys.exit(1 if not res else 0)
