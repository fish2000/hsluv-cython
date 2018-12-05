#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import os.path
import six
# import sys

# import before Cython stuff, to avoid
# overriding Cython’s Extension class:
from psutil import cpu_count
from setuptools import setup, find_packages
from distutils.sysconfig import get_python_inc
from Cython.Distutils import Extension
from Cython.Build import cythonize
import typing as tx

UTF8_ENCODING = 'UTF-8'
TOKEN: str = ' -'

try:
    import numpy
except ImportError:
    class FakeNumpy(object):
        def get_include(self):
            return os.path.curdir
    numpy = FakeNumpy()
    print("import: NUMPY NOT FOUND (using shim)")
else:
    print(f"import: module {numpy.__name__} found")

try:
    import pythran
except ImportError:
    print("import: PYTHRAN NOT FOUND")
else:
    print(f"import: module {pythran.__name__} found")

def tuplize(*items) -> tx.Tuple[tx.Any, ...]:
    """ Return a new tuple containing all non-`None` arguments """
    return tuple(item for item in items if item is not None)

def uniquify(*items) -> tx.Tuple[tx.Any, ...]:
    """ Return a tuple with a unique set of all non-`None` arguments """
    return tuple(frozenset(item for item in items if item is not None))

def listify(*items) -> tx.List[tx.Any]:
    """ Return a new list containing all non-`None` arguments """
    return list(item for item in items if item is not None)

StringType = tx.TypeVar('StringType', bound=type, covariant=True)
PredicateType = tx.Callable[[tx.Any], bool]

string_types: tx.Tuple[tx.Type[StringType], ...] = uniquify(type(''),
                                                            type(b''),
                                                            type(f''),
                                                            type(r''),
                                                           *six.string_types)

is_string: PredicateType = lambda thing: isinstance(thing, string_types)

def u8encode(source: tx.Any) -> bytes:
    """ Encode a source as bytes using the UTF-8 codec """
    return bytes(source, encoding=UTF8_ENCODING)

def u8bytes(source: tx.Any) -> bytes:
    """ Encode a source as bytes using the UTF-8 codec, guaranteeing
        a proper return value without raising an error
    """
    if type(source) is bytes:
        return source
    elif type(source) is str:
        return u8encode(source)
    elif isinstance(source, string_types):
        return u8encode(source)
    elif isinstance(source, (int, float)):
        return u8encode(str(source))
    elif type(source) is bool:
        return source and b'True' or b'False'
    elif source is None:
        return b'None'
    return bytes(source)

def u8str(source: tx.Any) -> str:
    """ Encode a source as a Python string, guaranteeing a proper return
        value without raising an error
    """
    return type(source) is str and source \
                        or u8bytes(source).decode(UTF8_ENCODING)

MaybeStr = tx.Optional[str]
MacroTuple = tx.Tuple[str, str]

class Macro(object):
    
    __slots__: tx.ClassVar[tx.Tuple[str, ...]] = ('name',
                                                  'definition',
                                                  'undefine')
    
    STRING_ZERO: str = '0'
    STRING_ONE: str  = '1'
    
    @staticmethod
    def is_string_value(putative: tx.Any, value: int = 0) -> bool:
        """ Predicate function for checking for the values of stringified integers """
        if not is_string(putative):
            return False
        intdef: int = 0
        try:
            intdef += int(putative, base=10)
        except ValueError:
            return False
        return intdef == int(value)
    
    def __init__(self, name: str, definition: MaybeStr = None,
                                  *,
                                  undefine: bool = False):
        """ Initialize a new Macro instance, specifiying a name, a definition (optionally),
            and a boolean flag (optionally) indicating that the macro is “undefined” --
            that is to say, that it is a so-called “undef macro”.
        """
        if not name:
            raise ValueError("Macro() requires a valid name")
        string_zero: bool = self.is_string_value(definition)
        string_one: bool = self.is_string_value(definition, value=1)
        string_something: bool = string_zero or string_one
        self.name: str = name
        self.definition: MaybeStr = (not string_something or None) and definition
        self.undefine: bool = string_zero or undefine
    
    def to_string(self) -> str:
        """ Stringify the macro instance as a GCC- or Clang-compatible command-line switch,
            e.g. “DMACRO_NAME=Some_Macro_Value” -- or just “DMACRO_NAME” or “UMACRO_NAME”,
            if applicable.
        """
        if self.undefine:
            return f"U{u8str(self.name)}"
        if self.definition is not None:
            return f"D{u8str(self.name)}={u8str(self.definition)}"
        return f"D{u8str(self.name)}"
    
    def to_tuple(self) -> MacroTuple:
        """ Tuple-ize the macro instance -- return a tuple in the form (name, value)
            as per the macro’s contents. The returned tuple always has a value field;
            in the case of undefined macros, the value is '0' -- stringified zero --
            and in the case of macros lacking definition values, the value is '1' --
            stringified one.
        """
        if self.undefine:
            return (u8str(self.name),
                          self.STRING_ZERO)
        if self.definition is not None:
            return (u8str(self.name),
                    u8str(self.definition))
        return (u8str(self.name),
                      self.STRING_ONE)
    
    def __str__(self) -> str:
        return self.to_string()
    
    def __bytes__(self) -> bytes:
        return u8bytes(self.to_string())
    
    def __bool__(self) -> bool:
        """ An instance of Macro is considered Falsey if undefined, Truthy if not. """
        return not self.undefine

class Macros(tx.Dict[str, str]):
    
    __slots__: tx.ClassVar[tx.Tuple[str, ...]] = tuple()
    
    def define(self, name: str, definition: MaybeStr = None,
                                *,
                                undefine: bool = False) -> Macro:
        return self.add(Macro(name,
                              definition,
                              undefine=undefine))
    
    def undefine(self, name: str, **kwargs) -> Macro:
        return self.add(Macro(name, undefine=True))
    
    def add(self, macro: Macro) -> Macro:
        name: str = macro.name
        if bool(macro):
            # macro is defined:
            self[name] = macro.definition or Macro.STRING_ONE
        else:
            # macro is an undef macro:
            self[name] = Macro.STRING_ZERO
        return macro
    
    def delete(self, name: str, **kwargs) -> bool:
        if name in self:
            del self[name]
            return True
        return False
    
    def definition_for(self, name: str) -> Macro:
        if name not in self:
            return Macro(name, undefine=True)
        return Macro(name, self[name])
    
    def to_list(self) -> tx.List[MacroTuple]:
        out: tx.List[MacroTuple] = []
        for k, v in self.items():
            out.append(Macro(k, v).to_tuple())
        return out
    
    def to_tuple(self) -> tx.Tuple[MacroTuple, ...]:
        return tuple(self.to_list())
    
    def to_string(self) -> str:
        global TOKEN
        stringified: str = TOKEN.join(
                           Macro(k, v).to_string() for k, v in self.items()).strip()
        return f"{TOKEN.lstrip()}{stringified}"
    
    def __str__(self) -> str:
        return self.to_string()
    
    def __bytes__(self) -> bytes:
        return u8bytes(self.to_string())

# VERSION & METADATA
__version__ = "<undefined>"
exec(compile(
    open(os.path.join(
         os.path.dirname(__file__),
        '__version__.py')).read(),
        '__version__.py', 'exec'))

long_description = """
Cython-based Python bindings to HSLuv-C
"""

classifiers = [
    'Development Status :: 5 - Production',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'Topic :: Multimedia',
    'Topic :: Scientific/Engineering :: Image Recognition',
    'Topic :: Software Development :: Libraries',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: C',
    'License :: OSI Approved :: MIT License']

api_extension_sources = [os.path.join('hsluv', 'api.pyx')]
other_source_names = ('hsluv.c',)
other_sources = [os.path.join('hsluv', source) for source in other_source_names]

hsluv_base_path = os.path.abspath(os.path.dirname(__file__))

include_dirs = [
    get_python_inc(plat_specific=True),
    numpy.get_include(),
    os.path.join(hsluv_base_path, 'hsluv'),
    os.path.curdir]

macros = Macros()
macros.define('NDEBUG')
macros.define('NUMPY')
macros.define('VERSION',                 __version__)
macros.define('NPY_NO_DEPRECATED_API',  'NPY_1_7_API_VERSION')
macros.define('PY_ARRAY_UNIQUE_SYMBOL', 'YO_DOGG_I_HEARD_YOU_LIKE_UNIQUE_SYMBOLS')

setup(name='hsluv-cython',
    version=__version__,
    description=long_description,
    long_description=long_description,
    author='Alexander Bohn',
    author_email='fish2000@gmail.com',
    license='MIT',
    platforms=['Any'],
    classifiers=classifiers,
    packages=find_packages(),
    package_dir={
        'hsluv'     : 'hsluv'
    },
    package_data=dict(),
    url='http://github.com/fish2000/halogen',
    test_suite='nose.collector',
    ext_modules=cythonize([
        Extension('hsluv.api',
            api_extension_sources + other_sources,
            include_dirs=[d for d in include_dirs if os.path.isdir(d)],
            define_macros=macros.to_list(),
            extra_compile_args=[
                '-Wno-unused-function',
                '-Wno-unneeded-internal-declaration',
                '-O3',
                '-fstrict-aliasing',
                '-funroll-loops',
                '-mtune=native']
        )],
        nthreads=cpu_count(),
        compiler_directives=dict(language_level=3,
                                 infer_types=True,
                                 embedsignature=True)
    )
)
