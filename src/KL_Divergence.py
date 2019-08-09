# This file was automatically generated by SWIG (http://www.swig.org).
# Version 3.0.12
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.

from sys import version_info as _swig_python_version_info
if _swig_python_version_info >= (2, 7, 0):
    def swig_import_helper():
        import importlib
        pkg = __name__.rpartition('.')[0]
        mname = '.'.join((pkg, '_KL_Divergence')).lstrip('.')
        try:
            return importlib.import_module(mname)
        except ImportError:
            return importlib.import_module('_KL_Divergence')
    _KL_Divergence = swig_import_helper()
    del swig_import_helper
elif _swig_python_version_info >= (2, 6, 0):
    def swig_import_helper():
        from os.path import dirname
        import imp
        fp = None
        try:
            fp, pathname, description = imp.find_module('_KL_Divergence', [dirname(__file__)])
        except ImportError:
            import _KL_Divergence
            return _KL_Divergence
        try:
            _mod = imp.load_module('_KL_Divergence', fp, pathname, description)
        finally:
            if fp is not None:
                fp.close()
        return _mod
    _KL_Divergence = swig_import_helper()
    del swig_import_helper
else:
    import _KL_Divergence
del _swig_python_version_info

try:
    _swig_property = property
except NameError:
    pass  # Python < 2.2 doesn't have 'property'.

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

def _swig_setattr_nondynamic(self, class_type, name, value, static=1):
    if (name == "thisown"):
        return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'SwigPyObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name, None)
    if method:
        return method(self, value)
    if (not static):
        if _newclass:
            object.__setattr__(self, name, value)
        else:
            self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)


def _swig_setattr(self, class_type, name, value):
    return _swig_setattr_nondynamic(self, class_type, name, value, 0)


def _swig_getattr(self, class_type, name):
    if (name == "thisown"):
        return self.this.own()
    method = class_type.__swig_getmethods__.get(name, None)
    if method:
        return method(self)
    raise AttributeError("'%s' object has no attribute '%s'" % (class_type.__name__, name))


def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except __builtin__.Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)

try:
    _object = object
    _newclass = 1
except __builtin__.Exception:
    class _object:
        pass
    _newclass = 0

class SwigPyIterator(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, SwigPyIterator, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, SwigPyIterator, name)

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined - class is abstract")
    __repr__ = _swig_repr
    __swig_destroy__ = _KL_Divergence.delete_SwigPyIterator
    __del__ = lambda self: None

    def value(self):
        return _KL_Divergence.SwigPyIterator_value(self)

    def incr(self, n=1):
        return _KL_Divergence.SwigPyIterator_incr(self, n)

    def decr(self, n=1):
        return _KL_Divergence.SwigPyIterator_decr(self, n)

    def distance(self, x):
        return _KL_Divergence.SwigPyIterator_distance(self, x)

    def equal(self, x):
        return _KL_Divergence.SwigPyIterator_equal(self, x)

    def copy(self):
        return _KL_Divergence.SwigPyIterator_copy(self)

    def next(self):
        return _KL_Divergence.SwigPyIterator_next(self)

    def __next__(self):
        return _KL_Divergence.SwigPyIterator___next__(self)

    def previous(self):
        return _KL_Divergence.SwigPyIterator_previous(self)

    def advance(self, n):
        return _KL_Divergence.SwigPyIterator_advance(self, n)

    def __eq__(self, x):
        return _KL_Divergence.SwigPyIterator___eq__(self, x)

    def __ne__(self, x):
        return _KL_Divergence.SwigPyIterator___ne__(self, x)

    def __iadd__(self, n):
        return _KL_Divergence.SwigPyIterator___iadd__(self, n)

    def __isub__(self, n):
        return _KL_Divergence.SwigPyIterator___isub__(self, n)

    def __add__(self, n):
        return _KL_Divergence.SwigPyIterator___add__(self, n)

    def __sub__(self, *args):
        return _KL_Divergence.SwigPyIterator___sub__(self, *args)
    def __iter__(self):
        return self
SwigPyIterator_swigregister = _KL_Divergence.SwigPyIterator_swigregister
SwigPyIterator_swigregister(SwigPyIterator)

class VectorInt(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, VectorInt, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, VectorInt, name)
    __repr__ = _swig_repr

    def iterator(self):
        return _KL_Divergence.VectorInt_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _KL_Divergence.VectorInt___nonzero__(self)

    def __bool__(self):
        return _KL_Divergence.VectorInt___bool__(self)

    def __len__(self):
        return _KL_Divergence.VectorInt___len__(self)

    def __getslice__(self, i, j):
        return _KL_Divergence.VectorInt___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _KL_Divergence.VectorInt___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _KL_Divergence.VectorInt___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _KL_Divergence.VectorInt___delitem__(self, *args)

    def __getitem__(self, *args):
        return _KL_Divergence.VectorInt___getitem__(self, *args)

    def __setitem__(self, *args):
        return _KL_Divergence.VectorInt___setitem__(self, *args)

    def pop(self):
        return _KL_Divergence.VectorInt_pop(self)

    def append(self, x):
        return _KL_Divergence.VectorInt_append(self, x)

    def empty(self):
        return _KL_Divergence.VectorInt_empty(self)

    def size(self):
        return _KL_Divergence.VectorInt_size(self)

    def swap(self, v):
        return _KL_Divergence.VectorInt_swap(self, v)

    def begin(self):
        return _KL_Divergence.VectorInt_begin(self)

    def end(self):
        return _KL_Divergence.VectorInt_end(self)

    def rbegin(self):
        return _KL_Divergence.VectorInt_rbegin(self)

    def rend(self):
        return _KL_Divergence.VectorInt_rend(self)

    def clear(self):
        return _KL_Divergence.VectorInt_clear(self)

    def get_allocator(self):
        return _KL_Divergence.VectorInt_get_allocator(self)

    def pop_back(self):
        return _KL_Divergence.VectorInt_pop_back(self)

    def erase(self, *args):
        return _KL_Divergence.VectorInt_erase(self, *args)

    def __init__(self, *args):
        this = _KL_Divergence.new_VectorInt(*args)
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this

    def push_back(self, x):
        return _KL_Divergence.VectorInt_push_back(self, x)

    def front(self):
        return _KL_Divergence.VectorInt_front(self)

    def back(self):
        return _KL_Divergence.VectorInt_back(self)

    def assign(self, n, x):
        return _KL_Divergence.VectorInt_assign(self, n, x)

    def resize(self, *args):
        return _KL_Divergence.VectorInt_resize(self, *args)

    def insert(self, *args):
        return _KL_Divergence.VectorInt_insert(self, *args)

    def reserve(self, n):
        return _KL_Divergence.VectorInt_reserve(self, n)

    def capacity(self):
        return _KL_Divergence.VectorInt_capacity(self)
    __swig_destroy__ = _KL_Divergence.delete_VectorInt
    __del__ = lambda self: None
VectorInt_swigregister = _KL_Divergence.VectorInt_swigregister
VectorInt_swigregister(VectorInt)

class VectorVectorInt(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, VectorVectorInt, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, VectorVectorInt, name)
    __repr__ = _swig_repr

    def iterator(self):
        return _KL_Divergence.VectorVectorInt_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _KL_Divergence.VectorVectorInt___nonzero__(self)

    def __bool__(self):
        return _KL_Divergence.VectorVectorInt___bool__(self)

    def __len__(self):
        return _KL_Divergence.VectorVectorInt___len__(self)

    def __getslice__(self, i, j):
        return _KL_Divergence.VectorVectorInt___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _KL_Divergence.VectorVectorInt___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _KL_Divergence.VectorVectorInt___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _KL_Divergence.VectorVectorInt___delitem__(self, *args)

    def __getitem__(self, *args):
        return _KL_Divergence.VectorVectorInt___getitem__(self, *args)

    def __setitem__(self, *args):
        return _KL_Divergence.VectorVectorInt___setitem__(self, *args)

    def pop(self):
        return _KL_Divergence.VectorVectorInt_pop(self)

    def append(self, x):
        return _KL_Divergence.VectorVectorInt_append(self, x)

    def empty(self):
        return _KL_Divergence.VectorVectorInt_empty(self)

    def size(self):
        return _KL_Divergence.VectorVectorInt_size(self)

    def swap(self, v):
        return _KL_Divergence.VectorVectorInt_swap(self, v)

    def begin(self):
        return _KL_Divergence.VectorVectorInt_begin(self)

    def end(self):
        return _KL_Divergence.VectorVectorInt_end(self)

    def rbegin(self):
        return _KL_Divergence.VectorVectorInt_rbegin(self)

    def rend(self):
        return _KL_Divergence.VectorVectorInt_rend(self)

    def clear(self):
        return _KL_Divergence.VectorVectorInt_clear(self)

    def get_allocator(self):
        return _KL_Divergence.VectorVectorInt_get_allocator(self)

    def pop_back(self):
        return _KL_Divergence.VectorVectorInt_pop_back(self)

    def erase(self, *args):
        return _KL_Divergence.VectorVectorInt_erase(self, *args)

    def __init__(self, *args):
        this = _KL_Divergence.new_VectorVectorInt(*args)
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this

    def push_back(self, x):
        return _KL_Divergence.VectorVectorInt_push_back(self, x)

    def front(self):
        return _KL_Divergence.VectorVectorInt_front(self)

    def back(self):
        return _KL_Divergence.VectorVectorInt_back(self)

    def assign(self, n, x):
        return _KL_Divergence.VectorVectorInt_assign(self, n, x)

    def resize(self, *args):
        return _KL_Divergence.VectorVectorInt_resize(self, *args)

    def insert(self, *args):
        return _KL_Divergence.VectorVectorInt_insert(self, *args)

    def reserve(self, n):
        return _KL_Divergence.VectorVectorInt_reserve(self, n)

    def capacity(self):
        return _KL_Divergence.VectorVectorInt_capacity(self)
    __swig_destroy__ = _KL_Divergence.delete_VectorVectorInt
    __del__ = lambda self: None
VectorVectorInt_swigregister = _KL_Divergence.VectorVectorInt_swigregister
VectorVectorInt_swigregister(VectorVectorInt)

class VectorShort(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, VectorShort, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, VectorShort, name)
    __repr__ = _swig_repr

    def iterator(self):
        return _KL_Divergence.VectorShort_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _KL_Divergence.VectorShort___nonzero__(self)

    def __bool__(self):
        return _KL_Divergence.VectorShort___bool__(self)

    def __len__(self):
        return _KL_Divergence.VectorShort___len__(self)

    def __getslice__(self, i, j):
        return _KL_Divergence.VectorShort___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _KL_Divergence.VectorShort___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _KL_Divergence.VectorShort___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _KL_Divergence.VectorShort___delitem__(self, *args)

    def __getitem__(self, *args):
        return _KL_Divergence.VectorShort___getitem__(self, *args)

    def __setitem__(self, *args):
        return _KL_Divergence.VectorShort___setitem__(self, *args)

    def pop(self):
        return _KL_Divergence.VectorShort_pop(self)

    def append(self, x):
        return _KL_Divergence.VectorShort_append(self, x)

    def empty(self):
        return _KL_Divergence.VectorShort_empty(self)

    def size(self):
        return _KL_Divergence.VectorShort_size(self)

    def swap(self, v):
        return _KL_Divergence.VectorShort_swap(self, v)

    def begin(self):
        return _KL_Divergence.VectorShort_begin(self)

    def end(self):
        return _KL_Divergence.VectorShort_end(self)

    def rbegin(self):
        return _KL_Divergence.VectorShort_rbegin(self)

    def rend(self):
        return _KL_Divergence.VectorShort_rend(self)

    def clear(self):
        return _KL_Divergence.VectorShort_clear(self)

    def get_allocator(self):
        return _KL_Divergence.VectorShort_get_allocator(self)

    def pop_back(self):
        return _KL_Divergence.VectorShort_pop_back(self)

    def erase(self, *args):
        return _KL_Divergence.VectorShort_erase(self, *args)

    def __init__(self, *args):
        this = _KL_Divergence.new_VectorShort(*args)
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this

    def push_back(self, x):
        return _KL_Divergence.VectorShort_push_back(self, x)

    def front(self):
        return _KL_Divergence.VectorShort_front(self)

    def back(self):
        return _KL_Divergence.VectorShort_back(self)

    def assign(self, n, x):
        return _KL_Divergence.VectorShort_assign(self, n, x)

    def resize(self, *args):
        return _KL_Divergence.VectorShort_resize(self, *args)

    def insert(self, *args):
        return _KL_Divergence.VectorShort_insert(self, *args)

    def reserve(self, n):
        return _KL_Divergence.VectorShort_reserve(self, n)

    def capacity(self):
        return _KL_Divergence.VectorShort_capacity(self)
    __swig_destroy__ = _KL_Divergence.delete_VectorShort
    __del__ = lambda self: None
VectorShort_swigregister = _KL_Divergence.VectorShort_swigregister
VectorShort_swigregister(VectorShort)

class VectorDouble(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, VectorDouble, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, VectorDouble, name)
    __repr__ = _swig_repr

    def iterator(self):
        return _KL_Divergence.VectorDouble_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _KL_Divergence.VectorDouble___nonzero__(self)

    def __bool__(self):
        return _KL_Divergence.VectorDouble___bool__(self)

    def __len__(self):
        return _KL_Divergence.VectorDouble___len__(self)

    def __getslice__(self, i, j):
        return _KL_Divergence.VectorDouble___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _KL_Divergence.VectorDouble___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _KL_Divergence.VectorDouble___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _KL_Divergence.VectorDouble___delitem__(self, *args)

    def __getitem__(self, *args):
        return _KL_Divergence.VectorDouble___getitem__(self, *args)

    def __setitem__(self, *args):
        return _KL_Divergence.VectorDouble___setitem__(self, *args)

    def pop(self):
        return _KL_Divergence.VectorDouble_pop(self)

    def append(self, x):
        return _KL_Divergence.VectorDouble_append(self, x)

    def empty(self):
        return _KL_Divergence.VectorDouble_empty(self)

    def size(self):
        return _KL_Divergence.VectorDouble_size(self)

    def swap(self, v):
        return _KL_Divergence.VectorDouble_swap(self, v)

    def begin(self):
        return _KL_Divergence.VectorDouble_begin(self)

    def end(self):
        return _KL_Divergence.VectorDouble_end(self)

    def rbegin(self):
        return _KL_Divergence.VectorDouble_rbegin(self)

    def rend(self):
        return _KL_Divergence.VectorDouble_rend(self)

    def clear(self):
        return _KL_Divergence.VectorDouble_clear(self)

    def get_allocator(self):
        return _KL_Divergence.VectorDouble_get_allocator(self)

    def pop_back(self):
        return _KL_Divergence.VectorDouble_pop_back(self)

    def erase(self, *args):
        return _KL_Divergence.VectorDouble_erase(self, *args)

    def __init__(self, *args):
        this = _KL_Divergence.new_VectorDouble(*args)
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this

    def push_back(self, x):
        return _KL_Divergence.VectorDouble_push_back(self, x)

    def front(self):
        return _KL_Divergence.VectorDouble_front(self)

    def back(self):
        return _KL_Divergence.VectorDouble_back(self)

    def assign(self, n, x):
        return _KL_Divergence.VectorDouble_assign(self, n, x)

    def resize(self, *args):
        return _KL_Divergence.VectorDouble_resize(self, *args)

    def insert(self, *args):
        return _KL_Divergence.VectorDouble_insert(self, *args)

    def reserve(self, n):
        return _KL_Divergence.VectorDouble_reserve(self, n)

    def capacity(self):
        return _KL_Divergence.VectorDouble_capacity(self)
    __swig_destroy__ = _KL_Divergence.delete_VectorDouble
    __del__ = lambda self: None
VectorDouble_swigregister = _KL_Divergence.VectorDouble_swigregister
VectorDouble_swigregister(VectorDouble)

class VectorVectorShort(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, VectorVectorShort, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, VectorVectorShort, name)
    __repr__ = _swig_repr

    def iterator(self):
        return _KL_Divergence.VectorVectorShort_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _KL_Divergence.VectorVectorShort___nonzero__(self)

    def __bool__(self):
        return _KL_Divergence.VectorVectorShort___bool__(self)

    def __len__(self):
        return _KL_Divergence.VectorVectorShort___len__(self)

    def __getslice__(self, i, j):
        return _KL_Divergence.VectorVectorShort___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _KL_Divergence.VectorVectorShort___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _KL_Divergence.VectorVectorShort___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _KL_Divergence.VectorVectorShort___delitem__(self, *args)

    def __getitem__(self, *args):
        return _KL_Divergence.VectorVectorShort___getitem__(self, *args)

    def __setitem__(self, *args):
        return _KL_Divergence.VectorVectorShort___setitem__(self, *args)

    def pop(self):
        return _KL_Divergence.VectorVectorShort_pop(self)

    def append(self, x):
        return _KL_Divergence.VectorVectorShort_append(self, x)

    def empty(self):
        return _KL_Divergence.VectorVectorShort_empty(self)

    def size(self):
        return _KL_Divergence.VectorVectorShort_size(self)

    def swap(self, v):
        return _KL_Divergence.VectorVectorShort_swap(self, v)

    def begin(self):
        return _KL_Divergence.VectorVectorShort_begin(self)

    def end(self):
        return _KL_Divergence.VectorVectorShort_end(self)

    def rbegin(self):
        return _KL_Divergence.VectorVectorShort_rbegin(self)

    def rend(self):
        return _KL_Divergence.VectorVectorShort_rend(self)

    def clear(self):
        return _KL_Divergence.VectorVectorShort_clear(self)

    def get_allocator(self):
        return _KL_Divergence.VectorVectorShort_get_allocator(self)

    def pop_back(self):
        return _KL_Divergence.VectorVectorShort_pop_back(self)

    def erase(self, *args):
        return _KL_Divergence.VectorVectorShort_erase(self, *args)

    def __init__(self, *args):
        this = _KL_Divergence.new_VectorVectorShort(*args)
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this

    def push_back(self, x):
        return _KL_Divergence.VectorVectorShort_push_back(self, x)

    def front(self):
        return _KL_Divergence.VectorVectorShort_front(self)

    def back(self):
        return _KL_Divergence.VectorVectorShort_back(self)

    def assign(self, n, x):
        return _KL_Divergence.VectorVectorShort_assign(self, n, x)

    def resize(self, *args):
        return _KL_Divergence.VectorVectorShort_resize(self, *args)

    def insert(self, *args):
        return _KL_Divergence.VectorVectorShort_insert(self, *args)

    def reserve(self, n):
        return _KL_Divergence.VectorVectorShort_reserve(self, n)

    def capacity(self):
        return _KL_Divergence.VectorVectorShort_capacity(self)
    __swig_destroy__ = _KL_Divergence.delete_VectorVectorShort
    __del__ = lambda self: None
VectorVectorShort_swigregister = _KL_Divergence.VectorVectorShort_swigregister
VectorVectorShort_swigregister(VectorVectorShort)

class VectorBool(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, VectorBool, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, VectorBool, name)
    __repr__ = _swig_repr

    def iterator(self):
        return _KL_Divergence.VectorBool_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _KL_Divergence.VectorBool___nonzero__(self)

    def __bool__(self):
        return _KL_Divergence.VectorBool___bool__(self)

    def __len__(self):
        return _KL_Divergence.VectorBool___len__(self)

    def __getslice__(self, i, j):
        return _KL_Divergence.VectorBool___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _KL_Divergence.VectorBool___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _KL_Divergence.VectorBool___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _KL_Divergence.VectorBool___delitem__(self, *args)

    def __getitem__(self, *args):
        return _KL_Divergence.VectorBool___getitem__(self, *args)

    def __setitem__(self, *args):
        return _KL_Divergence.VectorBool___setitem__(self, *args)

    def pop(self):
        return _KL_Divergence.VectorBool_pop(self)

    def append(self, x):
        return _KL_Divergence.VectorBool_append(self, x)

    def empty(self):
        return _KL_Divergence.VectorBool_empty(self)

    def size(self):
        return _KL_Divergence.VectorBool_size(self)

    def swap(self, v):
        return _KL_Divergence.VectorBool_swap(self, v)

    def begin(self):
        return _KL_Divergence.VectorBool_begin(self)

    def end(self):
        return _KL_Divergence.VectorBool_end(self)

    def rbegin(self):
        return _KL_Divergence.VectorBool_rbegin(self)

    def rend(self):
        return _KL_Divergence.VectorBool_rend(self)

    def clear(self):
        return _KL_Divergence.VectorBool_clear(self)

    def get_allocator(self):
        return _KL_Divergence.VectorBool_get_allocator(self)

    def pop_back(self):
        return _KL_Divergence.VectorBool_pop_back(self)

    def erase(self, *args):
        return _KL_Divergence.VectorBool_erase(self, *args)

    def __init__(self, *args):
        this = _KL_Divergence.new_VectorBool(*args)
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this

    def push_back(self, x):
        return _KL_Divergence.VectorBool_push_back(self, x)

    def front(self):
        return _KL_Divergence.VectorBool_front(self)

    def back(self):
        return _KL_Divergence.VectorBool_back(self)

    def assign(self, n, x):
        return _KL_Divergence.VectorBool_assign(self, n, x)

    def resize(self, *args):
        return _KL_Divergence.VectorBool_resize(self, *args)

    def insert(self, *args):
        return _KL_Divergence.VectorBool_insert(self, *args)

    def reserve(self, n):
        return _KL_Divergence.VectorBool_reserve(self, n)

    def capacity(self):
        return _KL_Divergence.VectorBool_capacity(self)
    __swig_destroy__ = _KL_Divergence.delete_VectorBool
    __del__ = lambda self: None
VectorBool_swigregister = _KL_Divergence.VectorBool_swigregister
VectorBool_swigregister(VectorBool)

class VectorVectorBool(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, VectorVectorBool, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, VectorVectorBool, name)
    __repr__ = _swig_repr

    def iterator(self):
        return _KL_Divergence.VectorVectorBool_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _KL_Divergence.VectorVectorBool___nonzero__(self)

    def __bool__(self):
        return _KL_Divergence.VectorVectorBool___bool__(self)

    def __len__(self):
        return _KL_Divergence.VectorVectorBool___len__(self)

    def __getslice__(self, i, j):
        return _KL_Divergence.VectorVectorBool___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _KL_Divergence.VectorVectorBool___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _KL_Divergence.VectorVectorBool___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _KL_Divergence.VectorVectorBool___delitem__(self, *args)

    def __getitem__(self, *args):
        return _KL_Divergence.VectorVectorBool___getitem__(self, *args)

    def __setitem__(self, *args):
        return _KL_Divergence.VectorVectorBool___setitem__(self, *args)

    def pop(self):
        return _KL_Divergence.VectorVectorBool_pop(self)

    def append(self, x):
        return _KL_Divergence.VectorVectorBool_append(self, x)

    def empty(self):
        return _KL_Divergence.VectorVectorBool_empty(self)

    def size(self):
        return _KL_Divergence.VectorVectorBool_size(self)

    def swap(self, v):
        return _KL_Divergence.VectorVectorBool_swap(self, v)

    def begin(self):
        return _KL_Divergence.VectorVectorBool_begin(self)

    def end(self):
        return _KL_Divergence.VectorVectorBool_end(self)

    def rbegin(self):
        return _KL_Divergence.VectorVectorBool_rbegin(self)

    def rend(self):
        return _KL_Divergence.VectorVectorBool_rend(self)

    def clear(self):
        return _KL_Divergence.VectorVectorBool_clear(self)

    def get_allocator(self):
        return _KL_Divergence.VectorVectorBool_get_allocator(self)

    def pop_back(self):
        return _KL_Divergence.VectorVectorBool_pop_back(self)

    def erase(self, *args):
        return _KL_Divergence.VectorVectorBool_erase(self, *args)

    def __init__(self, *args):
        this = _KL_Divergence.new_VectorVectorBool(*args)
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this

    def push_back(self, x):
        return _KL_Divergence.VectorVectorBool_push_back(self, x)

    def front(self):
        return _KL_Divergence.VectorVectorBool_front(self)

    def back(self):
        return _KL_Divergence.VectorVectorBool_back(self)

    def assign(self, n, x):
        return _KL_Divergence.VectorVectorBool_assign(self, n, x)

    def resize(self, *args):
        return _KL_Divergence.VectorVectorBool_resize(self, *args)

    def insert(self, *args):
        return _KL_Divergence.VectorVectorBool_insert(self, *args)

    def reserve(self, n):
        return _KL_Divergence.VectorVectorBool_reserve(self, n)

    def capacity(self):
        return _KL_Divergence.VectorVectorBool_capacity(self)
    __swig_destroy__ = _KL_Divergence.delete_VectorVectorBool
    __del__ = lambda self: None
VectorVectorBool_swigregister = _KL_Divergence.VectorVectorBool_swigregister
VectorVectorBool_swigregister(VectorVectorBool)

class PairIntUint(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, PairIntUint, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, PairIntUint, name)
    __repr__ = _swig_repr

    def __init__(self, *args):
        this = _KL_Divergence.new_PairIntUint(*args)
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this
    __swig_setmethods__["first"] = _KL_Divergence.PairIntUint_first_set
    __swig_getmethods__["first"] = _KL_Divergence.PairIntUint_first_get
    if _newclass:
        first = _swig_property(_KL_Divergence.PairIntUint_first_get, _KL_Divergence.PairIntUint_first_set)
    __swig_setmethods__["second"] = _KL_Divergence.PairIntUint_second_set
    __swig_getmethods__["second"] = _KL_Divergence.PairIntUint_second_get
    if _newclass:
        second = _swig_property(_KL_Divergence.PairIntUint_second_get, _KL_Divergence.PairIntUint_second_set)
    def __len__(self):
        return 2
    def __repr__(self):
        return str((self.first, self.second))
    def __getitem__(self, index): 
        if not (index % 2):
            return self.first
        else:
            return self.second
    def __setitem__(self, index, val):
        if not (index % 2):
            self.first = val
        else:
            self.second = val
    __swig_destroy__ = _KL_Divergence.delete_PairIntUint
    __del__ = lambda self: None
PairIntUint_swigregister = _KL_Divergence.PairIntUint_swigregister
PairIntUint_swigregister(PairIntUint)

class KL_Divergence(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, KL_Divergence, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, KL_Divergence, name)
    __repr__ = _swig_repr

    def __init__(self):
        this = _KL_Divergence.new_KL_Divergence()
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this
    __swig_destroy__ = _KL_Divergence.delete_KL_Divergence
    __del__ = lambda self: None

    def kl_div(self, p, q):
        return _KL_Divergence.KL_Divergence_kl_div(self, p, q)
KL_Divergence_swigregister = _KL_Divergence.KL_Divergence_swigregister
KL_Divergence_swigregister(KL_Divergence)

# This file is compatible with both classic and new-style classes.

