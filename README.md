# Eigen plugin for CoppeliaSim

Eigen plugin.

### Porting from coppeliaSim's legacy matrix.lua library

This library is mostly compatible with the old matrix.lua library.

Replace:

```
local matrix = require 'matrix-2'
```

with:

```
local matrix = require 'simEigen'
```

The main difference from `matrix.lua` is that the functions in this library are all (or almost) implemented in C++.

#### In place VS. return a copy

There are both functions that operate in place (i.e.: on the current matrix) and functions that return a new matrix, so it is possible to optimize code even further, by avoiding too many memory allocations.

#### Added/changed/removed methods

There are a few methods that have different names and different semantics with respect to `matrix.lua`, or have been added/removed:

 - `:sameshape()` / `:offset()` not present
 - `:get()` deprecated; use `:item()`
 - `:set()` deprecated; use `:setitem()`
 - `:rowref()` / `:dataref()` not present; also, you can't assign with `m[i][j]`, but have to use `:setitem()` (operation is anyways slow, and a vectorized operation would be preferable;
 - `:setrow()` / `:setcol()` work also with plain tables;
 - `:slice()` not present; use `:block()` or `:blockassign()` which have different order of parameters and different semantics
 - `:flip()` not present
 - `:droprow()` / `:dropcol()` not present
 - `:at()` not present
 - `:repmat()` / `:ones()` / `:zeros()` not present
 - `:applyfunc()` / `:applyfunc2()` / `:applyfuncidx()` not present
 - `:binop()` / `:fold()` / `:mult()` not present
 - `:random()` not present
 - `:tointeger()` not present
 - `:gauss()` not present
 - `:inv()` not present; use `:pinv()`
 - `:ult()` / `:power()` / `:diag()` not present
 - `:axis()` / `:hom()` / `:nonhom()` not present
 - `:eq()` / `:ne()` / `:lt()` / `:gt()` / `:le()` / `:ge()` / `:all()` / `:any()` / `:isnan()` / `:nonzero()` / `:where()` not present
 - `m.t()` can be also written as `m.T`, and `m.transpose()` transposes in place

 - `Vector:linspace()` now is in `simEigen.Matrix`; `:logspace()` / `:geomspace()` not present
 - classes `Vector3` / `Vector4` / `Vector7` / `Matrix3x3` / `Matrix4x4` removed or replaced by `Vector` / `Quaternion` / `Pose`
### Compiling

1. Install required packages for simStubsGen: see simStubsGen's [README](https://github.com/CoppeliaRobotics/include/blob/master/simStubsGen/README.md)
2. Checkout, compile and install into CoppeliaSim:
```sh
$ git clone https://github.com/CoppeliaRobotics/simEigen.git
$ cd simEigen
$ git checkout coppeliasim-v4.5.0-rev0
$ mkdir -p build && cd build
$ cmake -DCMAKE_BUILD_TYPE=Release ..
$ cmake --build .
$ cmake --install .
```

NOTE: replace `coppeliasim-v4.5.0-rev0` with the actual CoppeliaSim version you have.
