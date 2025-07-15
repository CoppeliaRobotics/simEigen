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

 - `m:sameshape()` / `m:offset()` not present
 - `m:get()` deprecated; use `m:item()`
 - `m:set()` deprecated; use `m:setitem()`
 - `m:rowref()` / `m:dataref()` not present (operation is anyways slow, and a vectorized operation would be preferable;
 - `m:setrow()` / `m:setcol()` work also with plain tables;
 - `m:slice()` not present; use `m:block()` or `m:blockassign()` which have different order of parameters and different semantics
 - `m:flip()` not present
 - `m:droprow()` / `m:dropcol()` not present
 - `m:at()` not present
 - `m:repmat()` / `m:ones()` / `m:zeros()` not present
 - `m:applyfunc()` / `m:applyfunc2()` / `m:applyfuncidx()` not present
 - `m:binop()` / `m:fold()` / `m:mult()` not present
 - `m:random()` not present
 - `m:tointeger()` not present
 - `m:gauss()` not present
 - `m:inv()` not present; use `m:pinv()`
 - `m:ult()` / `m:power()` / `m:diag()` not present
 - `m:axis()` / `m:hom()` / `m:nonhom()` not present
 - `m:eq()` / `m:ne()` / `m:lt()` / `m:gt()` / `m:le()` / `m:ge()` / `m:all()` / `m:any()` / `m:isnan()` / `m:nonzero()` / `m:where()` not present
 - `m:t()` can be also written as `m.T`, and `m:transpose()` transposes in place

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
