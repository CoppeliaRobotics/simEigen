local simEigen = loadPlugin 'simEigen';
(require 'simEigen-typecheck')(simEigen)

simEigen.Matrix = {}

-- @fun {lua_only=true} Matrix:abs compute element-wise absolute value
-- @ret table m a new matrix with result (Matrix)
function simEigen.Matrix:abs()
    return self:op(simEigen.op.abs, nil, false)
end

-- @fun {lua_only=true} Matrix:acos compute element-wise arccosine
-- @ret table m a new matrix with result (Matrix)
function simEigen.Matrix:acos()
    return self:op(simEigen.op.acos, nil, false)
end

-- @fun {lua_only=true} Matrix:add compute element-wise addition with another matrix or scalar
-- @arg table m2 the other matrix (Matrix) or a scalar (float)
-- @ret table m a new matrix with result (Matrix)
function simEigen.Matrix:add(m)
    return self:op(simEigen.op.add, m, false)
end

-- @fun {lua_only=true} Matrix:asin compute element-wise arcsine
-- @ret table m a new matrix with result (Matrix)
function simEigen.Matrix:asin()
    return self:op(simEigen.op.asin, nil, false)
end

-- @fun {lua_only=true} Matrix:atan compute element-wise arctangent
-- @ret table m a new matrix with result (Matrix)
function simEigen.Matrix:atan()
    return self:op(simEigen.op.atan, nil, false)
end

-- @fun {lua_only=true} Matrix:block return a block of this matrix
-- @arg int i start row
-- @arg int j start column
-- @arg int p block rows
-- @arg int q block columns
-- @ret table m a new matrix with result (Matrix)
function simEigen.Matrix:block(i, j, p, q)
    i = i or 1
    j = j or 1
    p = p or -1
    q = q or -1
    assert(math.type(i) == 'integer')
    assert(math.type(j) == 'integer')
    assert(math.type(p) == 'integer')
    assert(math.type(q) == 'integer')
    local m = simEigen.mtxBlock(self.__handle, i - 1, j - 1, p, q)
    m = simEigen.Matrix(m)
    return m
end

-- @fun {lua_only=true} Matrix:blockassign assign a matrix to a block of this matrix
-- @arg table m the other matrix (Matrix)
-- @arg int i start row
-- @arg int j start column
-- @arg int p block rows
-- @arg int q block columns
-- @ret table self this matrix (Matrix)
function simEigen.Matrix:blockassign(m, i, j, p, q)
    assert(simEigen.Matrix:ismatrix(m), 'argument must be a Matrix')
    i = i or 1
    j = j or 1
    p = p or -1
    q = q or -1
    assert(math.type(i) == 'integer')
    assert(math.type(j) == 'integer')
    assert(math.type(p) == 'integer')
    assert(math.type(q) == 'integer')
    simEigen.mtxBlockAssign(self.__handle, m.__handle, i - 1, j - 1, p, q)
    return self
end

-- @fun {lua_only=true} Matrix:ceil compute element-wise ceiling
-- @ret table m a new matrix with result (Matrix)
function simEigen.Matrix:ceil()
    return self:op(simEigen.op.ceil, nil, false)
end

-- @fun {lua_only=true} Matrix:col return the j-th column as a new vector
-- @arg int j column index
-- @ret table m a new matrix with result (Matrix)
function simEigen.Matrix:col(j)
    return self:block(1, j, -1, 1)
end

-- @fun {lua_only=true} Matrix:coldata get the data of the j-th column
-- @arg int j column index
-- @ret table.float a table of numbers
function simEigen.Matrix:coldata(j)
    assert(math.type(j) == 'integer')
    return simEigen.mtxGetColData(self.__handle, j - 1)
end

-- @fun {lua_only=true} Matrix:cols return the number of columns
-- @ret int number of columns
function simEigen.Matrix:cols()
    local rows, cols = simEigen.mtxGetSize(self.__handle)
    return cols
end

-- @fun {lua_only=true} Matrix:copy create a copy of this matrix
-- @ret table m a new matrix with same dimensions and data (Matrix)
function simEigen.Matrix:copy()
    local m = simEigen.mtxCopy(self.__handle)
    m = simEigen.Matrix(m)
    return m
end

-- @fun {lua_only=true} Matrix:cos compute element-wise cosine
-- @ret table m a new matrix with result (Matrix)
function simEigen.Matrix:cos()
    return self:op(simEigen.op.cos, nil, false)
end

-- @fun {lua_only=true} Matrix:count return the number of data items
-- @ret int number of data items
function simEigen.Matrix:count()
    local rows, cols = simEigen.mtxGetSize(self.__handle)
    return rows * cols
end

-- @fun {lua_only=true} Matrix:cross compute the vector cross product with 'v2'
-- @arg table v2 the other vector (Matrix)
-- @ret table v the resulting vector (Matrix)
function simEigen.Matrix:cross(m)
    assert(simEigen.Matrix:ismatrix(m), 'argument must be a Matrix')
    local r = simEigen.mtxCross(self.__handle, m.__handle)
    r = simEigen.Matrix(r)
    return r
end

-- @fun {lua_only=true} Matrix:data get the data of this matrix, in row-major order
-- @ret table.float a table of numbers
function simEigen.Matrix:data()
    return simEigen.mtxGetData(self.__handle)
end

-- @fun {lua_only=true} Matrix:deg compute element-wise radians to degree conversion
-- @ret table m a new matrix with result (Matrix)
function simEigen.Matrix:deg()
    return self:op(simEigen.op.deg, nil, false)
end

-- @fun {lua_only=true} Matrix:div compute element-wise division with another matrix or scalar
-- @arg table m2 the other matrix (Matrix) or a scalar (float)
-- @ret table m a new matrix with result (Matrix)
function simEigen.Matrix:div(m)
    return self:op(simEigen.op.div, m, false)
end

-- @fun {lua_only=true} Matrix:dot compute the dot product with 'v2'
-- @arg table v2 the other vector (Matrix)
-- @ret float the result
function simEigen.Matrix:dot(m)
    assert(simEigen.Matrix:ismatrix(m), 'argument must be a Matrix')
    return simEigen.mtxDot(self.__handle, m.__handle)
end

-- @fun {lua_only=true} Matrix:exp compute element-wise exponential
-- @ret table m a new matrix with result (Matrix)
function simEigen.Matrix:exp()
    return self:op(simEigen.op.exp, nil, false)
end

-- @fun {lua_only=true} Matrix.eye (static method) create a new identity matrix of given size
-- @arg int n size
-- @ret table m a new matrix with result (Matrix)
function simEigen.Matrix:eye(size)
    if simEigen.Matrix:ismatrix(self) then error('static method') end
    if size == nil and math.type(self) == 'integer' then size = self end
    assert(math.type(size) == 'integer', 'argument must be integer')
    local data = {}
    for i = 1, size do for j = 1, size do table.insert(data, i == j and 1 or 0) end end
    return simEigen.Matrix(size, size, data)
end

-- @fun {lua_only=true} Matrix:floor compute element-wise floor
-- @ret table m a new matrix with result (Matrix)
function simEigen.Matrix:floor()
    return self:op(simEigen.op.floor, nil, false)
end

function simEigen.Matrix:fromtable(t)
    assert(type(t) == 'table', 'bad type')
    if t.dims ~= nil and t.data ~= nil then
        assert(#t.dims == 2, 'only 2d grids are supported by this class')
        return Matrix(t.dims[1], t.dims[2], t.data)
    elseif type(t[1]) == 'table' then
        return Matrix(t)
    elseif #t == 0 then
        return Matrix(0, 0)
    end
end

function simEigen.Matrix:get(i, j)
    sim.addLog(sim.verbosity_warnings | sim.verbosity_once, 'm:get(i, j) is deprecated. use m:item(i, j) instead')
    return self:item(i, g)
end

-- @fun {lua_only=true} Matrix:horzcat stack two or more matrices horizontally
-- @arg table m2 matrix to stack (Matrix)
-- @ret table m a new matrix with result (Matrix)
function simEigen.Matrix:horzcat(...)
    local ms = {...}
    if simEigen.Matrix:ismatrix(self) then table.insert(ms, 1, self) end
    local m = simEigen.mtxHorzCat(map(function(m) return m.__handle end, ms))
    m = simEigen.Matrix(m)
    return m
end

-- @fun {lua_only=true} Matrix:iabs compute element-wise absolute value, in place
-- @ret table self this matrix (Matrix)
function simEigen.Matrix:iabs()
    return self:op(simEigen.op.abs, nil, true)
end

-- @fun {lua_only=true} Matrix:iacos compute element-wise arccosine, in place
-- @ret table self this matrix (Matrix)
function simEigen.Matrix:iacos()
    return self:op(simEigen.op.acos, nil, true)
end

-- @fun {lua_only=true} Matrix:iadd compute element-wise addition with another matrix or scalar, in place
-- @arg table m the other matrix (Matrix) or a scalar (float)
-- @ret table self this matrix (Matrix)
function simEigen.Matrix:iadd(m)
    return self:op(simEigen.op.add, m, true)
end

-- @fun {lua_only=true} Matrix:iasin compute element-wise arcsine, in place
-- @ret table self this matrix (Matrix)
function simEigen.Matrix:iasin()
    return self:op(simEigen.op.asin, nil, true)
end

-- @fun {lua_only=true} Matrix:iatan compute element-wise arctangent, in place
-- @ret table self this matrix (Matrix)
function simEigen.Matrix:iatan()
    return self:op(simEigen.op.atan, nil, true)
end

-- @fun {lua_only=true} Matrix:iceil compute element-wise ceiling, in place
-- @ret table self this matrix (Matrix)
function simEigen.Matrix:iceil()
    return self:op(simEigen.op.ceil, nil, true)
end

-- @fun {lua_only=true} Matrix:icos compute element-wise cosine, in place
-- @ret table self this matrix (Matrix)
function simEigen.Matrix:icos()
    return self:op(simEigen.op.cos, nil, true)
end

-- @fun {lua_only=true} Matrix:ideg compute element-wise radians to degrees conversion, in place
-- @ret table self this matrix (Matrix)
function simEigen.Matrix:ideg()
    return self:op(simEigen.op.deg, nil, true)
end

-- @fun {lua_only=true} Matrix:idiv compute element-wise division with another matrix or scalar, in place
-- @arg table m the other matrix (Matrix) or a scalar (float)
-- @ret table self this matrix (Matrix)
function simEigen.Matrix:idiv(m)
    return self:op(simEigen.op.div, m, true)
end

-- @fun {lua_only=true} Matrix:iexp compute element-wise exponential, in place
-- @ret table self this matrix (Matrix)
function simEigen.Matrix:iexp()
    return self:op(simEigen.op.exp, nil, true)
end

-- @fun {lua_only=true} Matrix:ifloor compute element-wise floor, in place
-- @ret table self this matrix (Matrix)
function simEigen.Matrix:ifloor()
    return self:op(simEigen.op.floor, nil, true)
end

-- @fun {lua_only=true} Matrix:iintdiv compute element-wise integer division with another matrix or scalar, in place
-- @arg table m the other matrix (Matrix) or a scalar (float)
-- @ret table self this matrix (Matrix)
function simEigen.Matrix:iintdiv(m)
    return self:op(simEigen.op.intdiv, m, true)
end

-- @fun {lua_only=true} Matrix:ilog compute element-wise natural logarithm, in place
-- @ret table self this matrix (Matrix)
function simEigen.Matrix:ilog()
    return self:op(simEigen.op.log, nil, true)
end

-- @fun {lua_only=true} Matrix:ilog2 compute element-wise base-2 logarithm, in place
-- @ret table self this matrix (Matrix)
function simEigen.Matrix:ilog2()
    return self:op(simEigen.op.log2, nil, true)
end

-- @fun {lua_only=true} Matrix:ilog10 compute element-wise base-10 logarithm, in place
-- @ret table self this matrix (Matrix)
function simEigen.Matrix:ilog10()
    return self:op(simEigen.op.log10, nil, true)
end

-- @fun {lua_only=true} Matrix:imax compute element-wise maximum with another matrix or scalar, in place
-- @arg table m the other matrix (Matrix) or a scalar (float)
-- @ret table self this matrix (Matrix)
function simEigen.Matrix:imax(m)
    return self:op(simEigen.op.max, m, true)
end

-- @fun {lua_only=true} Matrix:imin compute element-wise minimum with another matrix or scalar, in place
-- @arg table m the other matrix (Matrix) or a scalar (float)
-- @ret table self this matrix (Matrix)
function simEigen.Matrix:imin(m)
    return self:op(simEigen.op.min, m, true)
end

-- @fun {lua_only=true} Matrix:imod compute element-wise modulo with another matrix or scalar, in place
-- @arg table m the other matrix (Matrix) or a scalar (float)
-- @ret table self this matrix (Matrix)
function simEigen.Matrix:imod(m)
    return self:op(simEigen.op.mod, m, true)
end

-- @fun {lua_only=true} Matrix:imul compute matrix multiplication with another matrix or scalar, in place
-- @arg table m the other matrix (Matrix) or a scalar (float)
-- @ret table self this matrix (Matrix)
function simEigen.Matrix:imul(m)
    if type(m) == 'number' then
        return self:op(simEigen.op.times, m, true)
    end

    assert(simEigen.Matrix:ismatrix(m), 'argument must be a Matrix')
    simEigen.mtxIMul(self.__handle, m.__handle)
    return self
end

-- @fun {lua_only=true} Matrix:intdiv compute element-wise integer division with another matrix or scalar
-- @arg table m2 the other matrix (Matrix) or a scalar (float)
-- @ret table m a new matrix with result (Matrix)
function simEigen.Matrix:intdiv(m)
    return self:op(simEigen.op.intdiv, m, false)
end

-- @fun {lua_only=true} Matrix:irad compute element-wise degrees to radians conversion, in place
-- @ret table self this matrix (Matrix)
function simEigen.Matrix:irad()
    return self:op(simEigen.op.rad, nil, true)
end

-- @fun {lua_only=true} Matrix:isin compute element-wise sine, in place
-- @ret table self this matrix (Matrix)
function simEigen.Matrix:isin()
    return self:op(simEigen.op.sin, nil, true)
end

-- @fun {lua_only=true} Matrix.ismatrix (static method) check wether the argument is a simEigen.Matrix
-- @arg any m
-- @ret bool true if the argument is an instance of simEigen.Matrix
function simEigen.Matrix:ismatrix(m)
    if getmetatable(self) == simEigen.Matrix then
        -- when used as non-static returns true
        assert(m == nil, 'method does not take any arguments')
        return true
    else
        if m == nil then m = self end
        assert(m ~= nil, 'argument required')
        return getmetatable(m) == simEigen.Matrix
    end
end

-- @fun {lua_only=true} Matrix:isqrt compute element-wise square root, in place
-- @ret table self this matrix (Matrix)
function simEigen.Matrix:isqrt()
    return self:op(simEigen.op.sqrt, nil, true)
end

-- @fun {lua_only=true} Matrix:isub compute element-wise subtraction with another matrix or scalar, in place
-- @arg table m the other matrix (Matrix) or a scalar (float)
-- @ret table self this matrix (Matrix)
function simEigen.Matrix:isub(m)
    return self:op(simEigen.op.sub, m, true)
end

-- @fun {lua_only=true} Matrix:itan compute element-wise tangent, in place
-- @ret table self this matrix (Matrix)
function simEigen.Matrix:itan()
    return self:op(simEigen.op.tan, nil, true)
end

-- @fun {lua_only=true} Matrix:itimes compute element-wise multiplication with another matrix or scalar, in place
-- @arg table m the other matrix (Matrix) or a scalar (float)
-- @ret table self this matrix (Matrix)
function simEigen.Matrix:itimes(m)
    return self:op(simEigen.op.times, m, true)
end

-- @fun {lua_only=true} Matrix:item return the item at position (i, j)
-- @arg int i row index
-- @arg int j column index
-- @ret float value
function simEigen.Matrix:item(i, j)
    assert(math.type(i) == 'integer')
    assert(math.type(j) == 'integer')
    return simEigen.mtxGetItem(self.__handle, i - 1, j - 1)
end

-- @fun {lua_only=true} Matrix:kron compute kronecker product with another matrix
-- @arg table m2 the other matrix (Matrix)
-- @ret table m a new matrix with result (Matrix)
function simEigen.Matrix:kron(m)
    assert(simEigen.Matrix:ismatrix(m), 'argument must be a Matrix')
    local r = simEigen.mtxKron(self.__handle, m.__handle)
    r = simEigen.Matrix(r)
    return r
end

-- @fun {lua_only=true} Matrix:linspace create a new matrix of 'count' evenly spaced elements from 'low' to 'high'
-- @arg float low lower bound
-- @arg float high upper bound
-- @arg int count number of elements
-- @ret table m a new matrix (Matrix)
function simEigen.Matrix:linspace(low, high, count)
    if math.type(low) == 'integer' and high == nil and count == nil then
        low, high, count = 1, low, low
    end
    assert(math.type(count) == 'integer' and count > 1, 'invalid count')
    local m = simEigen.mtxLinSpaced(count, low, high)
    m = simEigen.Matrix(m)
    return m
end

-- @fun {lua_only=true} Matrix:log compute element-wise natural logarithm
-- @ret table m a new matrix with result (Matrix)
function simEigen.Matrix:log()
    return self:op(simEigen.op.log, nil, false)
end

-- @fun {lua_only=true} Matrix:log2 compute element-wise base-2 logarithm
-- @ret table m a new matrix with result (Matrix)
function simEigen.Matrix:log2()
    return self:op(simEigen.op.log2, nil, false)
end

-- @fun {lua_only=true} Matrix:log10 compute element-wise base-10 logarithm
-- @ret table m a new matrix with result (Matrix)
function simEigen.Matrix:log10()
    return self:op(simEigen.op.log10, nil, false)
end

-- @fun {lua_only=true} Matrix:max compute element-wise maximum with another matrix or scalar
-- @arg table m2 the other matrix (Matrix) or a scalar (float)
-- @ret table m a new matrix with result (Matrix)
function simEigen.Matrix:max(m)
    return self:op(simEigen.op.max, m, false)
end

-- @fun {lua_only=true} Matrix:maxcoeff compute the maximum value amongst all elements
-- @ret float result
function simEigen.Matrix:maxcoeff()
    return simEigen.mtxMaxCoeff(self.__handle)
end

-- @fun {lua_only=true} Matrix:maxcoeff compute the mean value amongst all elements
-- @ret float result
function simEigen.Matrix:mean()
    return simEigen.mtxMean(self.__handle)
end

-- @fun {lua_only=true} Matrix:min compute element-wise minimum with another matrix or scalar
-- @arg table m2 the other matrix (Matrix) or a scalar (float)
-- @ret table m a new matrix with result (Matrix)
function simEigen.Matrix:min(m)
    return self:op(simEigen.op.min, m, false)
end

-- @fun {lua_only=true} Matrix:maxcoeff compute the minimum value amongst all elements
-- @ret float result
function simEigen.Matrix:mincoeff()
    return simEigen.mtxMinCoeff(self.__handle)
end

-- @fun {lua_only=true} Matrix:mod compute element-wise modulo with another matrix or scalar
-- @arg table m2 the other matrix (Matrix) or a scalar (float)
-- @ret table m a new matrix with result (Matrix)
function simEigen.Matrix:mod(m)
    return self:op(simEigen.op.mod, m, false)
end

-- @fun {lua_only=true} Matrix:mul compute element-wise multiplication with another matrix or scalar
-- @ret table m a new matrix with result (Matrix)
function simEigen.Matrix:mul(m)
    if type(m) == 'number' then
        return self:op(simEigen.op.times, m, false)
    end

    assert(simEigen.Matrix:ismatrix(m), 'argument must be a Matrix')
    local r = simEigen.mtxMul(self.__handle, m.__handle)
    r = simEigen.Matrix(r)
    return r
end

-- @fun {lua_only=true} Matrix:norm compute the euclidean norm
-- @ret float result
function simEigen.Matrix:norm()
    return simEigen.mtxNorm(self.__handle)
end

-- @fun {lua_only=true} Matrix:normalize normalize the value of elements, in place
-- @ret table self this matrix (Matrix)
function simEigen.Matrix:normalize()
    simEigen.mtxNormalize(self.__handle)
    return self
end

-- @fun {lua_only=true} Matrix:normalized return a new matrix with element values normalized
-- @ret table m a new matrix with result (Matrix)
function simEigen.Matrix:normalized()
    local m = simEigen.mtxNormalized(self.__handle)
    m = simEigen.Matrix(m)
    return m
end

function simEigen.Matrix:op(op, x, inplace)
    local r
    if type(x) == 'number' then
        r = simEigen.mtxOpK(self.__handle, op, x, inplace)
    elseif simEigen.Matrix:ismatrix(x) or x == nil then
        r = simEigen.mtxOp(self.__handle, op, (x or {}).__handle, inplace)
    else
        error('invalid operand type')
    end
    r = inplace and self or simEigen.Matrix(r)
    return r
end

-- @fun {lua_only=true} Matrix:pinv compute the pseudo inverse of this matrix, and if 'b' is passed, return also the 'x' solution to m*x=b
-- @arg table b optional vector to solve for m*x=b
-- @arg float damping
-- @ret table m a new matrix with result (Matrix)
-- @ret table the solution to m*x=b, if b was passed (Matrix)
function simEigen.Matrix:pinv(b, damping)
    assert(b == nil or simEigen.Matrix:ismatrix(b), 'b must be a Matrix or nil')
    damping = damping or 0.0
    local m, x = simEigen.mtxPInv(self.__handle, (b or {}).__handle)
    m = simEigen.Matrix(m)
    if x then x = simEigen.Matrix(x) end
    return m, x
end

-- @fun {lua_only=true} Matrix:print print the contents of this matrix
function simEigen.Matrix:print(numToString)
    print(self:__todisplay(numToString))
end

-- @fun {lua_only=true} Matrix:prod compute the product of all elements of this matrix
-- @ret float damping
function simEigen.Matrix:prod()
    return simEigen.mtxProd(self.__handle)
end

-- @fun {lua_only=true} Matrix:rad compute element-wise degrees to radians conversion
-- @ret table m a new matrix with result (Matrix)
function simEigen.Matrix:rad()
    return self:op(simEigen.op.rad, nil, false)
end

-- @fun {lua_only=true} Matrix:reshaped return a reshaped version of this matrix
-- @arg int rows new row count
-- @arg int cols new column count
-- @ret table m a new matrix with result (Matrix)
function simEigen.Matrix:reshaped(rows, cols)
    local m = simEigen.mtxReshaped(self.__handle, rows, cols)
    m = simEigen.Matrix(m)
    return m
end

-- @fun {lua_only=true} Matrix:row return the i-th row as a new vector
-- @arg int i row index
-- @ret table m a new matrix with result (Matrix)
function simEigen.Matrix:row(i)
    return self:block(i, 1, 1, -1)
end

-- @fun {lua_only=true} Matrix:rowdata get the data of the i-th row
-- @arg int i row index
-- @ret table.float a table of numbers
function simEigen.Matrix:rowdata(i)
    assert(math.type(i) == 'integer')
    return simEigen.mtxGetRowData(self.__handle, i - 1)
end

-- @fun {lua_only=true} Matrix:rows return the number of rows
-- @ret int number of rows
function simEigen.Matrix:rows()
    local rows, cols = simEigen.mtxGetSize(self.__handle)
    return rows
end

function simEigen.Matrix:set(i, j, data)
    sim.addLog(sim.verbosity_warnings | sim.verbosity_once, 'm:set(i, j, val) is deprecated. use m:setitem(i, j, val) instead')
    return self:setitem(i, j, data)
end

-- @fun {lua_only=true} Matrix:setcol assign a vector to the j-th column
-- @arg int j column index
-- @arg table col a column vector
-- @ret table self this matrix (Matrix)
function simEigen.Matrix:setcol(j, col)
    return self:blockassign(col, 1, j, -1, 1)
end

-- @fun {lua_only=true} Matrix:setcoldata assign data to the j-th column
-- @arg int j column index
-- @arg table.float data column data
-- @ret table self this matrix (Matrix)
function simEigen.Matrix:setcoldata(j, data)
    assert(math.type(j) == 'integer')
    simEigen.mtxSetColData(self.__handle, j - 1, data)
    return self
end

-- @fun {lua_only=true} Matrix:setdata assign data to the matrix, in row-major order
-- @arg table.float data matrix data
-- @ret table self this matrix (Matrix)
function simEigen.Matrix:setdata(data)
    assert(type(data) == 'table')
    simEigen.mtxSetData(self.__handle, data)
    return self
end

-- @fun {lua_only=true} Matrix:setitem change one element in the matrix
-- @arg int i row index
-- @arg int j column index
-- @arg table.float data element value
-- @ret table self this matrix (Matrix)
function simEigen.Matrix:setitem(i, j, data)
    assert(math.type(i) == 'integer')
    assert(math.type(j) == 'integer')
    simEigen.mtxSetItem(self.__handle, i - 1, j - 1, data)
    return self
end

-- @fun {lua_only=true} Matrix:setrow assign a vector to the i-th row
-- @arg int i row index
-- @arg table row a row vector
-- @ret table m a new matrix with result (Matrix)
function simEigen.Matrix:setrow(i, row)
    return self:blockassign(row, i, 1, 1, -1)
end

-- @fun {lua_only=true} Matrix:setrowdata assign data to the i-th row
-- @arg int i row index
-- @arg table.float data row data
-- @ret table self this matrix (Matrix)
function simEigen.Matrix:setrowdata(i, data)
    assert(math.type(i) == 'integer')
    simEigen.mtxSetRowData(self.__handle, i - 1, data)
    return self
end

-- @fun {lua_only=true} Matrix:sin compute element-wise sine
-- @ret table m a new matrix with result (Matrix)
function simEigen.Matrix:sin()
    return self:op(simEigen.op.sin, nil, false)
end

-- @fun {lua_only=true} Matrix:sqrt compute element-wise square root
-- @ret table m a new matrix with result (Matrix)
function simEigen.Matrix:sqrt()
    return self:op(simEigen.op.sqrt, nil, false)
end

-- @fun {lua_only=true} Matrix:norm compute the squared euclidean norm
-- @ret float result
function simEigen.Matrix:squarednorm()
    return simEigen.mtxSquaredNorm(self.__handle)
end

-- @fun {lua_only=true} Matrix:sub compute element-wise subtraction with another matrix or scalar
-- @arg table m2 the other matrix (Matrix) or a scalar (float)
-- @ret table m a new matrix with result (Matrix)
function simEigen.Matrix:sub(m)
    return self:op(simEigen.op.sub, m, false)
end

-- @fun {lua_only=true} Matrix:prod compute the sum of all elements of this matrix
-- @ret float damping
function simEigen.Matrix:sum()
    return simEigen.mtxSum(self.__handle)
end

-- @fun {lua_only=true} Matrix:svd compute the singular value decomposition
-- @arg {type='bool',default=false} computeThinU
-- @arg {type='bool',default=false} computeThinV
-- @arg {type='table',default='nil'} b 'b' vector, optional (Matrix)
-- @ret table s (Matrix)
-- @ret table u (Matrix)
-- @ret table v (Matrix)
-- @ret table x (Matrix)
function simEigen.Matrix:svd(computeThinU, computeThinV, b)
    computeThinU = computeThinU == true
    computeThinV = computeThinV == true
    assert(b == nil or simEigen.Matrix:ismatrix(b), 'b must be a Matrix or nil')
    local s, u, v, x = simEigen.mtxSVD(self.__handle, computeThinU, computeThinV, (b or {}).__handle)
    s = simEigen.Matrix(s)
    u = simEigen.Matrix(u)
    v = simEigen.Matrix(v)
    if x then x = simEigen.Matrix(x) end
    return s, u, v, x
end

function simEigen.Matrix:t()
    return self:transposed()
end

-- @fun {lua_only=true} Matrix:tan compute element-wise tangent
-- @ret table m a new matrix with result (Matrix)
function simEigen.Matrix:tan()
    return self:op(simEigen.op.tan, nil, false)
end

-- @fun {lua_only=true} Matrix:times compute element-wise multiplication with another matrix or scalar
-- @arg table m2 the other matrix (Matrix) or a scalar (float)
-- @ret table m a new matrix with result (Matrix)
function simEigen.Matrix:times(m)
    return self:op(simEigen.op.times, m, false)
end

function simEigen.Matrix:totable(format)
    if type(format) == 'table' and #format == 0 then
        local d = {}
        for i = 1, self:rows() do
            for j = 1, self:cols() do table.insert(d, self:item(i, j)) end
        end
        return {dims = {self:rows(), self:cols()}, data = d}
    elseif format == nil then
        local t = {}
        for i = 1, self:rows() do
            local row = {}
            for j = 1, self:cols() do table.insert(row, self:item(i, j)) end
            table.insert(t, row)
        end
        return t
    end
end

-- @fun {lua_only=true} Matrix:trace compute the matrix trace
-- @ret float trace
function simEigen.Matrix:trace()
    return simEigen.mtxTrace(self.__handle)
end

-- @fun {lua_only=true} Matrix:transpose transpose the matrix, in place
-- @ret table self this matrix (Matrix)
function simEigen.Matrix:transpose()
    simEigen.mtxTranspose(self.__handle)
    return self
end

-- @fun {lua_only=true} Matrix:transposed return transposed matrix
-- @ret table m a new matrix transposed (Matrix)
function simEigen.Matrix:transposed()
    local m = simEigen.mtxTransposed(self.__handle)
    m = simEigen.Matrix(m)
    return m
end

-- @fun {lua_only=true} Matrix:vertcat stack two or more matrices vertically
-- @arg table m2 matrix to stack (Matrix)
-- @ret table m a new matrix with result (Matrix)
function simEigen.Matrix:vertcat(...)
    local ms = {...}
    if simEigen.Matrix:ismatrix(self) then table.insert(ms, 1, self) end
    local m = simEigen.mtxVertCat(map(function(m) return m.__handle end, ms))
    m = simEigen.Matrix(m)
    return m
end

function simEigen.Matrix:__add(m)
    if type(self) == 'number' then self, m = m, self end
    return self:add(m)
end

function simEigen.Matrix:__concat(m)
    return self:dot(m)
end

function simEigen.Matrix:__div(k)
    return self:div(k)
end

function simEigen.Matrix:__eq(m)
    if simEigen.Matrix:ismatrix(m) then
        return self.__handle == m.__handle
    else
        return false
    end
end

function simEigen.Matrix:__gc()
    simEigen.mtxDestroy(self.__handle)
end

function simEigen.Matrix:__idiv(k)
    return self:intdiv(k)
end

function simEigen.Matrix:__index(k)
    if math.type(k) == 'integer' then
        if self:rows() == 1 then
            return self:item(1, k)
        elseif self:cols() == 1 then
            return self:item(k, 1)
        else
            return setmetatable(
                {
                    __handle = self.__handle,
                    __row = k,
                },
                {
                    __index = function(t, j)
                        return self:item(t.__row, j)
                    end,
                    __newindex = function(t, j, v)
                        self:setitem(t.__row, j, v)
                    end,
                    __tostring = function(t)
                        return string.format('<reference to %s row %d>', t.__handle, t.__row)
                    end,
                }
            )
        end
    else
        return simEigen.Matrix[k]
    end
end

function simEigen.Matrix:__len()
    if self:rows() == 1 then
        return self:cols()
    else
        return self:rows()
    end
end

function simEigen.Matrix:__mod(k)
    return self:mod(k)
end

function simEigen.Matrix:__mul(m)
    if type(self) == 'number' then self, m = m, self end
    return self:mul(m)
end

function simEigen.Matrix:__newindex(k, v)
    if math.type(k) == 'integer' then
        if self:rows() == 1 then
            return self:setitem(1, k, v)
        elseif self:cols() == 1 then
            return self:setitem(k, 1, v)
        else
            return self:setrow(k, v)
        end
    else
        return simEigen.Matrix[k]
    end
end

function simEigen.Matrix:__pairs()
    -- for completion, return methods of simEigen.Matrix
    return pairs(simEigen.Matrix)
end

function simEigen.Matrix:__pow(m)
    return self:cross(m)
end

function simEigen.Matrix:__sub(m)
    if type(self) == 'number' then return m * (-1) + self end
    return self:sub(m)
end

function simEigen.Matrix:__tocbor(sref, stref)
    local _cbor = cbor or require 'org.conman.cbor'
    return _cbor.TYPE.ARRAY(self:totable(), sref, stref)
end

function simEigen.Matrix:__todisplay(opts)
    opts = opts or {}
    local out = ''

    opts.numToString = opts.numToString or function(x) return _S.anyToString(x) end
    local s = {}
    local colwi, colwd = {}, {}
    for i = 1, self:rows() do
        s[i] = self:rowdata(i)
        for j = 1, #s[i] do
            local ns = opts.numToString(s[i][j])
            local ns = string.split(ns .. '.', '%.')
            ns = {ns[1], ns[2]}
            if math.type(s[i][j]) == 'float' then ns[2] = '.' .. ns[2] end
            s[i][j] = ns
            colwi[j] = math.max(colwi[j] or 0, #s[i][j][1])
            colwd[j] = math.max(colwd[j] or 0, #s[i][j][2])
        end
    end

    parenthesesRenderStyles = parenthesesRenderStyles or {
        curly = {
            left  = {top = '\u{23A7}', mid = '\u{23A8}', btm = '\u{23A9}', single = '{'},
            right = {top = '\u{23AD}', mid = '\u{23AA}', btm = '\u{23AB}', single = '}'},
        },
        round = {
            left  = {top = '\u{239B}', mid = '\u{239C}', btm = '\u{239D}', single = '('},
            right = {top = '\u{239E}', mid = '\u{239F}', btm = '\u{23A0}', single = ')'},
        },
        square = {
            left  = {top = '\u{23A1}', mid = '\u{23A2}', btm = '\u{23A3}', single = '['},
            right = {top = '\u{23A4}', mid = '\u{23A5}', btm = '\u{23A6}', single = ']'},
        },
    }
    parenthesesRenderStyle = parenthesesRenderStyle or parenthesesRenderStyles.round

    for i = 1, self:rows() do
        out = out .. (i > 1 and '\n' or '')
        local tmb
        if self:rows() == 1 then
            tmb = 'single'
        elseif i == 1 then
            tmb = 'top'
        elseif i == self:rows() then
            tmb = 'btm'
        else
            tmb = 'mid'
        end
        out = out .. parenthesesRenderStyle.left[tmb] .. ' '
        for j = 1, #s[i] do
            out = out .. (j > 1 and '  ' or '')
            out = out .. string.format('%' .. colwi[j] .. 's', s[i][j][1])
            out = out .. string.format('%-' .. colwd[j] .. 's', s[i][j][2])
        end
        out = out .. ' ' .. parenthesesRenderStyle.right[tmb]
    end

    return out
end

function simEigen.Matrix:__tostring()
    local out = ''
    local rows, cols = simEigen.mtxGetSize(self.__handle)
    out = out .. 'simEigen.Matrix(' .. rows .. ', ' .. cols .. ', {'
    local data = self:data()
    for i = 0, rows - 1 do
        for j = 0, cols - 1 do
            out = out .. (i == 0 and j == 0 and '' or ', ') .. tostring(data[1 + cols * i + j])
        end
    end
    out = out .. '})'
    return out
end

function simEigen.Matrix:__unm()
    return self:op(simEigen.op.unm, nil, false)
end

-- @fun {lua_only=true} Matrix construct a new matrix; can also use the form: simEigen.Matrix{{1, 2}, {3, 4}} to construct directly from data, size will be determined automatically
-- @arg int rows number of rows
-- @arg int cols number of columns
-- @arg table.float data initialization data (optional; can also be a single value)
-- @ret table m the new matrix (Matrix)
setmetatable(
    simEigen.Matrix, {
        __call = function(self, rows, cols, data)
            local h = nil
            if type(rows) == 'string' and cols == nil and data == nil then
                -- construct from handle:
                h = rows
            elseif type(rows) == 'table' and cols == nil and data == nil then
                -- construct from 2D table data:
                assert(#rows > 0, 'invalid table data')
                assert(type(rows[1]) == 'table', 'invalid table data')
                local data = rows
                rows, cols = #data, #data[1]
                for _, row in ipairs(data) do
                    assert(type(row) == 'table', 'invalid table data')
                    assert(#row == cols, 'invalid table data')
                end
                return simEigen.Matrix(rows, cols, reduce(table.add, data, {}))
            else
                assert(math.type(rows) == 'integer', 'rows must be an integer')
                assert(math.type(cols) == 'integer', 'cols must be an integer')
                if data == nil then
                    assert(rows > 0, 'rows must be positive')
                    assert(cols > 0, 'cols must be positive')
                    h = simEigen.mtxNew(rows, cols)
                else
                    assert(rows == -1 or rows > 0, 'rows must be positive')
                    assert(cols == -1 or cols > 0, 'cols must be positive')
                    assert(not (rows == -1 and cols == -1), 'rows and cols cannot be both -1')
                    if type(data) == 'number' then data = {data} end
                    assert(type(data) == 'table' and #data > 0, 'data must be a non-empty table')
                    if rows == -1 then
                        rows = #data // cols
                    elseif cols == -1 then
                        cols = #data // rows
                    end
                    assert(#data == rows * cols or #data == 1, 'invalid number of elements')
                    h = simEigen.mtxNew(rows, cols, data)
                end
            end
            assert(h ~= nil)
            return setmetatable({__handle = h}, self)
        end,
    }
)

-- @fun {lua_only=true} Vector construct a new vector (that is: a one-column matrix); can also use the form: simEigen.Vector{1, 2, 3, 4} to construct directly from data, size will be determined automatically
-- @arg int size number of elements (matrix rows)
-- @arg table.float data initialization data (optional; can also be a single value)
-- @ret table v the new vector (Matrix)
function simEigen.Vector(v, fv)
    if type(v) == 'table' and fv == nil then
        -- construct from vector data:
        return simEigen.Matrix(-1, 1, v)
    elseif math.type(v) == 'integer' then
        -- construct from size, [fillValue]:
        return simEigen.Matrix(v, 1, fv)
    else
        error('invalid arguments')
    end
end

function simEigen.import(...)
    local names = {...}
    if #names == 1 and names[1] == '*' then
        simEigen.import('Matrix', 'Vector')
        _G.simEigen = simEigen
        return
    end
    for _, name in ipairs(names) do
        _G[name] = simEigen[name]
    end
end

return simEigen
