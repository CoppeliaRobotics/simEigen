local simEigen = loadPlugin 'simEigen';
(require 'simEigen-typecheck')(simEigen)

simEigen.Matrix = {}

function simEigen.Matrix:add(m)
    assert(simEigen.Matrix:ismatrix(m), 'argument must be a Matrix')
    local r = simEigen.mtxAdd(self.__handle, m.__handle)
    r = simEigen.Matrix(r)
    return r
end

function simEigen.Matrix:addk(k)
    assert(type(k) == 'number', 'argument must be a number')
    local r = simEigen.mtxAddK(self.__handle, k)
    r = simEigen.Matrix(r)
    return r
end

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
end

function simEigen.Matrix:col(j)
    return self:block(1, j, -1, 1)
end

function simEigen.Matrix:coldata(j)
    assert(math.type(j) == 'integer')
    return simEigen.mtxGetCol(self.__handle, j - 1)
end

function simEigen.Matrix:cols()
    local rows, cols = simEigen.mtxGetSize(self.__handle)
    return cols
end

function simEigen.Matrix:copy()
    local m = simEigen.mtxCopy(self.__handle)
    m = simEigen.Matrix(m)
    return m
end

function simEigen.Matrix:count()
    local rows, cols = simEigen.mtxGetSize(self.__handle)
    return rows * cols
end

function simEigen.Matrix:cross(m)
    assert(simEigen.Matrix:ismatrix(m), 'argument must be a Matrix')
    local r = simEigen.mtxCross(self.__handle, m.__handle)
    r = simEigen.Matrix(r)
    return r
end

function simEigen.Matrix:data()
    return simEigen.mtxGetData(self.__handle)
end

function simEigen.Matrix:dot(m)
    assert(simEigen.Matrix:ismatrix(m), 'argument must be a Matrix')
    return simEigen.mtxDot(self.__handle, m.__handle)
end

function simEigen.Matrix:eye(size)
    local data = {}
    for i = 1, size do for j = 1, size do table.insert(data, i == j and 1 or 0) end end
    return simEigen.Matrix(size, size, data)
end

function simEigen.Matrix:iadd(m)
    assert(simEigen.Matrix:ismatrix(m), 'argument must be a Matrix')
    simEigen.mtxIAdd(self.__handle, m.__handle)
    return self
end

function simEigen.Matrix:iaddk(k)
    assert(type(k) == 'number', 'argument must be a number')
    simEigen.mtxIAddK(self.__handle, k)
    return self
end

function simEigen.Matrix:imul(m)
    assert(simEigen.Matrix:ismatrix(m), 'argument must be a Matrix')
    simEigen.mtxIMul(self.__handle, m.__handle)
    return self
end

function simEigen.Matrix:imulk(k)
    assert(type(k) == 'number', 'argument must be a number')
    simEigen.mtxIMulK(self.__handle, k)
    return self
end

function simEigen.Matrix:ismatrix(m)
    return getmetatable(m) == simEigen.Matrix
end

function simEigen.Matrix:isub(m)
    assert(simEigen.Matrix:ismatrix(m), 'argument must be a Matrix')
    simEigen.mtxISub(self.__handle, m.__handle)
    return self
end

function simEigen.Matrix:item(i, j)
    assert(math.type(i) == 'integer')
    assert(math.type(j) == 'integer')
    return simEigen.mtxGetItem(self.__handle, i - 1, j - 1)
end

function simEigen.Matrix:maxCoeff()
    return simEigen.mtxMaxCoeff(self.__handle)
end

function simEigen.Matrix:mean()
    return simEigen.mtxMean(self.__handle)
end

function simEigen.Matrix:minCoeff()
    return simEigen.mtxMinCoeff(self.__handle)
end

function simEigen.Matrix:mul(m)
    assert(simEigen.Matrix:ismatrix(m), 'argument must be a Matrix')
    local r = simEigen.mtxMul(self.__handle, m.__handle)
    r = simEigen.Matrix(r)
    return r
end

function simEigen.Matrix:mulk(k)
    assert(type(k) == 'number', 'argument must be a number')
    local r = simEigen.mtxMulK(self.__handle, k)
    r = simEigen.Matrix(r)
    return r
end

function simEigen.Matrix:norm()
    return simEigen.mtxNorm(self.__handle)
end

function simEigen.Matrix:normalize()
    simEigen.mtxNormalize(self.__handle)
    return self
end

function simEigen.Matrix:normalized()
    local m = simEigen.mtxNormalized(self.__handle)
    m = simEigen.Matrix(m)
    return m
end

function simEigen.Matrix:pinv(b, damping)
    assert(b == nil or simEigen.Matrix:ismatrix(m), 'b must be a Matrix or nil')
    damping = damping or 0.0
    local m, x = simEigen.mtxPInv(self.__handle, (b or {}).__handle)
    m = simEigen.Matrix(m)
    if x then x = simEigen.Matrix(x) end
    return m, x
end

function simEigen.Matrix:prod()
    return simEigen.mtxProd(self.__handle)
end

function simEigen.Matrix:row(i)
    return self:block(i, 1, 1, -1)
end

function simEigen.Matrix:rowdata(i)
    assert(math.type(i) == 'integer')
    return simEigen.mtxGetRow(self.__handle, i - 1)
end

function simEigen.Matrix:rows()
    local rows, cols = simEigen.mtxGetSize(self.__handle)
    return rows
end

function simEigen.Matrix:setcol(j, data)
    assert(math.type(j) == 'integer')
    simEigen.mtxSetCol(self.__handle, j - 1, data)
    return self
end

function simEigen.Matrix:setdata(data)
    assert(type(data) == 'table')
    simEigen.mtxSetData(self.__handle, data)
    return self
end

function simEigen.Matrix:setitem(i, j, data)
    assert(math.type(i) == 'integer')
    assert(math.type(j) == 'integer')
    simEigen.mtxSetItem(self.__handle, i - 1, j - 1, data)
    return self
end

function simEigen.Matrix:setrow(i, data)
    assert(math.type(i) == 'integer')
    simEigen.mtxSetRow(self.__handle, i - 1, data)
    return self
end

function simEigen.Matrix:squaredNorm()
    return simEigen.mtxSquaredNorm(self.__handle)
end

function simEigen.Matrix:sub(m)
    assert(simEigen.Matrix:ismatrix(m), 'argument must be a Matrix')
    local r = simEigen.mtxSub(self.__handle, m.__handle)
    r = simEigen.Matrix(r)
    return r
end

function simEigen.Matrix:sum()
    return simEigen.mtxSum(self.__handle)
end

function simEigen.Matrix:svd(computeThinU, computeThinV, b)
    assert(b == nil or simEigen.Matrix:ismatrix(b), 'b must be a Matrix or nil')
    local s, u, v, x = simEigen.mtxSVD(self.__handle, computeThinU, computeThinV, (b or {}).__handle)
    s = simEigen.Matrix(s)
    u = simEigen.Matrix(u)
    v = simEigen.Matrix(v)
    if x then x = simEigen.Matrix(x) end
    return s, u, v, x
end

function simEigen.Matrix:trace()
    return simEigen.mtxTrace(self.__handle)
end

function simEigen.Matrix:transpose()
    simEigen.mtxTranspose(self.__handle)
    return self
end

function simEigen.Matrix:transposed()
    local m = simEigen.mtxTransposed(self.__handle)
    m = simEigen.Matrix(m)
    return m
end

function simEigen.Matrix:__index(k)
    return simEigen.Matrix[k]
end

function simEigen.Matrix:__tostring()
    local rows, cols = simEigen.mtxGetSize(self.__handle)
    s = 'simEigen.Matrix(' .. rows .. ', ' .. cols .. ', {'
    local data = self:data()
    for i = 0, rows - 1 do
        for j = 0, cols - 1 do
            s = s .. (i == 0 and j == 0 and '' or ', ') .. tostring(data[1 + cols * i + j])
        end
    end
    s = s .. '})'
    return s
end

setmetatable(
    simEigen.Matrix, {
        __call = function(self, rows, cols, data)
            local h = nil
            if type(rows) == 'string' and cols == nil and data == nil then
                -- construct from handle:
                h = rows
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

function simEigen.Vector(v)
    assert(type(v) == 'table')
    return simEigen.Matrix(-1, 1, v)
end

return simEigen
