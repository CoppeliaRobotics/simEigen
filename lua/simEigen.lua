local simEigen = loadPlugin 'simEigen';
(require 'simEigen-typecheck')(simEigen)

simEigen.Matrix = {}

function simEigen.Matrix:add(m)
    assert(getmetatable(m) == simEigen.Matrix)
    simEigen.mtxAdd(self.__handle, m.__handle)
    return self
end

function simEigen.Matrix:addk(k)
    simEigen.mtxAddK(self.__handle, k)
    return self
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
    return simEigen.Matrix(m)
end

function simEigen.Matrix:blockassign(m, i, j, p, q)
    assert(getmetatable(m) == simEigen.Matrix)
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

function simEigen.Matrix:cols()
    local rows, cols = simEigen.mtxGetSize(self.__handle)
    return cols
end

function simEigen.Matrix:copy()
    return simEigen.Matrix(simEigen.mtxCopy(self.__handle))
end

function simEigen.Matrix:count()
    local rows, cols = simEigen.mtxGetSize(self.__handle)
    return rows * cols
end

function simEigen.Matrix:data()
    return simEigen.mtxGetData(self.__handle)
end

function simEigen.Matrix:eye(size)
    local data = {}
    for i = 1, size do for j = 1, size do table.insert(data, i == j and 1 or 0) end end
    return simEigen.Matrix(size, size, data)
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
    assert(getmetatable(m) == simEigen.Matrix)
    simEigen.mtxMul(self.__handle, m.__handle)
    return self
end

function simEigen.Matrix:mulk(k)
    simEigen.mtxMulK(self.__handle, k)
    return self
end

function simEigen.Matrix:norm()
    return simEigen.mtxNorm(self.__handle)
end

function simEigen.Matrix:pinv(b, damping)
    assert(b == nil or getmetatable(b) == simEigen.Matrix)
    damping = damping or 0.0
    local m, x = simEigen.mtxPInv(self.__handle, (b or {}).__handle)
    m = simEigen.Matrix(m)
    if x then x = simEigen.Matrix(x) end
end

function simEigen.Matrix:prod()
    return simEigen.mtxProd(self.__handle)
end

function simEigen.Matrix:rows()
    local rows, cols = simEigen.mtxGetSize(self.__handle)
    return rows
end

function simEigen.Matrix:setdata(data)
    assert(type(data) == 'table')
    simEigen.mtxSetData(self.__handle, data)
end

function simEigen.Matrix:squaredNorm()
    return simEigen.mtxSquaredNorm(self.__handle)
end

function simEigen.Matrix:sub(m)
    assert(getmetatable(m) == simEigen.Matrix)
    simEigen.mtxSub(self.__handle, m.__handle)
    return self
end

function simEigen.Matrix:subk(k)
    simEigen.mtxSubK(self.__handle, k)
    return self
end

function simEigen.Matrix:sum()
    return simEigen.mtxSum(self.__handle)
end

function simEigen.Matrix:svd(computeThinU, computeThinV, b)
    assert(b == nil or getmetatable(b) == simEigen.Matrix)
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
