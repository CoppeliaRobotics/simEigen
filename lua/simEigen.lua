local simEigen = loadPlugin 'simEigen';
(require 'simEigen-typecheck')(simEigen)

simEigen.Matrix = {}

function simEigen.Matrix:abs()
    return self:op(simEigen.op.abs, nil, false)
end

function simEigen.Matrix:acos()
    return self:op(simEigen.op.acos, nil, false)
end

function simEigen.Matrix:add(m)
    return self:op(simEigen.op.add, m, false)
end

function simEigen.Matrix:asin()
    return self:op(simEigen.op.asin, nil, false)
end

function simEigen.Matrix:atan()
    return self:op(simEigen.op.atan, nil, false)
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
    return self
end

function simEigen.Matrix:ceil()
    return self:op(simEigen.op.ceil, nil, false)
end

function simEigen.Matrix:col(j)
    return self:block(1, j, -1, 1)
end

function simEigen.Matrix:coldata(j)
    assert(math.type(j) == 'integer')
    return simEigen.mtxGetColData(self.__handle, j - 1)
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

function simEigen.Matrix:cos()
    return self:op(simEigen.op.cos, nil, false)
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

function simEigen.Matrix:deg()
    return self:op(simEigen.op.deg, nil, false)
end

function simEigen.Matrix:div(m)
    return self:op(simEigen.op.div, m, false)
end

function simEigen.Matrix:dot(m)
    assert(simEigen.Matrix:ismatrix(m), 'argument must be a Matrix')
    return simEigen.mtxDot(self.__handle, m.__handle)
end

function simEigen.Matrix:exp()
    return self:op(simEigen.op.exp, nil, false)
end

function simEigen.Matrix:eye(size)
    if simEigen.Matrix:ismatrix(self) then error('static method') end
    if size == nil and math.type(self) == 'integer' then size = self end
    assert(math.type(size) == 'integer', 'argument must be integer')
    local data = {}
    for i = 1, size do for j = 1, size do table.insert(data, i == j and 1 or 0) end end
    return simEigen.Matrix(size, size, data)
end

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

function simEigen.Matrix:horzcat(...)
    local ms = {...}
    if simEigen.Matrix:ismatrix(self) then table.insert(ms, 1, self) end
    local m = simEigen.mtxHorzCat(map(function(m) return m.__handle end, ms))
    m = simEigen.Matrix(m)
    return m
end

function simEigen.Matrix:iabs()
    return self:op(simEigen.op.abs, nil, true)
end

function simEigen.Matrix:iacos()
    return self:op(simEigen.op.acos, nil, true)
end

function simEigen.Matrix:iadd(m)
    return self:op(simEigen.op.add, m, true)
end

function simEigen.Matrix:iasin()
    return self:op(simEigen.op.asin, nil, true)
end

function simEigen.Matrix:iatan()
    return self:op(simEigen.op.atan, nil, true)
end

function simEigen.Matrix:iceil()
    return self:op(simEigen.op.ceil, nil, true)
end

function simEigen.Matrix:icos()
    return self:op(simEigen.op.cos, nil, true)
end

function simEigen.Matrix:ideg()
    return self:op(simEigen.op.deg, nil, true)
end

function simEigen.Matrix:idiv(m)
    return self:op(simEigen.op.div, m, true)
end

function simEigen.Matrix:iexp()
    return self:op(simEigen.op.exp, nil, true)
end

function simEigen.Matrix:ifloor()
    return self:op(simEigen.op.floor, nil, true)
end

function simEigen.Matrix:iintdiv(m)
    return self:op(simEigen.op.intdiv, m, true)
end

function simEigen.Matrix:ilog()
    return self:op(simEigen.op.log, nil, true)
end

function simEigen.Matrix:ilog2()
    return self:op(simEigen.op.log2, nil, true)
end

function simEigen.Matrix:ilog10()
    return self:op(simEigen.op.log10, nil, true)
end

function simEigen.Matrix:imax(m)
    return self:op(simEigen.op.max, m, true)
end

function simEigen.Matrix:imin(m)
    return self:op(simEigen.op.min, m, true)
end

function simEigen.Matrix:imod(m)
    return self:op(simEigen.op.mod, m, true)
end

function simEigen.Matrix:imul(m)
    if type(m) == 'number' then
        return self:op(simEigen.op.times, m, true)
    end

    assert(simEigen.Matrix:ismatrix(m), 'argument must be a Matrix')
    simEigen.mtxIMul(self.__handle, m.__handle)
    return self
end

function simEigen.Matrix:intdiv(m)
    return self:op(simEigen.op.intdiv, m, false)
end

function simEigen.Matrix:irad()
    return self:op(simEigen.op.rad, nil, true)
end

function simEigen.Matrix:isin()
    return self:op(simEigen.op.sin, nil, true)
end

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

function simEigen.Matrix:isqrt()
    return self:op(simEigen.op.sqrt, nil, true)
end

function simEigen.Matrix:isub(m)
    return self:op(simEigen.op.sub, m, true)
end

function simEigen.Matrix:itan()
    return self:op(simEigen.op.tan, nil, true)
end

function simEigen.Matrix:itimes(m)
    return self:op(simEigen.op.times, m, true)
end

function simEigen.Matrix:item(i, j)
    assert(math.type(i) == 'integer')
    assert(math.type(j) == 'integer')
    return simEigen.mtxGetItem(self.__handle, i - 1, j - 1)
end

function simEigen.Matrix:linspace(low, high, count)
    if math.type(low) == 'integer' and high == nil and count == nil then
        low, high, count = 1, low, low
    end
    assert(math.type(count) == 'integer' and count > 1, 'invalid count')
    local m = simEigen.mtxLinSpaced(count, low, high)
    m = simEigen.Matrix(m)
    return m
end

function simEigen.Matrix:log()
    return self:op(simEigen.op.log, nil, false)
end

function simEigen.Matrix:log2()
    return self:op(simEigen.op.log2, nil, false)
end

function simEigen.Matrix:log10()
    return self:op(simEigen.op.log10, nil, false)
end

function simEigen.Matrix:max(m)
    return self:op(simEigen.op.max, m, false)
end

function simEigen.Matrix:maxcoeff()
    return simEigen.mtxMaxCoeff(self.__handle)
end

function simEigen.Matrix:mean()
    return simEigen.mtxMean(self.__handle)
end

function simEigen.Matrix:min(m)
    return self:op(simEigen.op.min, m, false)
end

function simEigen.Matrix:mincoeff()
    return simEigen.mtxMinCoeff(self.__handle)
end

function simEigen.Matrix:mod(m)
    return self:op(simEigen.op.mod, m, false)
end

function simEigen.Matrix:mul(m)
    if type(m) == 'number' then
        return self:op(simEigen.op.times, m, false)
    end

    assert(simEigen.Matrix:ismatrix(m), 'argument must be a Matrix')
    local r = simEigen.mtxMul(self.__handle, m.__handle)
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

function simEigen.Matrix:pinv(b, damping)
    assert(b == nil or simEigen.Matrix:ismatrix(m), 'b must be a Matrix or nil')
    damping = damping or 0.0
    local m, x = simEigen.mtxPInv(self.__handle, (b or {}).__handle)
    m = simEigen.Matrix(m)
    if x then x = simEigen.Matrix(x) end
    return m, x
end

function simEigen.Matrix:print(numToString)
    print(self:__tostring(true, numToString))
end

function simEigen.Matrix:prod()
    return simEigen.mtxProd(self.__handle)
end

function simEigen.Matrix:rad()
    return self:op(simEigen.op.rad, nil, false)
end

function simEigen.Matrix:row(i)
    return self:block(i, 1, 1, -1)
end

function simEigen.Matrix:rowdata(i)
    assert(math.type(i) == 'integer')
    return simEigen.mtxGetRowData(self.__handle, i - 1)
end

function simEigen.Matrix:rows()
    local rows, cols = simEigen.mtxGetSize(self.__handle)
    return rows
end

function simEigen.Matrix:set(i, j, data)
    sim.addLog(sim.verbosity_warnings | sim.verbosity_once, 'm:set(i, j, val) is deprecated. use m:setitem(i, j, val) instead')
    return self:setitem(i, j, data)
end

function simEigen.Matrix:setcol(j, col)
    return self:blockassign(col, 1, j, -1, 1)
end

function simEigen.Matrix:setcoldata(j, data)
    assert(math.type(j) == 'integer')
    simEigen.mtxSetColData(self.__handle, j - 1, data)
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

function simEigen.Matrix:setrow(i, row)
    return self:blockassign(row, i, 1, 1, -1)
end

function simEigen.Matrix:setrowdata(i, data)
    assert(math.type(i) == 'integer')
    simEigen.mtxSetRowData(self.__handle, i - 1, data)
    return self
end

function simEigen.Matrix:sin()
    return self:op(simEigen.op.sin, nil, false)
end

function simEigen.Matrix:sqrt()
    return self:op(simEigen.op.sqrt, nil, false)
end

function simEigen.Matrix:squarednorm()
    return simEigen.mtxSquaredNorm(self.__handle)
end

function simEigen.Matrix:sub(m)
    return self:op(simEigen.op.sub, m, false)
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

function simEigen.Matrix:t()
    return self:transposed()
end

function simEigen.Matrix:tan()
    return self:op(simEigen.op.tan, nil, false)
end

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

function simEigen.Matrix:vertcat(...)
    local ms = {...}
    if simEigen.Matrix:ismatrix(self) then table.insert(ms, 1, self) end
    local m = simEigen.mtxVertCat(map(function(m) return m.__handle end, ms))
    m = simEigen.Matrix(m)
    return m
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

function simEigen.Matrix:__gc()
    simEigen.mtxDestroy(self.__handle)
end

function simEigen.Matrix:__tostring(forDisplay, numToString)
    local out = ''

    if forDisplay or __tostring_for_display then
        if __tostring_for_display then -- set by _evalExec (base.lua) for rendering in statusbar
            __tostring_for_display.add_nl = true -- signal we need a newline after comma
            if not __tostring_for_display.first_of_line then
                out = out .. '\n\n' -- keep at start of line
            end
        end

        -- grid format:
        numToString = numToString or function(x) return _S.anyToString(x) end
        local s = {}
        local colwi, colwd = {}, {}
        for i = 1, self:rows() do
            s[i] = self:rowdata(i)
            for j = 1, #s[i] do
                local ns = numToString(s[i][j])
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
    else
        -- compact format:
        local rows, cols = simEigen.mtxGetSize(self.__handle)
        out = out .. 'simEigen.Matrix(' .. rows .. ', ' .. cols .. ', {'
        local data = self:data()
        for i = 0, rows - 1 do
            for j = 0, cols - 1 do
                out = out .. (i == 0 and j == 0 and '' or ', ') .. tostring(data[1 + cols * i + j])
            end
        end
        out = out .. '})'
    end
    return out
end

function simEigen.Matrix:__add(m)
    if type(self) == 'number' then self, m = m, self end
    return self:add(m)
end

function simEigen.Matrix:__sub(m)
    if type(self) == 'number' then return m * (-1) + self end
    return self:sub(m)
end

function simEigen.Matrix:__mul(m)
    if type(self) == 'number' then self, m = m, self end
    return self:mul(m)
end

function simEigen.Matrix:__div(k)
    return self:div(k)
end

function simEigen.Matrix:__idiv(k)
    return self:intdiv(k)
end

function simEigen.Matrix:__mod(k)
    return self:mod(k)
end

function simEigen.Matrix:__unm()
    return self:op(simEigen.op.unm, nil, false)
end

function simEigen.Matrix:__pow(m)
    return self:cross(m)
end

function simEigen.Matrix:__concat(m)
    return self:dot(m)
end

function simEigen.Matrix:__eq(m)
    error 'unsupported operation'
end

function simEigen.Matrix:__ipairs()
    error 'unsupported operation'
end

function simEigen.Matrix:__pairs()
    -- for completion, return methods of simEigen.Matrix
    return pairs(simEigen.Matrix)
end

function simEigen.Matrix:__tocbor(sref, stref)
    local _cbor = cbor or require 'org.conman.cbor'
    return _cbor.TYPE.ARRAY(self:totable(), sref, stref)
end

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
