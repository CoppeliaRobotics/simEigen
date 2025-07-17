local Matrix = require 'simEigen.Matrix'

local Vector = {}

-- @fun {lua_only=true} Vector construct a new vector (that is: a one-column matrix); can also use the form: simEigen.Vector{1, 2, 3, 4} to construct directly from data, size will be determined automatically
-- @arg int size number of elements (matrix rows)
-- @arg table.float data initialization data (optional; can also be a single value)
-- @ret table v the new vector (Matrix)
function Vector:new(v, fv)
    assert(self == Vector, 'class method')
    if type(v) == 'table' and fv == nil then
        -- construct from vector data:
        return Matrix(-1, 1, v)
    elseif math.type(v) == 'integer' then
        -- construct from size, [fillValue]:
        return Matrix(v, 1, fv)
    else
        error('invalid arguments')
    end
end

-- @fun {lua_only=true} Vector:isvector (class method) check wether the argument is a Nx1 Matrix (i.e. a Vector)
-- @arg any m
-- @ret bool true if the argument is an instance of Matrix of size Nx1
function Vector:isvector(...)
    assert(self == Vector, 'class method')
    return Matrix:isvector(...)
end

return setmetatable(Vector, {
    __call = function(self, ...) return Vector:new(...) end,
    __tostring = function() return 'alias of simEigen.Matrix' end,
})
