local Matrix = require 'simEigen.Matrix'

-- @fun {lua_only=true} Vector construct a new vector (that is: a one-column matrix); can also use the form: simEigen.Vector{1, 2, 3, 4} to construct directly from data, size will be determined automatically
-- @arg int size number of elements (matrix rows)
-- @arg table.float data initialization data (optional; can also be a single value)
-- @ret table v the new vector (Matrix)
local function Vector(v, fv)
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

return Vector
