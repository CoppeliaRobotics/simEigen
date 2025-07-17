local simEigen = loadPlugin 'simEigen';
(require 'simEigen-typecheck')(simEigen)

local class = require 'middleclass'

local Pose = class 'simEigen.Pose'

local Matrix = require 'simEigen.Matrix'

-- @fun {lua_only=true} Pose a combination of a rotation and a translation
-- @arg table t the translation vector (Matrix)
-- @arg table q the rotation quaternion (Quaternion)
-- @ret table p the pose (Pose)
function Pose:initialize(t, q)
    if q == nil then
        -- called with only 1 arg: construct from 7D vector or table

        if not Matrix:ismatrix(t) then
            assert(type(t) == 'table', 'invalid type')
            assert(#t == 7, 'invalid table size')
            t = simEigen.Vector(t)
        end

        assert(Matrix:ismatrix(t), 'invalid type')
        assert(t:isvector(7), 'invalid matrix shape')
        t, q = t:block(1,1,3,1), t:block(4,1,-1,1)
    end

    assert(Matrix:isvector(t, 3), 'argument 1 must be a 3D vector')
    if Matrix:ismatrix(q) then
        q = Quaternion(q)
    end
    assert(Quaternion:isquaternion(q), 'argument 2 must be a Quaternion')
    self.t = t
    self.q = q
end

-- @fun {lua_only=true} Pose:copy create a copy of this pose
-- @ret table m a new pose with same data (Pose)
function Pose:copy()
    return Pose(self.t:copy(), self.q:copy())
end

-- @fun {lua_only=true} Pose:data get the data of this pose, in (tx, ty, tz, qx, qy, qz, qw) order
-- @ret table.double data the pose data
function Pose:data()
    return table.add(self.t:data(), self.q:data())
end

-- @fun {lua_only=true} Pose:fromtransform (class method) convert 4x4 transform matrix to new pose
-- @arg table m a 4x4 transform matrix (Matrix)
-- @ret table p a new pose with result (Pose)
function Pose:fromtransform(m)
    assert(self == Pose, 'class method')
    assert(Matrix:ismatrix(m, 4, 4), 'only works on 4x4 matrices')
    local R, t = m:block(1, 1, 3, 3), m:block(1, 4, 3, 1)
    return Pose(t, Quaternion:fromrotation(R))
end

-- @fun {lua_only=true} Pose:inv return a new pose inverse of this
-- @ret table result inverse pose (Pose)
function Pose:inv()
    local invq = self.q:inv()
    return Pose(invq * (-self.t), invq)
end

-- @fun {lua_only=true} Pose:ispose (class method) check wether the argument is a Pose
-- @arg any m
-- @ret bool true if the argument is an instance of Pose
function Pose:ispose(m)
    assert(self == Pose, 'class method')
    return Pose.isInstanceOf(m, Pose)
end

-- @fun {lua_only=true} Pose:mul multiply with another pose/vector, returning new pose/vector
-- @arg table o the other pose (Pose)
-- @ret table p a new pose with result (Pose)
function Pose:mul(o)
    if Matrix:isvector(o, 3) then
        return self.q * o + self.t
    elseif Pose:ispose(o) then
        return Pose(self.q * o.t + self.t, o.q * self.q)
    else
        error 'invalid argument type'
    end
end

-- @fun {lua_only=true} Pose:totransform convert pose to 4x4 transform matrix
-- @ret table m a new 4x4 transform matrix (Matrix)
function Pose:totransform()
    return self.q:torotation():horzcat(self.t):vertcat(Matrix(1, 4, {0, 0, 0, 1}))
end

function Pose:__copy()
    return self:copy()
end

function Pose:__deepcopy(m)
    return self:copy()
end

function Pose:__index(k)
    if math.type(k) == 'integer' then
        error 'not implemented'
    else
        return rawget(self, k)
    end
end

function Pose:__ispose()
    return Pose:ispose(self)
end

function Pose:__len()
    return 4
end

function Pose:__mul(m)
    return self:mul(m)
end

function Pose:__newindex(k, v)
    if math.type(k) == 'integer' then
        error 'not implemented'
    else
        rawset(self, k, v)
    end
end

function Pose:__pairs()
    -- for completion, return methods of Pose
    return pairs(Pose)
end

function Pose:__topose()
    return self:data()
end

function Pose:__tostring()
    local out = ''
    out = out .. 'simEigen.Pose({'
    for i, x in ipairs(self:data()) do out = out .. (i > 1 and ', ' or '') .. tostring(x) end
    out = out ..'})'
    return out
end

function Pose:__unm()
    return self:inv()
end

return Pose
