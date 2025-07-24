local simEigen = loadPlugin 'simEigen';

local class = require 'middleclass'

local Pose = class 'simEigen.Pose'

local Matrix = require 'simEigen.Matrix'
local Vector = require 'simEigen.Vector'
local Quaternion = require 'simEigen.Quaternion'

-- @fun {lua_only=true} Pose a combination of a rotation and a translation
-- @arg table t the translation vector (Matrix)
-- @arg table q the rotation quaternion (Quaternion)
-- @ret table p the pose (Pose)
function Pose:initialize(t, q)
    if q == nil then
        -- called with only 1 arg: construct from 7D vector or table
        t = Vector:tovector(t, 7)
        t, q = t:block(1,1,3,1), t:block(4,1,-1,1)
    end
    self.t = Vector:tovector(t, 3)
    self.q = Quaternion:toquaternion(q)
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
    m = Matrix:tomatrix(m, 4, 4)
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
    if Vector:isvector(o, 3) or #o == 3 then
        o = Vector:tovector(o, 3)
        return self.q * o + self.t
    elseif Pose:ispose(o) or #o == 7 then
        o = Pose:topose(o)
        return Pose(self.q * o.t + self.t, o.q * self.q)
    else
        error 'invalid argument type'
    end
end

-- @fun {lua_only=true} Pose:interp interpolate poses
-- @arg double t interpolation factor 0..1
-- @arg table p the other pose (Pose)
-- @ret table q a new quaternion with result (Quaternion)
function Pose:interp(t, p)
    t = t or 0.5
    return Pose(0.5 * self.t + 0.5 * p.t, self.q:slerp(t, p.q))
end

function Pose:topose(v)
    assert(self == Pose, 'class method')
    if Pose:ispose(v) then return v end
    if type(v) == 'table' then return Pose(v) end
    error 'invalid data'
end

-- @fun {lua_only=true} Pose:totransform convert pose to 4x4 transform matrix
-- @ret table m a new 4x4 transform matrix (Matrix)
function Pose:totransform()
    return self.q:torotation():horzcat(self.t):vertcat(Matrix(1, 4, {0, 0, 0, 1}))
end

function Pose:tovector()
    return Vector(self:data())
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
    out = out .. (Pose == _G.Pose and '' or 'simEigen.') .. 'Pose'
    out = out .. '({'
    for i, x in ipairs(self:data()) do out = out .. (i > 1 and ', ' or '') .. tostring(x) end
    out = out ..'})'
    return out
end

function Pose:__unm()
    return self:inv()
end

return Pose
