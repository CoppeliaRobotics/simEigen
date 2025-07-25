local simEigen = loadPlugin 'simEigen';

local class = require 'middleclass'

local Quaternion = class 'simEigen.Quaternion'

local Matrix = require 'simEigen.Matrix'
local Vector = require 'simEigen.Vector'

-- @fun {lua_only=true} Quaternion construct a new quaternion
-- @arg {type='table',item_type='float',default={0,0,0,1}} data initialization data, in (qx, qy, qz, qw) order
-- @ret table q the new quaternion (Quaternion)
function Quaternion:initialize(data)
    -- construct from handle:
    if type(data) == 'string' then
        self.__handle = data
        return
    end

    -- construct from Matrix
    if Matrix:ismatrix(data) then
        assert(data:isvector(4), 'invalid matrix shape')
        data = data:data()
    end

    assert((type(data) == 'table' and #data == 4) or data == nil, 'invalid data')

    if data == nil then
        self.__handle = simEigen.quatNew()
    else
        self.__handle = simEigen.quatNew(data)
    end
end

-- @fun {lua_only=true} Quaternion:axisangle compute the axis/angle to rotate from first quaternion to second
-- @ret table axis a new vector 3D with rotation axis (Matrix)
-- @ret double angle the rotation angle
function Quaternion:axisangle(q2)
    local axis, angle = self:inv():mul(q2):toaxisangle()
    axis = self:mul(axis)
    return axis, angle
end

-- @fun {lua_only=true} Quaternion:copy create a copy of this quaternion
-- @ret table m a new quaternion with same data (Quaternion)
function Quaternion:copy()
    local m = simEigen.quatCopy(self.__handle)
    m = Quaternion(m)
    return m
end

-- @fun {lua_only=true} Quaternion:data get the data of this quaternion, in (qx, qy, qz, qw) order
-- @ret table.double data the quaternion data
function Quaternion:data()
    return simEigen.quatGetData(self.__handle)
end

-- @fun {lua_only=true} Quaternion:fromaxisangle (class method) create a new quaternion from axis/angle
-- @arg table axis the rotation axis vector 3D (Matrix)
-- @arg double angle the rotation angle
-- @ret table q the quaternion (Quaternion)
function Quaternion:fromaxisangle(axis, angle)
    assert(self == Quaternion, 'class method')
    axis = Vector:tovector(axis, 3)
    assert(type(angle) == 'number', 'angle must be a number')
    local q = simEigen.quatFromAxisAngle(axis.__handle, angle)
    q = Quaternion(q)
    return q
end

-- @fun {lua_only=true} Quaternion:fromeuler (class method) create a new quaternion from euler angles
-- @arg table euler the Euler angles as 3D vector (Matrix)
-- @ret table q the quaternion (Quaternion)
function Quaternion:fromeuler(euler)
    assert(self == Quaternion, 'class method')
    euler = Vector:tovector(euler, 3)
    local q = simEigen.quatFromEuler(euler.__handle)
    q = Quaternion(q)
    return q
end

-- @fun {lua_only=true} Quaternion:fromrotation (class method) create a new quaternion from rotation matrix
-- @arg table r the rotation matrix (Matrix)
-- @ret table q the quaternion (Quaternion)
function Quaternion:fromrotation(r)
    assert(self == Quaternion, 'class method')
    r = Matrix:tomatrix(r, 3, 3)
    local q = simEigen.quatFromRotation(r.__handle)
    q = Quaternion(q)
    return q
end

-- @fun {lua_only=true} Quaternion:fromtwovectors (class method) create a new quaternion from rotation matrix
-- @arg table m the first vector (Matrix)
-- @arg table m2 the second vector (Matrix)
-- @ret table q the quaternion (Quaternion)
function Quaternion:fromtwovectors(v1, v2)
    assert(self == Quaternion, 'class method')
    v1 = Vector:tovector(v1, 3)
    v2 = Vector:tovector(v2, 3)
    local q = simEigen.quatFromTwoVectors(v1.__handle, v2.__handle)
    q = Quaternion(q)
    return q
end

-- @fun {lua_only=true} Quaternion:fromypr (class method) create a new quaternion from YPR angles
-- @arg table ypr the YPR angles as 3D vector (Matrix)
-- @ret table q the quaternion (Quaternion)
function Quaternion:fromypr(ypr)
    assert(self == Quaternion, 'class method')
    ypr = Vector:tovector(ypr, 3)
    local q = simEigen.quatFromYPR(ypr.__handle)
    q = Quaternion(q)
    return q
end

-- @fun {lua_only=true} Quaternion:imul multiply with another quaternion, in place
-- @arg table o the other quaternion (Quaternion)
-- @ret table self this quaternion (Quaternion)
function Quaternion:imul(o)
    if Quaternion:isquaternion(o) then
        simEigen.quatMulQuat(self.__handle, o.__handle, true)
        return self
    else
        error 'invalid type'
    end
end

-- @fun {lua_only=true} Quaternion:inv return a new quaternion inverse of this
-- @ret table result inverted quaternion (Quaternion)
function Quaternion:inv()
    local q = simEigen.quatInv(self.__handle)
    q = Quaternion(q)
    return q
end

-- @fun {lua_only=true} Quaternion:isquaternion (class method) check wether the argument is a Quaternion
-- @arg any m
-- @ret bool true if the argument is an instance of Quaternion
function Quaternion:isquaternion(m)
    assert(self == Quaternion, 'class method')
    return Quaternion.isInstanceOf(m, Quaternion)
end

-- @fun {lua_only=true} Quaternion:mul multiply with another quaternion/vector, returning new quaternion/vector
-- @arg table o the other quaternion (Quaternion)
-- @ret table q a new quaternion with result (Quaternion)
function Quaternion:mul(o)
    if Quaternion:isquaternion(o) then
        local q = simEigen.quatMulQuat(self.__handle, o.__handle, false)
        q = Quaternion(q)
        return q
    else
        o = Vector:tovector(o, 3)
        local v = simEigen.quatMulVec(self.__handle, o.__handle)
        v = Matrix(v)
        return v
    end
end

-- @fun {lua_only=true} Quaternion:random (class method) create a new random quaternion with uniform distribution
-- @ret table q the quaternion (Quaternion)
function Quaternion:random()
    assert(self == Quaternion, 'class method')
    local q = simEigen.quatRandom()
    q = Quaternion(q)
    return q
end

-- @fun {lua_only=true} Quaternion:slerp interpolate quaternions
-- @arg double t interpolation factor 0..1
-- @arg table q2 the other quaternion (Quaternion)
-- @ret table q a new quaternion with result (Quaternion)
function Quaternion:slerp(t, q2)
    assert(type(t) == 'number', 't must be a number')
    assert(Quaternion:isquaternion(q2), 'not a quaternion')
    local q = simEigen.quatSLERP(self.__handle, q2.__handle, t)
    q = Quaternion(q)
    return q
end

-- @fun {lua_only=true} Quaternion:toaxisangle convert this quaternion to a axis/angle representation
-- @ret table axis a new vector 3D with rotation axis (Matrix)
-- @ret double angle the rotation angle
function Quaternion:toaxisangle()
    local axis, angle = simEigen.quatToAxisAngle(self.__handle)
    axis = Matrix(axis)
    return axis, angle
end

-- @fun {lua_only=true} Quaternion:toeuler convert this quaternion to a Euler angles representation
-- @ret table euler a new vector 3D with euler angles (Matrix)
function Quaternion:toeuler()
    local euler = simEigen.quatToEuler(self.__handle)
    euler = Matrix(euler)
    return euler
end

function Quaternion:toquaternion(v)
    assert(self == Quaternion, 'class method')
    if Quaternion:isquaternion(v) then return v end
    if type(v) == 'table' then
        assert(#v == 4, 'invalid length (must have 4 values)')
        return Quaternion(v)
    end
    error 'invalid data'
end

-- @fun {lua_only=true} Quaternion:torotation convert this quaternion to a rotation matrix
-- @ret table q a new matrix with result (Matrix)
function Quaternion:torotation()
    local r = simEigen.quatToRotation(self.__handle)
    r = Matrix(r)
    return r
end

function Quaternion:tovector()
    return Vector(self:data())
end

-- @fun {lua_only=true} Quaternion:toypr convert this quaternion to a YPR angles representation
-- @ret table ypr a new vector 3D with YPR angles (Matrix)
function Quaternion:toypr()
    local ypr = simEigen.quatToYPR(self.__handle)
    ypr = Matrix(ypr)
    return ypr
end

function Quaternion:__copy()
    return self:copy()
end

function Quaternion:__deepcopy(m)
    return self:copy()
end

function Quaternion:__eq(m)
    if Quaternion:isquaternion(m) then
        return self.__handle == m.__handle
    else
        return false
    end
end

function Quaternion:__gc()
    simEigen.quatDestroy(self.__handle)
end

function Quaternion:__index(k)
    if math.type(k) == 'integer' then
        error 'not implemented'
    else
        return rawget(self, k)
    end
end

function Quaternion:__isquaternion()
    return Quaternion:isquaternion(self)
end

function Quaternion:__len()
    return 4
end

function Quaternion:__mul(m)
    return self:mul(m)
end

function Quaternion:__newindex(k, v)
    if math.type(k) == 'integer' then
        local data = simEigen.quatGetData(self.__handle)
        data[k] = v
        simEigen.quatSetData(self.__handle, data)
    else
        rawset(self, k, v)
    end
end

function Quaternion:__pairs()
    -- for completion, return methods of Quaternion
    return pairs(Quaternion)
end

function Quaternion:__tocbor(sref, stref)
    local _cbor = cbor or require 'simCBOR'
    return _cbor.TYPE.ARRAY(self:totable(), sref, stref)
end

function Quaternion:__toquaternion()
    return self:data()
end

function Quaternion:__tostring()
    local out = ''
    out = out .. (Quaternion == _G.Quaternion and '' or 'simEigen.') .. 'Quaternion'
    out = out .. '({'
    for i, x in ipairs(self:data()) do out = out .. (i > 1 and ', ' or '') .. tostring(x) end
    out = out ..'})'
    return out
end

function Quaternion:__unm()
    return self:inv()
end

return Quaternion
