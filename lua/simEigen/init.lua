--local simEigen = loadPlugin 'simEigen';
--(require 'simEigen-typecheck')(simEigen)

local simEigen = {}

simEigen.__all = {'Matrix', 'Vector', 'Quaternion', 'Pose'}

for _, n in ipairs(simEigen.__all) do simEigen[n] = require('simEigen.' .. n) end

function simEigen.unittest()
    local Matrix = simEigen.Matrix
    local Vector = simEigen.Vector
    local Quaternion = simEigen.Quaternion
    local Pose = simEigen.Pose

    local function assertApproxEq(a, b, tol)
        tol = tol or 1e-7
        if type(a) == 'number' then
            assert(type(b) == 'number', 'mismatching type')
            assert(math.abs(a - b) < tol)
        elseif Matrix:ismatrix(a) then
            assert(Matrix:ismatrix(b), 'mismatching type')
            assert(a:rows() == b:rows() and a:cols() == b:cols(), 'mismatching matrix size')
            assertApproxEq(a:data(), b:data())
        elseif Quaternion:isquaternion(a) then
            assert(Quaternion:isquaternion(b), 'mismatching type')
            assertApproxEq(a:data(), b:data())
        elseif Pose:ispose(a) then
            assert(Pose:ispose(b), 'mismatching type')
            assertApproxEq(a.t, b.t)
            assertApproxEq(a.q, b.q)
        elseif type(a) == 'table' then
            assert(type(b) == 'table')
            assert(#a == #b, 'mismatching table size')
            for i = 1, #a do assertApproxEq(a[i], b[i]) end
        else
            error('invalid types: ' .. type(a) .. ' and ' .. type(b))
        end
    end

    local m = Matrix(3, 4, {11, 12, 13, 14, 21, 22, 23, 24, 31, 32, 33, 34})

    assertApproxEq(m, Matrix:fromtable{dims = {3, 4}, data = {11, 12, 13, 14, 21, 22, 23, 24, 31, 32, 33, 34}})
    assertApproxEq(m, Matrix:fromtable{{11, 12, 13, 14}, {21, 22, 23, 24}, {31, 32, 33, 34}})
    assert(m:rows() == 3)
    assert(m:cols() == 4)
    assert(m:count() == 12)
    assert(m:totable{}.dims[1] == m:rows())
    assert(m:totable{}.dims[2] == m:cols())
    assertApproxEq(m:totable()[3][2], 32)
    assertApproxEq(m:totable()[2][4], 24)
    for i = 1, 3 do
        assertApproxEq(m:row(i), Matrix(1, 4, {i * 10 + 1, i * 10 + 2, i * 10 + 3, i * 10 + 4}))
    end
    for j = 1, 4 do
        assertApproxEq(m:col(j), Matrix(3, 1, {10 + j, 20 + j, 30 + j}))
    end
    for i = 1, 3 do
        for j = 1, 4 do
            assertApproxEq(m:item(i, j), i * 10 + j)
        end
    end
    assertApproxEq(m:item(2, 3), m[2][3])
    assertApproxEq(m[2][3], m:row(2)[3])
    assertApproxEq(m.T:col(2).T, m:row(2))
    assertApproxEq(m:t():col(2):t(), m:row(2))
    assertApproxEq(m * Matrix(4, 1, {1, 0, 0, 1}), Matrix(3, 1, {25, 45, 65}))
    assertApproxEq(2 * m, 2 * m)
    assertApproxEq(m + m, 2 * m)
    assertApproxEq(m - m, 0 * m)
    assertApproxEq(m * m.T, Matrix(3, 3, {630, 1130, 1630, 1130, 2030, 2930, 1630, 2930, 4230}))
    assertApproxEq(m * m:t(), Matrix(3, 3, {630, 1130, 1630, 1130, 2030, 2930, 1630, 2930, 4230}))
    assertApproxEq(m * m:t() * m * m:t(), Matrix(3, 3, {
        4330700, 7781700, 11232700,
        7781700, 13982700, 20183700,
        11232700, 20183700, 29134700,
    }))
    assertApproxEq(m * m.T * m * m.T, Matrix(3, 3, {
        4330700, 7781700, 11232700,
        7781700, 13982700, 20183700,
        11232700, 20183700, 29134700,
    }))
    assertApproxEq(m.T * m, Matrix(4, 4, {
        1523, 1586, 1649, 1712,
        1586, 1652, 1718, 1784,
        1649, 1718, 1787, 1856,
        1712, 1784, 1856, 1928,
    }))
    assertApproxEq(Vector {2.1, 7, 8.2} // 2, Vector {1, 3, 4})
    assertApproxEq(Matrix:fromtable{{1, 0, 0, 0}}.T:norm(), 1)
    assertApproxEq(Matrix(3, 1, {3, 4, 0}):norm(), 5)
    assertApproxEq(Matrix(3, 1, {3, 4, 0}):dot(Matrix(3, 1, {-4, 3, 5})), 0)
    assertApproxEq(Matrix(3, 1, {3, 4, 0}):data()[1], 3)
    assertApproxEq(Matrix(3, 1, {3, 4, 0}):data()[2], 4)
    assertApproxEq(Matrix(1, 3, {3, 4, 0}):data()[1], 3)
    assertApproxEq(Matrix(1, 3, {3, 4, 0}):data()[2], 4)
    local x, y, z = Matrix(3, 1, {1, 0, 0}), Matrix(3, 1, {0, 1, 0}), Matrix(3, 1, {0, 0, 1})
    --assertApproxNEq(x:dot(y:cross(z)), 0)
    assertApproxEq(y:dot(y:cross(z)), 0)
    assertApproxEq(z:dot(y:cross(z)), 0)
    assertApproxEq(Matrix(2, 2, {2, -2, 4, -4}), -Matrix(2, 2, {-2, 2, -4, 4}))
    local i = Matrix(3, 3)
    i:setcol(1, Matrix(3, 1, {1, 0, 0}))
    i:setcol(2, Matrix(3, 1, {0, 2, 0}))
    i:setcol(3, Matrix(3, 1, {0, 0, 3}))
    assertApproxEq(i, Matrix(3, 3, {1, 0, 0, 0, 2, 0, 0, 0, 3}))
    i:setrow(1, Matrix(1, 3, {0, 1, 1}))
    i:setrow(2, Matrix(1, 3, {2, 0, 2}))
    i:setrow(3, Matrix(1, 3, {3, 3, 0}))
    assertApproxEq(i, Matrix(3, 3, {0, 1, 1, 2, 0, 2, 3, 3, 0}))
    i:setitem(1, 1, 9)
    i:setitem(2, 2, 9)
    i:setitem(3, 3, 9)
    assertApproxEq(i, Matrix(3, 3, {9, 1, 1, 2, 9, 2, 3, 3, 9}))
    local s = Matrix(3, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9})
    local temp = s:row(1)
    s:setrow(1, s:row(2))
    s:setrow(2, temp)
    assertApproxEq(s, Matrix(3, 3, {4, 5, 6, 1, 2, 3, 7, 8, 9}))
    local m1 = Matrix(2, 2, {1, 0, 0, 1})
    local m2 = m1
    m2:setitem(1, 1, 6)
    assertApproxEq(m1:item(1, 1), 6)
    local m3 = m1:copy()
    m3:setitem(1, 1, 9)
    assertApproxEq(m3:item(1, 1), 9)
    assertApproxEq(m1:item(1, 1), 6)
    -- data should be copied, not referenced:
    local d = {100, 200, 300}
    m4 = Matrix(3, 1, d)
    table.remove(d)
    assert(pcall(function() tostring(m4) end))
    m5 = Matrix:fromtable{{1, 20, 5, 3}, {10, 2, 28, 4}, {2, 5, 7, 9}}
    assertApproxEq(m5:sum(), 96)
    assertApproxEq(m5:prod(), 423360000)
    assertApproxEq(m5:mean(), 8)
    rot_e = Quaternion:fromeuler{0.7853982, 0.5235988, 1.5707963}
    rot_m = Matrix(3, 3, {0.0000000, -0.8660254, 0.5000000, 0.7071068, -0.3535534, -0.6123725, 0.7071068, 0.3535534, 0.6123725})
    rot_q = Quaternion{0.4304593, -0.092296, 0.7010574, 0.5609855}
    assertApproxEq(rot_e, Quaternion:fromrotation(rot_m))
    assertApproxEq(rot_e, rot_q)
    assertApproxEq(Quaternion:fromrotation(rot_m), rot_q)
    assertApproxEq(Pose{0, 0, 0, 0, 0, 0, 1}, Pose(Vector{0, 0, 0}, Quaternion{0, 0, 0, 1}))
    assertApproxEq(Quaternion:fromrotation(Matrix(3, 3, {-1, 0, 0, 0, -1, 0, 0, 0, 1})), Quaternion{0, 0, 1, 0})
    assertApproxEq(Matrix(2, 2, {1, 2, 3, 4}):kron(Matrix(2, 2, {0, 5, 6, 7})),
            Matrix(4, 4, {0, 5, 0, 10, 6, 7, 12, 14, 0, 15, 0, 20, 18, 21, 24, 28}))
    assertApproxEq(
        Matrix(2, 3, {1, -4, 7, -2, 3, 3}):kron(
            Matrix(4, 4, {8, -9, -6, 5, 1, -3, -4, 7, 2, 8, -8, -3, 1, 2, -5, -1})),
            Matrix(8, 12, {
                8, -9, -6, 5, -32, 36, 24, -20, 56, -63, -42, 35, 1, -3, -4, 7, -4, 12, 16, -28, 7,
                -21, -28, 49, 2, 8, -8, -3, -8, -32, 32, 12, 14, 56, -56, -21, 1, 2, -5, -1, -4, -8,
                20, 4, 7, 14, -35, -7, -16, 18, 12, -10, 24, -27, -18, 15, 24, -27, -18, 15, -2, 6,
                8, -14, 3, -9, -12, 21, 3, -9, -12, 21, -4, -16, 16, 6, 6, 24, -24, -9, 6, 24, -24,
                -9, -2, -4, 10, 2, 3, 6, -15, -3, 3, 6, -15, -3,
            }
        )
    )
    assertApproxEq(Matrix:linspace(0, 1, 5), Vector {0., 0.25, 0.5, 0.75, 1.})
    i, j = Matrix(2, 2, {1, 1, 2, 2}), Matrix(2, 2, {1, 2, 3, 4})
    assertApproxEq(Matrix:horzcat(Vector {1, 0, 0}, Vector {0, 1, 0}, Vector {0, 0, 1}), Matrix:eye(3))
    assertApproxEq(Matrix:vertcat(Matrix(4, 3, 1), Matrix:eye(3)):col(2), Vector {1, 1, 1, 1, 0, 1, 0})
    assertApproxEq(Vector {-1, 0, 90, -4}:abs(), Vector {1, 0, 90, 4})
    assertApproxEq(Vector {-0.6, 0.3}:acos(), Vector {math.acos(-0.6), math.acos(0.3)})
    assertApproxEq(-24, Matrix(4, 4, {1, 3, 0, 1, 0, 0, 3, 2, 2, 0, 3, 2, 1, 2, 1, 0}):det())
    assertApproxEq(0, Matrix(4, 4, {0, 0, 0, 0, 1, 0, 3, 3, 1, 1, 1, 3, 1, 0, 3, 1}):det())
    assertApproxEq(-22, Matrix(5, 5, {3, 2, 2, 1, 3, 0, 3, 0, 1, 3, 3, 0, 4, 3, 2, 2, 2, 1, 2, 2, 4, 3, 3, 1, 4}):det())
    for i = 1, 1000 do
        local p = Pose(Matrix:random(3, 1), Quaternion:fromeuler(Matrix:random(3, 1)))
        local m = p:totransform()
        local mi = m:pinv()
        assertApproxEq(m * mi, Matrix:eye(4))
        assertApproxEq(mi * m, Matrix:eye(4))
    end
    -- inference of row/col count (only 1 at a time):
    assert(Matrix(3, -1, {1, 2, 3, 4, 5, 6}):cols() == 2)
    assert(Matrix(-1, 2, {1, 2, 3, 4, 5, 6}):rows() == 3)
    function asserterror(f, ...)
        if pcall(f, ...) then error() end
    end
    asserterror(function() Matrix(-1, 3, {1, 2, 3, 4, 5, 6, 7}) end)

    -- tests for bugs
    local p = Pose{0, 0, 0, 0, 0, 0, 1}
    assertApproxEq(p.t, Vector{0, 0, 0})
    assertApproxEq(p.q, Quaternion{0, 0, 0, 1})
    assertApproxEq(Matrix(2, 2, {1, 2, 3, 4}), Matrix{{1, 2}, {3, 4}})
    local v = Vector{8, 3, 0.2}
    assert(Vector:isvector(v))
    assert(Vector:isvector(v, 3))
    assert(v:isvector())
    assert(v:isvector(3))
    assert(not Matrix{{1, 2}, {8, 0.1}}:isvector())

    local pa = Pose(Matrix:random(3, 1), Quaternion:random())
    local pb = Pose(Matrix:random(3, 1), Quaternion:random())
    local pr = pa:inv()*pb
    local ps = Pose:fromtransform(pa:totransform():pinv()*(pb:totransform()))
    assert((pr.t - ps.t):norm() < 1e-5 and ({pr.q:axisangle(ps.q)})[2] < 1e-3)

    print(debug.getinfo(1, 'S').source, 'tests passed')
end

return simEigen
