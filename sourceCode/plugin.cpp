#include <stdexcept>
#include <vector>
#include <cmath>
#include <simPlusPlus/Plugin.h>
#include <simPlusPlus/Handles.h>
#include "plugin.h"
#include "stubs.h"
#include "config.h"
#include <Eigen/Dense>
#include <Eigen/QR>
#include <unsupported/Eigen/KroneckerProduct>

using namespace std;
using namespace Eigen;

namespace simEigen {
    using Matrix = ::Matrix<double, ::Dynamic, ::Dynamic, ::RowMajor>;
    using Quaternion = ::Quaterniond;
}

class Plugin : public sim::Plugin
{
public:
    void onInit()
    {
        if(!registerScriptStuff())
            throw sim::exception("failed to register script stuff");

        setExtVersion("Eigen");
        setBuildDate(BUILD_DATE);
    }

    void onCleanup()
    {
    }

    void onScriptStateAboutToBeDestroyed(int scriptHandle, long long scriptUid)
    {
        for(auto m : mtxHandles.find(scriptHandle))
        {
            delete mtxHandles.remove(m);
        }
    }

    void mtxBlock(mtxBlock_in *in, mtxBlock_out *out)
    {
        auto m = mtxHandles.get(in->handle);
        if(in->p == -1) in->p = m->rows() - in->i;
        if(in->q == -1) in->q = m->cols() - in->j;
        if(in->p < 1 || in->q < 1)
            throw std::runtime_error("Invalid size");
        if(in->i < 0 || (in->i + in->p) > m->rows() || in->j < 0 || (in->j + in->q) > m->cols())
            throw std::runtime_error("Size or offset out of bounds");
        auto m2 = new simEigen::Matrix(in->p, in->q);
        *m2 = m->block(in->i, in->j, in->p, in->q);
        out->handle = mtxHandles.add(m2, in->_.scriptID);
    }

    void mtxBlockAssign(mtxBlockAssign_in *in, mtxBlockAssign_out *out)
    {
        auto m = mtxHandles.get(in->handle);
        if(in->p == -1) in->p = m->rows() - in->i;
        if(in->q == -1) in->q = m->cols() - in->j;
        if(in->p < 1 || in->q < 1)
            throw std::runtime_error("Invalid size");
        if(in->i < 0 || (in->i + in->p) > m->rows() || in->j < 0 || (in->j + in->q) > m->cols())
            throw std::runtime_error("Size or offset out of bounds");
        auto m2 = mtxHandles.get(in->handle2);
        m->block(in->i, in->j, in->p, in->q) = *m2;
    }

    void mtxCopy(mtxCopy_in *in, mtxCopy_out *out)
    {
        auto m = mtxHandles.get(in->handle);
        auto m2 = new simEigen::Matrix(m->rows(), m->cols());
        *m2 = *m;
        out->handle = mtxHandles.add(m2, in->_.scriptID);
    }

    void mtxCross(mtxCross_in *in, mtxCross_out *out)
    {
        auto m = mtxHandles.get(in->handle);
        auto m2 = mtxHandles.get(in->handle2);
        auto mr = new simEigen::Matrix;
        Vector3d a, b;
        if(m->rows() == 3 && m->cols() == 1 && m2->rows() == 3 && m2->cols() == 1) {
            a = m->col(0);
            b = m2->col(0);
            *mr = a.cross(b);
        } else if(m->rows() == 1 && m->cols() == 3 && m2->rows() == 1 && m2->cols() == 3) {
            a = m->row(0).transpose();
            b = m2->row(0).transpose();
            *mr = a.cross(b).transpose();
        } else {
            throw std::invalid_argument("Both arguments must be 3D column or row vectors");
        }
        out->handle = mtxHandles.add(mr, in->_.scriptID);
    }

    void mtxDestroy(mtxDestroy_in *in, mtxDestroy_out *out)
    {
        auto m = mtxHandles.get(in->handle);
        delete mtxHandles.remove(m);
    }

    void mtxDot(mtxDot_in *in, mtxDot_out *out)
    {
        auto m = mtxHandles.get(in->handle);
        auto m2 = mtxHandles.get(in->handle2);
        VectorXd a, b;
        if(m->cols() == 1 && m2->cols() == 1) {
            a = m->col(0);
            b = m2->col(0);
        } else if(m->rows() == 1 && m2->rows() == 1) {
            a = m->row(0);
            b = m2->row(0);
        } else {
            throw std::invalid_argument("Both arguments must be 3D column or row vectors");
        }
        out->result = a.dot(b);
    }

    void mtxGetColData(mtxGetColData_in *in, mtxGetColData_out *out)
    {
        auto m = mtxHandles.get(in->handle);
        if(in->j < 0 || in->j >= m->cols())
            throw std::runtime_error("Invalid indices");
        out->data.resize(m->rows());
        for(int i = 0; i < m->rows(); ++i)
            out->data[i] = (*m)(i, in->j);
    }

    void mtxGetData(mtxGetData_in *in, mtxGetData_out *out)
    {
        auto m = mtxHandles.get(in->handle);
        out->data.resize(m->size());
        for(int i = 0; i < m->rows(); ++i)
            for(int j = 0; j < m->cols(); ++j)
                out->data[i * m->cols() + j] = (*m)(i, j);
    }

    void mtxGetItem(mtxGetItem_in *in, mtxGetItem_out *out)
    {
        auto m = mtxHandles.get(in->handle);
        if(in->i < 0 || in->i >= m->rows() || in->j < 0 || in->j >= m->cols())
            throw std::runtime_error("Invalid indices");
        out->data = (*m)(in->i, in->j);
    }

    void mtxGetRowData(mtxGetRowData_in *in, mtxGetRowData_out *out)
    {
        auto m = mtxHandles.get(in->handle);
        if(in->i < 0 || in->i >= m->rows())
            throw std::runtime_error("Invalid indices");
        out->data.resize(m->cols());
        for(int j = 0; j < m->cols(); ++j)
            out->data[j] = (*m)(in->i, j);
    }

    void mtxGetSize(mtxGetSize_in *in, mtxGetSize_out *out)
    {
        auto m = mtxHandles.get(in->handle);
        out->rows = m->rows();
        out->cols = m->cols();
    }

    void mtxHorzCat(mtxHorzCat_in *in, mtxHorzCat_out *out)
    {
        if(in->handles.size() < 2)
            throw std::runtime_error("not enough matrices");
        std::vector<simEigen::Matrix*> m {in->handles.size()};
        int rows = 0, cols = 0;
        for(size_t i = 0; i < in->handles.size(); ++i)
        {
            m[i] = mtxHandles.get(in->handles[i]);
            if(i == 0)
                rows = m[i]->rows();
            else if(rows != m[i]->rows())
                throw std::runtime_error("matrices row count mismatch");
            cols += m[i]->cols();
        }
        int j = 0;
        auto mr = new simEigen::Matrix(rows, cols);
        for(auto mi : m)
        {
            mr->block(0, j, mi->rows(), mi->cols()) = *mi;
            j += mi->cols();
        }
        out->handle = mtxHandles.add(mr, in->_.scriptID);
    }

    void mtxIMul(mtxIMul_in *in, mtxIMul_out *out)
    {
        auto m = mtxHandles.get(in->handle);
        auto m2 = mtxHandles.get(in->handle2);
        if(m->cols() != m2->rows())
            throw std::runtime_error("Incompatible matrix dimensions for multiplication");
        *m = (*m) * (*m2);
    }

    void mtxKron(mtxKron_in *in, mtxKron_out *out)
    {
        auto m = mtxHandles.get(in->handle);
        auto m2 = mtxHandles.get(in->handle2);
        auto mr = new simEigen::Matrix(m->rows(), m->cols());
        *mr = Eigen::kroneckerProduct(*m, *m2).eval();
        out->handle = mtxHandles.add(mr, in->_.scriptID);
    }

    void mtxLinSpaced(mtxLinSpaced_in *in, mtxLinSpaced_out *out)
    {
        if(in->count < 1)
            throw std::runtime_error("Invalid count");
        auto m = new simEigen::Matrix(in->count, 1);
        double high = in->high.value_or(in->low + in->count - 1);
        *m = VectorXd::LinSpaced(in->count, in->low, high);
        out->handle = mtxHandles.add(m, in->_.scriptID);
    }

    void mtxMaxCoeff(mtxMaxCoeff_in *in, mtxMaxCoeff_out *out)
    {
        auto m = mtxHandles.get(in->handle);
        out->result = m->maxCoeff();
    }

    void mtxMean(mtxMean_in *in, mtxMean_out *out)
    {
        auto m = mtxHandles.get(in->handle);
        out->result = m->mean();
    }

    void mtxMinCoeff(mtxMinCoeff_in *in, mtxMinCoeff_out *out)
    {
        auto m = mtxHandles.get(in->handle);
        out->result = m->minCoeff();
    }

    void mtxMul(mtxMul_in *in, mtxMul_out *out)
    {
        auto m = mtxHandles.get(in->handle);
        auto m2 = mtxHandles.get(in->handle2);
        if(m->cols() != m2->rows())
            throw std::runtime_error("Incompatible matrix dimensions for multiplication");
        auto mr = new simEigen::Matrix(m->rows(), m2->cols());
        *mr = (*m) * (*m2);
        out->handle = mtxHandles.add(mr, in->_.scriptID);
    }

    void mtxNew(mtxNew_in *in, mtxNew_out *out)
    {
        auto m = new simEigen::Matrix(in->rows, in->cols);
        if(in->initialData.size() > 0)
        {
            if(in->initialData.size() == 1)
                m->setConstant(in->initialData[0]);
            else if(in->initialData.size() == m->rows() * m->cols())
                for(int i = 0; i < m->rows(); ++i)
                    for(int j = 0; j < m->cols(); ++j)
                        (*m)(i, j) = in->initialData[i * m->cols() + j];
            else
                throw std::runtime_error("Size mismatch between data and matrix dimensions");
        }
        out->handle = mtxHandles.add(m, in->_.scriptID);
    }

    void mtxNorm(mtxNorm_in *in, mtxNorm_out *out)
    {
        auto m = mtxHandles.get(in->handle);
        out->result = m->norm();
    }

    void mtxNormalize(mtxNormalize_in *in, mtxNormalize_out *out)
    {
        auto m = mtxHandles.get(in->handle);
        m->normalize();
    }

    void mtxNormalized(mtxNormalized_in *in, mtxNormalized_out *out)
    {
        auto m = mtxHandles.get(in->handle);
        auto mr = new simEigen::Matrix(m->rows(), m->cols());
        *mr = m->normalized();
        out->handle = mtxHandles.add(mr, in->_.scriptID);
    }

    void mtxOp(mtxOp_in *in, mtxOp_out *out)
    {
        auto m = mtxHandles.get(in->handle);
        simEigen::Matrix *mr = in->inplace ? nullptr : new simEigen::Matrix(m->rows(), m->cols());
        if(in->handle2)
        {
            // binary ops:
            auto m2 = mtxHandles.get(*in->handle2);
            if(m->rows() != m2->rows() || m->cols() != m2->cols())
                throw std::runtime_error("Matrix dimensions must be the same");
            switch(in->op)
            {
            case simeigen_op_add:
                if(in->inplace)
                    *m += (*m2);
                else
                    *mr = (*m) + (*m2);
                break;
            case simeigen_op_sub:
                if(in->inplace)
                    *m -= (*m2);
                else
                    *mr = (*m) - (*m2);
                break;
            case simeigen_op_times:
                if(in->inplace)
                    *m = m->array() * m2->array();
                else
                    *mr = m->array() * m2->array();
                break;
            case simeigen_op_div:
                if(in->inplace)
                    *m = m->array() / m2->array();
                else
                    *mr = m->array() / m2->array();
                break;
            case simeigen_op_intdiv:
                if(in->inplace)
                    *m = (m->array() / m2->array()).cast<long>().cast<double>();
                else
                    *mr = (m->array() / m2->array()).cast<long>().cast<double>();
                break;
            case simeigen_op_min:
                if(in->inplace)
                    *m = m->cwiseMin(*m2);
                else
                    *mr = m->cwiseMin(*m2);
                break;
            case simeigen_op_max:
                if(in->inplace)
                    *m = m->cwiseMax(*m2);
                else
                    *mr = m->cwiseMax(*m2);
                break;
            default:
                throw std::runtime_error("invalid operator");
            }
        }
        else
        {
            // unary ops:
            if(in->inplace)
                mr = m; // unary ops have no extra optimization by eigen...
            switch(in->op)
            {
            case simeigen_op_unm:
                *mr = -(*m);
                break;
            case simeigen_op_abs:
                *mr = m->array().abs();
                break;
            case simeigen_op_acos:
                *mr = m->array().acos();
                break;
            case simeigen_op_asin:
                *mr = m->array().asin();
                break;
            case simeigen_op_atan:
                *mr = m->array().atan();
                break;
            case simeigen_op_ceil:
                *mr = m->array().ceil();
                break;
            case simeigen_op_cos:
                *mr = m->array().cos();
                break;
            case simeigen_op_deg:
                *mr = m->array() * 180. / M_PI;
                break;
            case simeigen_op_exp:
                *mr = m->array().exp();
                break;
            case simeigen_op_floor:
                *mr = m->array().floor();
                break;
            case simeigen_op_log:
                *mr = m->array().log();
                break;
            case simeigen_op_log2:
                *mr = m->array().log2();
                break;
            case simeigen_op_log10:
                *mr = m->array().log10();
                break;
            case simeigen_op_rad:
                *mr = m->array() * M_PI / 180.;
                break;
            case simeigen_op_sin:
                *mr = m->array().sin();
                break;
            case simeigen_op_sqrt:
                *mr = m->array().sqrt();
                break;
            case simeigen_op_tan:
                *mr = m->array().tan();
                break;
            default:
                throw std::runtime_error("invalid operator");
            }
        }
        if(!in->inplace)
            out->handle = mtxHandles.add(mr, in->_.scriptID);
    }

    void mtxOpK(mtxOpK_in *in, mtxOpK_out *out)
    {
        auto m = mtxHandles.get(in->handle);
        simEigen::Matrix *mr = in->inplace ? m : new simEigen::Matrix(m->rows(), m->cols());
        {
            // binary ops:
            switch(in->op)
            {
            case simeigen_op_add:
                *mr = m->array() + in->k;
                break;
            case simeigen_op_sub:
                *mr = m->array() - in->k;
                break;
            case simeigen_op_times:
                *mr = m->array() * in->k;
                break;
            case simeigen_op_div:
                *mr = m->array() / in->k;
                break;
            case simeigen_op_intdiv:
                *mr = (m->array() / in->k).cast<long>().cast<double>();
                break;
            case simeigen_op_mod:
                *mr = m->unaryExpr([=] (double x) { return std::fmod(x, in->k); });
                break;
            case simeigen_op_min:
                *mr = m->array().cwiseMin(in->k);
                break;
            case simeigen_op_max:
                *mr = m->array().cwiseMax(in->k);
                break;
            default:
                throw std::runtime_error("invalid operator");
            }
        }
        if(!in->inplace)
            out->handle = mtxHandles.add(mr, in->_.scriptID);
    }

    void mtxPInv(mtxPInv_in *in, mtxPInv_out *out)
    {
        auto m = mtxHandles.get(in->m);

        if(in->damping > 1e-10)
        {
            int n = m->rows();
            auto minv = new simEigen::Matrix;
            *minv = m->transpose() * ((*m) * m->transpose() + in->damping * in->damping * simEigen::Matrix::Identity(n, n)).inverse();
            out->m = mtxHandles.add(minv, in->_.scriptID);

            if(in->b)
            {
                auto b = mtxHandles.get(*in->b);
                auto x = new simEigen::Matrix;
                *x = (*minv) * (*b);
                out->x = mtxHandles.add(x, in->_.scriptID);
            }
        }
        else
        {
            auto d = m->completeOrthogonalDecomposition();
            auto minv = new simEigen::Matrix;
            *minv = d.pseudoInverse();
            out->m = mtxHandles.add(minv, in->_.scriptID);

            if(in->b)
            {
                auto b = mtxHandles.get(*in->b);
                auto x = new simEigen::Matrix;
                *x = d.solve(*b);
                out->x = mtxHandles.add(x, in->_.scriptID);
            }
        }
    }

    void mtxReshaped(mtxReshaped_in *in, mtxReshaped_out *out)
    {
        auto m = mtxHandles.get(in->handle);
        if((m->rows() * m->cols()) != (in->rows * in->cols))
            throw std::runtime_error("incompatible dimensions");
        auto mr = new simEigen::Matrix;
        *mr = Eigen::Map<simEigen::Matrix>(m->data(), in->rows, in->cols);
        out->handle = mtxHandles.add(mr, in->_.scriptID);
    }

    void mtxProd(mtxProd_in *in, mtxProd_out *out)
    {
        auto m = mtxHandles.get(in->handle);
        out->result = m->prod();
    }

    void mtxSetColData(mtxSetColData_in *in, mtxSetColData_out *out)
    {
        auto m = mtxHandles.get(in->handle);
        if(in->j < 0 || in->j >= m->cols())
            throw std::runtime_error("Invalid indices");
        if(in->data.size() != m->rows())
            throw std::runtime_error("Size mismatch between data and matrix dimensions");
        for(int i = 0; i < m->rows(); ++i)
            (*m)(i, in->j) = in->data[i];
    }

    void mtxSetData(mtxSetData_in *in, mtxSetData_out *out)
    {
        auto m = mtxHandles.get(in->handle);
        if(in->data.size() != m->rows() * m->cols())
            throw std::runtime_error("Size mismatch between data and matrix dimensions");
        for(int i = 0; i < m->rows(); ++i)
            for(int j = 0; j < m->cols(); ++j)
                (*m)(i, j) = in->data[i * m->cols() + j];
    }

    void mtxSetItem(mtxSetItem_in *in, mtxSetItem_out *out)
    {
        auto m = mtxHandles.get(in->handle);
        if(in->i < 0 || in->i >= m->rows() || in->j < 0 || in->j >= m->cols())
            throw std::runtime_error("Invalid indices");
        (*m)(in->i, in->j) = in->data;
    }

    void mtxSetRowData(mtxSetRowData_in *in, mtxSetRowData_out *out)
    {
        auto m = mtxHandles.get(in->handle);
        if(in->i < 0 || in->i >= m->rows())
            throw std::runtime_error("Invalid indices");
        if(in->data.size() != m->cols())
            throw std::runtime_error("Size mismatch between data and matrix dimensions");
        for(int j = 0; j < m->cols(); ++j)
            (*m)(in->i, j) = in->data[j];
    }

    void mtxSquaredNorm(mtxSquaredNorm_in *in, mtxSquaredNorm_out *out)
    {
        auto m = mtxHandles.get(in->handle);
        out->result = m->squaredNorm();
    }

    void mtxSum(mtxSum_in *in, mtxSum_out *out)
    {
        auto m = mtxHandles.get(in->handle);
        out->result = m->sum();
    }

    void mtxSVD(mtxSVD_in *in, mtxSVD_out *out)
    {
        auto m = mtxHandles.get(in->m);

        unsigned int computationOptions = 0;
        if(in->computeThinU) computationOptions |= ComputeThinU;
        if(in->computeThinV) computationOptions |= ComputeThinV;
        JacobiSVD<simEigen::Matrix> svd(*m, ComputeThinU | ComputeThinV);

        auto s = new simEigen::Matrix;
        *s = svd.singularValues();
        out->s = mtxHandles.add(s, in->_.scriptID);
        auto u = new simEigen::Matrix;
        *u = svd.matrixU();
        out->u = mtxHandles.add(u, in->_.scriptID);
        auto v = new simEigen::Matrix;
        *v = svd.matrixV();
        out->v = mtxHandles.add(v, in->_.scriptID);

        if(in->b)
        {
            auto b = mtxHandles.get(*in->b);
            auto x = new simEigen::Matrix;
            *x = svd.solve(*b);
            out->x = mtxHandles.add(x, in->_.scriptID);
        }
    }

    void mtxTrace(mtxTrace_in *in, mtxTrace_out *out)
    {
        auto m = mtxHandles.get(in->handle);
        out->result = m->trace();
    }

    void mtxTranspose(mtxTranspose_in *in, mtxTranspose_out *out)
    {
        auto m = mtxHandles.get(in->handle);
        m->transposeInPlace();
    }

    void mtxTransposed(mtxTransposed_in *in, mtxTransposed_out *out)
    {
        auto m = mtxHandles.get(in->handle);
        auto m2 = new simEigen::Matrix;
        *m2 = m->transpose();
        out->handle = mtxHandles.add(m2, in->_.scriptID);
    }

    void mtxVertCat(mtxVertCat_in *in, mtxVertCat_out *out)
    {
        if(in->handles.size() < 2)
            throw std::runtime_error("not enough matrices");
        std::vector<simEigen::Matrix*> m {in->handles.size()};
        int rows = 0, cols = 0;
        for(size_t i = 0; i < in->handles.size(); ++i)
        {
            m[i] = mtxHandles.get(in->handles[i]);
            if(i == 0)
                cols = m[i]->cols();
            else if(cols != m[i]->cols())
                throw std::runtime_error("matrices column count mismatch");
            rows += m[i]->rows();
        }
        int i = 0;
        auto mr = new simEigen::Matrix(rows, cols);
        for(auto mi : m)
        {
            mr->block(i, 0, mi->rows(), mi->cols()) = *mi;
            i += mi->rows();
        }
        out->handle = mtxHandles.add(mr, in->_.scriptID);
    }

    void quatDestroy(quatDestroy_in *in, quatDestroy_out *out)
    {
        auto q = quatHandles.get(in->handle);
        delete quatHandles.remove(q);
    }

    void quatFromAxisAngle(quatFromAxisAngle_in *in, quatFromAxisAngle_out *out)
    {
        auto axis = mtxHandles.get(in->axisHandle);
        if(axis->rows() != 3 || axis->cols() != 1)
            throw std::runtime_error("not a vector 3D");
        auto q = new simEigen::Quaternion(Eigen::AngleAxisd(in->angle, *axis));
        out->handle = quatHandles.add(q, in->_.scriptID);
    }

    void quatFromEuler(quatFromEuler_in *in, quatFromEuler_out *out)
    {
        auto euler = mtxHandles.get(in->handle);
        if(euler->rows() != 3 || euler->cols() != 1)
            throw std::runtime_error("not a vector 3D");
        Eigen::AngleAxisd rollAngle((*euler)(0), Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd pitchAngle((*euler)(1), Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd yawAngle((*euler)(2), Eigen::Vector3d::UnitZ());
        auto q = new simEigen::Quaternion(rollAngle * pitchAngle * yawAngle);
        out->handle = quatHandles.add(q, in->_.scriptID);
    }

    void quatFromRotation(quatFromRotation_in *in, quatFromRotation_out *out)
    {
        auto m = mtxHandles.get(in->handle);
        if(m->rows() != 3 || m->cols() != 3)
            throw std::runtime_error("not a 3x3 matrix");
        Matrix3d R = *m;
        if(std::abs(R.determinant() - 1.0) > 1e-6)
            throw std::runtime_error("not a rotation matrix (det != 1)");
        if(!(R * R.transpose()).isApprox(Matrix3d::Identity(), 1e-6))
            throw std::runtime_error("not a rotation matrix (not orthogonal)");
        auto q = new simEigen::Quaternion(R);
        out->handle = quatHandles.add(q, in->_.scriptID);
    }

    void quatGetData(quatGetData_in *in, quatGetData_out *out)
    {
        auto q = quatHandles.get(in->handle);
        out->data.resize(4);
        out->data[3] = q->w();
        out->data[0] = q->x();
        out->data[1] = q->y();
        out->data[2] = q->z();
    }

    void quatInv(quatInv_in *in, quatInv_out *out)
    {
        auto q = quatHandles.get(in->handle);
        auto qr = new simEigen::Quaternion;
        *qr = q->inverse();
        out->handle = quatHandles.add(qr, in->_.scriptID);
    }

    void quatMulQuat(quatMulQuat_in *in, quatMulQuat_out *out)
    {
        auto q = quatHandles.get(in->handle);
        auto q2 = quatHandles.get(in->handle2);
        simEigen::Quaternion *qr = in->inplace ? q : new simEigen::Quaternion;
        *qr = (*q) * (*q2);
        if(!in->inplace)
            out->handle = quatHandles.add(qr, in->_.scriptID);
    }

    void quatMulVec(quatMulVec_in *in, quatMulVec_out *out)
    {
        auto q = quatHandles.get(in->handle);
        auto v = mtxHandles.get(in->vectorHandle);
        if(v->rows() != 3)
            throw std::runtime_error("invalid size");
        simEigen::Matrix *mr = new simEigen::Matrix(v->rows(), v->cols());
        *mr = q->toRotationMatrix() * (*v);
        out->handle = mtxHandles.add(mr, in->_.scriptID);
    }

    void quatNew(quatNew_in *in, quatNew_out *out)
    {
        const auto &d = in->initialData;
        double qx = d[0], qy = d[1], qz = d[2], qw = d[3];
        auto q = new simEigen::Quaternion(qw, qx, qy, qz);
        out->handle = quatHandles.add(q, in->_.scriptID);
    }

    void quatSetData(quatSetData_in *in, quatSetData_out *out)
    {
        auto q = quatHandles.get(in->handle);
        q->w() = in->data[3];
        q->x() = in->data[0];
        q->y() = in->data[1];
        q->z() = in->data[2];
    }

    void quatSLERP(quatSLERP_in *in, quatSLERP_out *out)
    {
        auto q = quatHandles.get(in->handle);
        auto q2 = quatHandles.get(in->handle2);
        auto qr = new simEigen::Quaternion;
        *qr = q->slerp(in->t, *q2);
        out->handle = quatHandles.add(qr, in->_.scriptID);
    }

    void quatToAxisAngle(quatToAxisAngle_in *in, quatToAxisAngle_out *out)
    {
        auto q = quatHandles.get(in->handle);
        Eigen::AngleAxisd angleAxis(*q);
        out->angle = angleAxis.angle();
        simEigen::Matrix *mr = new simEigen::Matrix(3, 1);
        *mr = angleAxis.axis();
        out->axisHandle = mtxHandles.add(mr, in->_.scriptID);
    }

    void quatToEuler(quatToEuler_in *in, quatToEuler_out *out)
    {
        auto q = quatHandles.get(in->handle);
        simEigen::Matrix *mr = new simEigen::Matrix(3, 1);
        *mr = q->toRotationMatrix().eulerAngles(0, 1, 2);
        out->handle = mtxHandles.add(mr, in->_.scriptID);
    }

    void quatToRotation(quatToRotation_in *in, quatToRotation_out *out)
    {
        auto q = quatHandles.get(in->handle);
        simEigen::Matrix *mr = new simEigen::Matrix(3, 3);
        *mr = q->toRotationMatrix();
        out->handle = mtxHandles.add(mr, in->_.scriptID);
    }

    // OLD FUNCTIONS:

    void toMatrix(const Grid<double> &g, simEigen::Matrix &m)
    {
        if(g.dims.size() != 2)
            throw sim::exception("grid must be a matrix");
        m.resize(g.dims[0], g.dims[1]);
        size_t k = 0;
        for(size_t i = 0; i < g.dims[0]; i++)
            for(size_t j = 0; j < g.dims[1]; j++)
                m(i, j) = g.data[k++];
    }

    void toGrid(const simEigen::Matrix &m, Grid<double> &g)
    {
        g.dims.clear();
        g.dims.push_back(m.rows());
        g.dims.push_back(m.cols());
        g.data.clear();
        size_t k = 0;
        for(size_t i = 0; i < g.dims[0]; i++)
            for(size_t j = 0; j < g.dims[1]; j++)
                g.data.push_back(m(i, j));
    }

    void svd(svd_in *in, svd_out *out)
    {
        simEigen::Matrix m, v;
        toMatrix(in->m, m);

        unsigned int computationOptions = 0;
        if(in->computeThinU) computationOptions |= ComputeThinU;
        if(in->computeThinV) computationOptions |= ComputeThinV;
        JacobiSVD<simEigen::Matrix> svd(m, ComputeThinU | ComputeThinV);

        toGrid(svd.singularValues(), out->s);
        toGrid(svd.matrixU(), out->u);
        toGrid(svd.matrixV(), out->v);

        if(in->b)
        {
            simEigen::Matrix b;
            toMatrix(*in->b, b);
            out->x = Grid<double>{};
            toGrid(svd.solve(b), *out->x);
        }
    }

    void pinv(pinv_in *in, pinv_out *out)
    {
        simEigen::Matrix m;
        toMatrix(in->m, m);

        if(in->damping > 1e-10)
        {
            int n = m.rows();
            simEigen::Matrix minv = m.transpose() * (m * m.transpose() + in->damping * in->damping * simEigen::Matrix::Identity(n, n)).inverse();
            toGrid(minv, out->m);

            if(in->b)
            {
                simEigen::Matrix b;
                toMatrix(*in->b, b);
                out->x = Grid<double>{};
                toGrid(minv * b, *out->x);
            }
        }
        else
        {
            auto d = m.completeOrthogonalDecomposition();
            simEigen::Matrix minv = d.pseudoInverse();
            toGrid(minv, out->m);

            if(in->b)
            {
                simEigen::Matrix b;
                toMatrix(*in->b, b);
                out->x = Grid<double>{};
                toGrid(d.solve(b), *out->x);
            }
        }
    }

private:
    sim::Handles<simEigen::Matrix*> mtxHandles{"simEigen.Matrix"};
    sim::Handles<simEigen::Quaternion*> quatHandles{"simEigen.Quaternion"};
};

SIM_PLUGIN(Plugin)
#include "stubsPlusPlus.cpp"
