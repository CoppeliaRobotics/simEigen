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

using namespace std;
using namespace Eigen;

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

    void mtxAdd(mtxAdd_in *in, mtxAdd_out *out)
    {
        auto m = mtxHandles.get(in->handle);
        auto m2 = mtxHandles.get(in->handle2);
        if(m->rows() != m2->rows() || m->cols() != m2->cols())
            throw std::runtime_error("Incompatible matrix dimensions for addition");
        *m += *m2;
    }

    void mtxAddK(mtxAddK_in *in, mtxAddK_out *out)
    {
        auto m = mtxHandles.get(in->handle);
        *m = m->array() + in->k;
    }

    void mtxBlock(mtxBlock_in *in, mtxBlock_out *out)
    {
        auto m = mtxHandles.get(in->handle);
        if(in->p == -1) in->p = m->rows() - in->i;
        if(in->q == -1) in->q = m->cols() - in->j;
        if(in->p < 1 || in->q < 1)
            throw std::runtime_error("Invalid size");
        auto m2 = new MatrixXd(in->p, in->q);
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
        auto m2 = mtxHandles.get(in->handle2);
        m->block(in->i, in->j, in->p, in->q) = *m2;
    }

    void mtxCopy(mtxCopy_in *in, mtxCopy_out *out)
    {
        auto m = mtxHandles.get(in->handle);
        auto m2 = new MatrixXd(m->rows(), m->cols());
        *m2 = *m;
        out->handle = mtxHandles.add(m2, in->_.scriptID);
    }

    void mtxDestroy(mtxDestroy_in *in, mtxDestroy_out *out)
    {
        auto m = mtxHandles.get(in->handle);
        delete mtxHandles.remove(m);
    }

    void mtxGetData(mtxGetData_in *in, mtxGetData_out *out)
    {
        auto m = mtxHandles.get(in->handle);
        out->data.resize(m->size());
        for(int i = 0; i < m->rows(); ++i)
            for(int j = 0; j < m->cols(); ++j)
                out->data[i * m->cols() + j] = (*m)(i, j);
    }

    void mtxGetSize(mtxGetSize_in *in, mtxGetSize_out *out)
    {
        auto m = mtxHandles.get(in->handle);
        out->rows = m->rows();
        out->cols = m->cols();
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
        *m = (*m) * (*m2);
    }

    void mtxMulK(mtxMulK_in *in, mtxMulK_out *out)
    {
        auto m = mtxHandles.get(in->handle);
        *m = (*m) * in->k;
    }

    void mtxNew(mtxNew_in *in, mtxNew_out *out)
    {
        auto m = new MatrixXd(in->rows, in->cols);
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

    void mtxPInv(mtxPInv_in *in, mtxPInv_out *out)
    {
        auto m = mtxHandles.get(in->m);

        if(in->damping > 1e-10)
        {
            int n = m->rows();
            auto minv = new Eigen::MatrixXd;
            *minv = m->transpose() * ((*m) * m->transpose() + in->damping * in->damping * MatrixXd::Identity(n, n)).inverse();
            out->m = mtxHandles.add(minv, in->_.scriptID);

            if(in->b)
            {
                auto b = mtxHandles.get(*in->b);
                auto x = new MatrixXd;
                *x = (*minv) * (*b);
                out->x = mtxHandles.add(x, in->_.scriptID);
            }
        }
        else
        {
            auto d = m->completeOrthogonalDecomposition();
            auto minv = new Eigen::MatrixXd;
            *minv = d.pseudoInverse();
            out->m = mtxHandles.add(minv, in->_.scriptID);

            if(in->b)
            {
                auto b = mtxHandles.get(*in->b);
                auto x = new MatrixXd;
                *x = d.solve(*b);
                out->x = mtxHandles.add(x, in->_.scriptID);
            }
        }
    }

    void mtxProd(mtxProd_in *in, mtxProd_out *out)
    {
        auto m = mtxHandles.get(in->handle);
        out->result = m->prod();
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

    void mtxSquaredNorm(mtxSquaredNorm_in *in, mtxSquaredNorm_out *out)
    {
        auto m = mtxHandles.get(in->handle);
        out->result = m->squaredNorm();
    }

    void mtxSub(mtxSub_in *in, mtxSub_out *out)
    {
        auto m = mtxHandles.get(in->handle);
        auto m2 = mtxHandles.get(in->handle2);
        if(m->rows() != m2->rows() || m->cols() != m2->cols())
            throw std::runtime_error("Incompatible matrix dimensions for addition");
        *m -= *m2;
    }

    void mtxSubK(mtxSubK_in *in, mtxSubK_out *out)
    {
        auto m = mtxHandles.get(in->handle);
        *m = m->array() - in->k;
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
        JacobiSVD<MatrixXd> svd(*m, ComputeThinU | ComputeThinV);

        auto s = new MatrixXd;
        *s = svd.singularValues();
        out->s = mtxHandles.add(s, in->_.scriptID);
        auto u = new MatrixXd;
        *u = svd.matrixU();
        out->u = mtxHandles.add(u, in->_.scriptID);
        auto v = new MatrixXd;
        *v = svd.matrixV();
        out->v = mtxHandles.add(v, in->_.scriptID);

        if(in->b)
        {
            auto b = mtxHandles.get(*in->b);
            auto x = new MatrixXd;
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
        auto m2 = new MatrixXd;
        *m2 = m->transpose();
        out->handle = mtxHandles.add(m2, in->_.scriptID);
    }

    // OLD FUNCTIONS:

    void toMatrix(const Grid<double> &g, MatrixXd &m)
    {
        if(g.dims.size() != 2)
            throw sim::exception("grid must be a matrix");
        m.resize(g.dims[0], g.dims[1]);
        size_t k = 0;
        for(size_t i = 0; i < g.dims[0]; i++)
            for(size_t j = 0; j < g.dims[1]; j++)
                m(i, j) = g.data[k++];
    }

    void toGrid(const MatrixXd &m, Grid<double> &g)
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
        MatrixXd m, v;
        toMatrix(in->m, m);

        unsigned int computationOptions = 0;
        if(in->computeThinU) computationOptions |= ComputeThinU;
        if(in->computeThinV) computationOptions |= ComputeThinV;
        JacobiSVD<MatrixXd> svd(m, ComputeThinU | ComputeThinV);

        toGrid(svd.singularValues(), out->s);
        toGrid(svd.matrixU(), out->u);
        toGrid(svd.matrixV(), out->v);

        if(in->b)
        {
            MatrixXd b;
            toMatrix(*in->b, b);
            out->x = Grid<double>{};
            toGrid(svd.solve(b), *out->x);
        }
    }

    void pinv(pinv_in *in, pinv_out *out)
    {
        MatrixXd m;
        toMatrix(in->m, m);

        if(in->damping > 1e-10)
        {
            int n = m.rows();
            Eigen::MatrixXd minv = m.transpose() * (m * m.transpose() + in->damping * in->damping * MatrixXd::Identity(n, n)).inverse();
            toGrid(minv, out->m);

            if(in->b)
            {
                MatrixXd b;
                toMatrix(*in->b, b);
                out->x = Grid<double>{};
                toGrid(minv * b, *out->x);
            }
        }
        else
        {
            auto d = m.completeOrthogonalDecomposition();
            Eigen::MatrixXd minv = d.pseudoInverse();
            toGrid(minv, out->m);

            if(in->b)
            {
                MatrixXd b;
                toMatrix(*in->b, b);
                out->x = Grid<double>{};
                toGrid(d.solve(b), *out->x);
            }
        }
    }

private:
    sim::Handles<MatrixXd*> mtxHandles{"simEigen.Matrix"};
};

SIM_PLUGIN(Plugin)
#include "stubsPlusPlus.cpp"
