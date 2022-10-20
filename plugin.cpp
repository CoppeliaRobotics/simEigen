#include <stdexcept>
#include <vector>
#include <cmath>
#include "simPlusPlus/Plugin.h"
#include "simPlusPlus/Handle.h"
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
    void onStart()
    {
        if(!registerScriptStuff())
            throw sim::exception("failed to register script stuff");

        setExtVersion("Eigen");
        setBuildDate(BUILD_DATE);
    }

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
};

SIM_PLUGIN(PLUGIN_NAME, PLUGIN_VERSION, Plugin)
#include "stubsPlusPlus.cpp"
