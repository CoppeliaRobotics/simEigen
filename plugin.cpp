#include <stdexcept>
#include <vector>
#include <cmath>
#include "simPlusPlus/Plugin.h"
#include "simPlusPlus/Handle.h"
#include "plugin.h"
#include "stubs.h"
#include "config.h"
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

class Plugin : public sim::Plugin
{
public:
    void onStart()
    {
        if(!registerScriptStuff())
            throw sim::exception("failed to register script stuff");

        setExtVersion("Bezier");
        setBuildDate(BUILD_DATE);
    }

    void toMatrix(const Grid<float> &g, MatrixXf &m)
    {
        if(g.dims.size() != 2)
            throw sim::exception("grid must be a matrix");
        size_t k = 0;
        for(size_t i = 0; i < g.dims[0]; i++)
            for(size_t j = 0; j < g.dims[1]; j++)
                m(i, j) = g.data[k++];
    }

    void toGrid(const MatrixXf &m, Grid<float> &g)
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
        MatrixXf m, v;
        toMatrix(in->m, m);

        unsigned int computationOptions = 0;
        if(in->computeThinU) computationOptions |= ComputeThinU;
        if(in->computeThinV) computationOptions |= ComputeThinV;
        JacobiSVD<MatrixXf> svd(m, ComputeThinU | ComputeThinV);

        toGrid(svd.singularValues(), out->s);
        toGrid(svd.matrixU(), out->u);
        toGrid(svd.matrixV(), out->v);

        if(in->v)
        {
            MatrixXf v;
            toMatrix(*in->v, v);
            out->x = {};
            toGrid(svd.solve(v), *out->x);
        }
    }
};

SIM_PLUGIN(PLUGIN_NAME, PLUGIN_VERSION, Plugin)
#include "stubsPlusPlus.cpp"
