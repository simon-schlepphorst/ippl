//
////
//// TestCurl.cpp
////   This program tests the Curl operator on a vector field.
////   The problem size can be given by the user (N^3), and a bool (0 or 1)
////   indicates whether the vector field is A=(xyz, xyz, xyz) or a Gaussian
////   field in all three dimensions.
////
//// Usage:
////   srun ./TestCurl N 0 --info 10
////
////
//

#include "Ippl.h"

#include <array>
#include <iostream>
#include <typeinfo>

KOKKOS_INLINE_FUNCTION double gaussian(double x, double y, double z, double sigma = 1.0,
                                       double mu = 0.5) {
    double pi        = std::acos(-1.0);
    double prefactor = (1 / std::sqrt(2 * 2 * 2 * pi * pi * pi)) * (1 / (sigma * sigma * sigma));
    double r2        = (x - mu) * (x - mu) + (y - mu) * (y - mu) + (z - mu) * (z - mu);

    return -prefactor * std::exp(-r2 / (2 * sigma * sigma));
}

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        constexpr unsigned int dim = 3;
        using Mesh_t               = ippl::UniformCartesian<double, dim>;
        using Centering_t          = Mesh_t::DefaultCentering;
        using Layout_t            = ippl::FieldLayout<dim>;

        int pt         = std::atoi(argv[1]);
        bool gauss_fct = std::atoi(argv[2]);
        ippl::Index I(pt);
        ippl::NDIndex<dim> owned(I, I, I);

        // Specifies SERIAL, PARALLEL dims
        std::array<bool, dim> isParallel;
        isParallel.fill(true);

        // all parallel layout, standard domain, normal axis order
        Layout_t layout(MPI_COMM_WORLD, owned, isParallel);

        // domain [0,1]^3
        double dx                      = 1.0 / double(pt);
        ippl::Vector<double, 3> hx     = {dx, dx, dx};
        ippl::Vector<double, 3> origin = {0.0, 0.0, 0.0};
        Mesh_t mesh(owned, hx, origin);

        // Number of DOFs equal to 1 for each element
        std::array<unsigned, dim + 1> DOFNum;
        for(auto& dofnum : DOFNum) {
            dofnum = 1;
        }

        typedef ippl::FEMContainer<double, dim> field_type;

        field_type testfield(DOFNum, mesh, layout);
    }
    ippl::finalize();

    return 0;
}
