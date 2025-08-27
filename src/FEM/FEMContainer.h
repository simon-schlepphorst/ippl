// Class FEMContainer
//    This class represents a one dimensional vector which can be used in the 
//    context of FEM to represent a field defined on the DOFs of a mesh.


#ifndef IPPL_FEMCONTAINER_H
#define IPPL_FEMCONTAINER_H

#include "Types/ViewTypes.h"
#include "Field/HaloCells.h"

#include <array>
#include <vector>
#include <memory>
#include <memory>

namespace ippl {

    template <typename T, unsigned Dim>
    class FEMContainer {
    public:

        using Mesh_t      = UniformCartesian<T, Dim>;
        using Layout_t    = FieldLayout<Dim>;
        using Field_t     = Field<T, Dim, Mesh_t, Cell>;

        FEMContainer();
        FEMContainer(std::array<unsigned, Dim+1> DOFNum, Mesh_t& m, Layout_t& l, int nghost = 1);
        FEMContainer(const FEMContainer<T, Dim>& other);

        void initialize(std::array<unsigned, Dim+1> DOFNum, Mesh_t& m, Layout_t& l, int nghost = 1);


        FEMContainer<T, Dim>& operator=(T value);
        FEMContainer<T, Dim>& operator=(const FEMContainer<T,Dim>& other);

        void fillHalo();
        void accumulateHalo();
        void setHalo(T setValue);


    private:

        std::array<unsigned int, 1 << Dim> numDOFs_m; // Number of DOFs for each field
        
        std::array<std::shared_ptr<Field_t>, 1 << Dim> data_m; // Fields with DOFs for each element type in each direction (vertices, edges, faces, etc.)

        std::array<std::shared_ptr<FieldLayout<Dim>>, 1 << Dim> layout_m;

        UniformCartesian<T, Dim>* mesh_m;

    };
}   // namespace ippl


#include "FEMContainer.hpp"

#endif  // IPPL_FEMCONTAINER_H