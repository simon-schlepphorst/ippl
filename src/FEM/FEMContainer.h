// Class FEMContainer
// This class holds a collection of DOFs (degrees of freedom) for a finite element mesh.
// The DOFs are stored on multiple ippl::Fields, split by different entity types (vertices, edges in x/y/z, faces in xy/xz/yz, etc.).
// This allows for easy boundary condition application and field operations.

#ifndef IPPL_FEMCONTAINER_H
#define IPPL_FEMCONTAINER_H

#include "Types/ViewTypes.h"
#include "Field/HaloCells.h"
#include "FEM/Entity.h"
#include "DOFArray.h"
#include "FEMHelperStructs.h"

#include <vector>
#include <memory>
#include <memory>
#include <tuple>
#include <type_traits>

namespace ippl {       
    
    template <typename T, unsigned Dim, typename EntityTypes, typename DOFNums>
    class FEMContainer {
        public:
        
        // check that EntityTypes and DOFNums are tuples
        static_assert(is_tuple_v<EntityTypes>, "EntityTypes must be a std::tuple");
        static_assert(is_tuple_v<DOFNums>, "DOFNums must be a std::tuple");
        // Check, that EntityTypes and DOFNums have same size
        static_assert(std::tuple_size_v<EntityTypes> == std::tuple_size_v<DOFNums>, "Number of EntityTypes must match number of DOFNums");
        // Check that all EntityTypes are unique
        static_assert(!tuple_has_duplicates_v<EntityTypes>, "EntityTypes must be unique types");
        // Check that all EntityTypes are derived from Entity
        // Check that all EntityTypes are derived from Entity with matching dimension
        static_assert([]{
            bool all_derived = true;
            std::apply([&all_derived](auto... entity_types) {
                ((all_derived = all_derived && std::is_base_of_v<ippl::Entity<decltype(entity_types), Dim>, decltype(entity_types)>), ...);
            }, EntityTypes{});
            return all_derived;
        }(), "All EntityTypes must be derived from Entity");
        
        static constexpr unsigned dim = Dim;
        using value_type = T;
        
        using Mesh_t      = UniformCartesian<T, Dim>;
        using Layout_t    = FieldLayout<Dim>;
        
        // Build tuple of field types
        using FieldTuple = typename FieldTupleBuilder<T, Dim, Mesh_t, EntityTypes, DOFNums>::type;
        using ViewTuple = typename FieldTupleBuilder<T, Dim, Mesh_t, EntityTypes, DOFNums>::view_type;

        static constexpr unsigned NEntitys = std::tuple_size_v<EntityTypes>; // Number of entity types with DOFs

        FEMContainer();
        FEMContainer(Mesh_t& m, const Layout_t& l, int nghost = 1);
        FEMContainer(const FEMContainer<T, Dim, EntityTypes, DOFNums>& other);

        void initialize(Mesh_t& m, const Layout_t& l, int nghost = 1);


        FEMContainer<T, Dim, EntityTypes, DOFNums>& operator=(T value);
        FEMContainer<T, Dim, EntityTypes, DOFNums>& operator=(const FEMContainer<T,Dim, EntityTypes, DOFNums>& other);

        FEMContainer<T, Dim, EntityTypes, DOFNums>& operator+=(const FEMContainer<T, Dim, EntityTypes, DOFNums>& other);
        FEMContainer<T, Dim, EntityTypes, DOFNums>& operator+=(T value);

        void fillHalo();
        void accumulateHalo();

        int getNGhost() const { return nghost_m; }

        unsigned int getNumFields() { return NEntitys; }

        KOKKOS_INLINE_FUNCTION Mesh_t& get_mesh() const { return *mesh_m; }

        KOKKOS_INLINE_FUNCTION const Layout_t& getVertexLayout() const { return *VertexLayout_m; }
        KOKKOS_INLINE_FUNCTION const Layout_t& getNumEntityLayout() const { return *EntityLayout_m; }

        // Access individual layouts
        template <typename EntityType>
        Layout_t& getLayout() { 
            constexpr unsigned index = TagIndex<EntityTypes>::template index<EntityType>();
            return std::get<index>(layout_m); 
        }

        // Const access individual layouts
        template <typename EntityType>
        const Layout_t& getLayout() const { 
            constexpr unsigned index = TagIndex<EntityTypes>::template index<EntityType>();
            return std::get<index>(layout_m); 
        }
        
        // Get the number of DOFs for each entity type
        template <typename EntityType>
        unsigned getNumDOFs() const {
            constexpr unsigned index = TagIndex<EntityTypes>::template index<EntityType>();
            return numDOFs_m[index];
        }

        template <typename EntityType>
        bool hasView() const {
            // check if EntityType is in EntityTypes
            return TagIndex<EntityTypes>::template contains<EntityType>();
        }

        template <typename EntityType>
        const decltype(std::tuple_element_t<TagIndex<EntityTypes>::template index<EntityType>(), FieldTuple>())::view_type& getView() const {
            // Get index of EntityType in EntityTypes
            constexpr unsigned index = TagIndex<EntityTypes>::template index<EntityType>();

            return std::get<index>(data_m).getView();
        }

        const ViewTuple getAllViews() const {
            ViewTuple views;
            for (unsigned i = 0; i < NEntitys; ++i) {
                views[i] = std::get<i>(data_m).getView();
            }
            return views;
        }

        constexpr static auto getEntityTypes() {
            return EntityTypes{};
        }

    private:


        // Helperfunction to create numDOFs_m array
        template <unsigned... DOFNum>
        static constexpr std::array<unsigned, NEntitys> createNumDOFsArray(std::tuple<std::integral_constant<unsigned, DOFNum>...>) {
            return std::array<unsigned, NEntitys>{ DOFNum... };
        }

        static constexpr std::array<unsigned, NEntitys> numDOFs_m = createNumDOFsArray(DOFNums{}); // Number of DOFs for each field

        
        FieldTuple data_m; // Fields with DOFs for each entity type with DOFs

        std::array<SubFieldLayout<Dim>, NEntitys> layout_m;   // Layouts for each entity type with DOFs

        int nghost_m;

        Mesh_t* mesh_m; // Pointer to the mesh

        const Layout_t* VertexLayout_m; // Layout of vertices
        const Layout_t* EntityLayout_m; // Layout of highest order entities (Dim-dimensional elements)
    };
}   // namespace ippl


#include "FEMContainer.hpp"

#endif  // IPPL_FEMCONTAINER_H