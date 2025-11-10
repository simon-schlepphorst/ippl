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

#include <vector>
#include <memory>
#include <memory>
#include <tuple>
#include <type_traits>

namespace ippl {

    // Helper struct to check weather templated input is a std::tuple
    // Default case: not a tuple
    template <typename T>
    struct is_tuple : std::false_type {};

    // Specialization for std::tuple
    template <typename... Ts>
    struct is_tuple<std::tuple<Ts...>> : std::true_type {};

    // Convenience variable template
    template <typename T>
    inline constexpr bool is_tuple_v = is_tuple<T>::value;



    // Helper struct to verify that all types in tuple are unique (we do not want duplicate EntityTypes)
    // Check if a single type T is in the list of types Rest...
    template <typename T, typename... Rest>
    struct contains : std::disjunction<std::is_same<T, Rest>...> {};

    // Recursive duplicate check
    template <typename... Ts>
    struct has_duplicates;

    // Default case: no types, no duplicates
    template <>
    struct has_duplicates<> : std::false_type {};

    // Recursive case
    template <typename T, typename... Ts>
    struct has_duplicates<T, Ts...>
        : std::disjunction<contains<T, Ts...>, has_duplicates<Ts...>> {};

    // Convenience variable template
    template <typename... Ts>
    inline constexpr bool has_duplicates_v = has_duplicates<Ts...>::value;

    // Wrapper for tuples
    template <typename Tuple>
    struct tuple_has_duplicates;

    template <typename... Ts>
    struct tuple_has_duplicates<std::tuple<Ts...>> : has_duplicates<Ts...> {};

    // Convenience variable template
    template <typename Tuple>
    inline constexpr bool tuple_has_duplicates_v = tuple_has_duplicates<Tuple>::value;




    // Helper struct to map EntityTypes and DOFNums to corresponding tuple of Field types
    // FieldTraits to map EntityType and NDOF to corresponding Field type
    template <typename T, unsigned Dim, typename Mesh_t, typename EntityType, unsigned NDOF>
    struct FieldTraits {
        // TODO: Currently only supports Cell centering, if Field changes, use centering that matches EntityType
        using type = Field<DOFArray<T, NDOF>, Dim, Mesh_t, Cell>;
    };


    // FieldTupleBuilder to build a tuple of field types based on EntityTypes and DOFNums
    template <typename T, unsigned Dim, typename Mesh_t, typename EntityTypes, typename DOFNums>
    struct FieldTupleBuilder;

    // Recursive case to build the tuple of field types
    template <typename T, unsigned Dim, typename Mesh_t, typename... EntityType, unsigned... DOFNum>
    struct FieldTupleBuilder<T, Dim, Mesh_t, std::tuple<EntityType...>, std::tuple<std::integral_constant<unsigned, DOFNum>...>> {

        using type = std::tuple<typename FieldTraits<T, Dim, Mesh_t, EntityType, DOFNum>::type...>;
        using view_type = std::tuple<typename FieldTraits<T, Dim, Mesh_t, EntityType, DOFNum>::type::view_type...>;
    };


    // Helper to get index of EntityType in EntityTypes...
    // TagIndex to index to access Fields over Entity Type
    template <typename EntityTypes>
    struct TagIndex;

    template <typename... EntityType>
    struct TagIndex<std::tuple<EntityType...>> {
        
        template <typename TestEntity>
        constexpr static bool contains() {
            return std::disjunction_v<std::is_same<TestEntity, EntityType>...>;
        }
        
        template <typename TestEntity>
        constexpr static unsigned index() {
            static_assert(contains<TestEntity>(), "EntityType not found in this FEMContainer");
            
            constexpr unsigned index = []<size_t... Is>(std::index_sequence<Is...>) {
                unsigned result = static_cast<unsigned>(-1);
                ((std::is_same_v<TestEntity, EntityType> ? result = static_cast<unsigned>(Is) : 0), ...);
                return result;
            }(std::make_index_sequence<sizeof...(EntityType)>{});
            
            return index;
        }
    };        
    
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
        FEMContainer(Mesh_t& m, Layout_t& l, int nghost = 1);
        FEMContainer(const FEMContainer<T, Dim, EntityTypes, DOFNums>& other);

        void initialize(Mesh_t& m, Layout_t& l, int nghost = 1);


        FEMContainer<T, Dim, EntityTypes, DOFNums>& operator=(T value);
        FEMContainer<T, Dim, EntityTypes, DOFNums>& operator=(const FEMContainer<T,Dim, EntityTypes, DOFNums>& other);

        FEMContainer<T, Dim, EntityTypes, DOFNums>& operator+=(const FEMContainer<T, Dim, EntityTypes, DOFNums>& other);
        FEMContainer<T, Dim, EntityTypes, DOFNums>& operator+=(T value);

        void fillHalo();
        void accumulateHalo();
//        void setHalo(T setValue);

        int getNGhost() const { return nghost_m; }

        unsigned int getNumFields() { return NEntitys; }

        KOKKOS_INLINE_FUNCTION Mesh_t& get_mesh() const { return *mesh_m; }

        KOKKOS_INLINE_FUNCTION Layout_t& getVertexLayout() const { return VertexLayout_m; }
        KOKKOS_INLINE_FUNCTION Layout_t& getNumEntityLayout() const { return EntityLayout_m; }

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

        Layout_t* VertexLayout_m; // Layout of vertices
        Layout_t* EntityLayout_m; // Layout of highest order entities (Dim-dimensional elements)
    };
}   // namespace ippl


#include "FEMContainer.hpp"

#endif  // IPPL_FEMCONTAINER_H