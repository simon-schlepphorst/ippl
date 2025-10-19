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

    // Element types for different types of elements in the mesh
    // These can be used to access fields associated with specific element types
    struct Vertex {
        std::array<bool, 3> dir = {false, false, false};
    };
    struct EdgeX {
        std::array<bool, 3> dir = {true, false, false};
    };
    struct EdgeY {
        std::array<bool, 3> dir = {false, true, false};
    };
    struct EdgeZ {
        std::array<bool, 3> dir = {false, false, true};
    };
    struct FaceXY {
        std::array<bool, 3> dir = {true, true, false};
    };
    struct FaceXZ {
        std::array<bool, 3> dir = {true, false, true};
    };
    struct FaceYZ {
        std::array<bool, 3> dir = {false, true, true};
    };
    struct Hexaedron {
        std::array<bool, 3> dir = {true, true, true};
    };

    enum SpaceType {
        Lagrange,
        RaviartThomas,
        Nedelec
    };

    // FEMTraits to associate element types with the different FEM spaces
    template <typename SpaceType, usigned Dim, unsinged Order>
    struct FEMTraits;

    template 



    // FieldTraits to associate element types with their respective DOF counts
    template <typename T, unsigned Dim, typename Mesh_t, typename ElementType, unsigned NDOF>
    struct FieldTraits {
        // TODO: Currently only supports Cell centering, if Field changes, use centering that matches ElementType
        static constexpr unsigned numDOFs = NDOF;
        using type = Field<Kokkos::Array<T, NDOF>, Dim, Mesh_t, Cell>;
    };


    // FieldTupleBuilder to build a tuple of field types based on ElementTypes and DOFNums
    template <typename T, unsigned Dim, typename Mesh_t>
    struct FieldTupleBuilder;

    // Base case for FieldTupleBuilder
    template <typename T, unsigned Dim, typename Mesh_t>
    struct FieldTupleBuilder<T, Dim, Mesh_t> {
        using type = std::tuple<>;
    };

    // Recursive case to build the tuple of field types
    template <typename T, unsigned Dim, typename Mesh_t, typename ElementType, typename... ElementTypeRest, unsigned DOFNum, unsigned... DOFNumRest>
    struct FieldTupleBuilder<T, Dim, Mesh_t, ElementType, ElementTypeRest..., DOFNum, DOFNumRest...> {
        
        using CurrentField = typename FieldTraits<T, Dim, Mesh_t, ElementType, DOFNum>::type;
        using type = decltype(std::tuple_cat(
            std::make_tuple(std::shared_ptr<CurrentField>()),
            typename FieldTupleBuilder<T, Dim, Mesh_t, ElementTypeRest..., DOFNumRest...>::type>()));
    };


    
    // Helper to get index of ElementType in ElementTypes...
    template <typename... ElementTypes>
    struct TagIndex {
        template <typename ElementType>
        constexpr static bool contains() {
            return (std::is_same<ElementType, ElementTypes>::value || ...);
        }

        template <typename ElementType>
        constexpr static int index() {
            static_assert(contains<ElementType>(), "ElementType not found in ElementTypes...");
            return (std::is_same<ElementType, ElementTypes>::value ? 0 : ... + 1);
        }
    };

    template <typename T, unsigned Dim, typename... ElementTypes, unsigned... DOFNums>
    class FEMContainer {
    public:

        typedef typename detail::ViewType<T, Dim>::view_type ViewType;
        typedef typename detail::ViewType<T, Dim, Kokkos::MemoryTraits<Kokkos::Atomic>>::view_type AtomicViewType;

        using Mesh_t      = UniformCartesian<T, Dim>;
        using Layout_t    = FieldLayout<Dim>;

        // Build tuple of field types
        using FieldTuple = typename FieldTupleBuilder<T, Dim, Mesh_t, ElementTypes..., DOFNums...>::type;
        using LayoutTuple = typename LayoutTupleBuilder<Layout_t, sizeof...(ElementTypes)>::type;

        static_assert(sizeof...(ElementTypes) == sizeof...(DOFNums), "Number of ElementTypes must match number of DOFNums");
        // TODO: Add static assert to check no duplicate ElementTypes

        FEMContainer();
        FEMContainer(Mesh_t& m, Layout_t& l, int nghost = 1);
        FEMContainer(const FEMContainer<T, Dim>& other);

        void initialize(Mesh_t& m, Layout_t& l, int nghost = 1);


        FEMContainer<T, Dim, ElementTypes..., DOFNums...>& operator=(T value);
        FEMContainer<T, Dim, ElementTypes..., DOFNums...>& operator=(const FEMContainer<T,Dim, ElementTypes..., DOFNums...>& other);

        void fillHalo();
        void accumulateHalo();
        void setHalo(T setValue);

        int getNGhost() const { return nghost_m; }

        unsigned int getNumFields() { return numDOFs_m.size(); }

        KOKKOS_INLINE_FUNCTION Mesh_t& get_mesh() const { return *mesh_m; }

        KOKKOS_INLINE_FUNCTION Layout_t& getVertexLayout() const { return VertexLayout_m; }
        KOKKOS_INLINE_FUNCTION Layout_t& getNumElementLayout() const { return ElementLayout_m; }

        // Access individual layouts
        template <typename ElementType>
        std::shared_ptr<FieldLayout<Dim>> getLayout() const { 
            constexpr unsigned index = tagIndex_m.index<ElementType>();
            return std::get<index>(layout_m); 
        }
        
        // Get the number of DOFs for each element type
        template <typename ElementType>
        unsigned getNumDOFs() const {
            constexpr unsigned index = tagIndex_m.index<ElementType>();
            return numDOFs_m[index];
        }

        template <typename ElementType>
        bool hasView() const {
            // check if ElementType is in ElementTypes...
            return tagIndex_m.contains<ElementType>();
        }

        template <typename ElementType>
        const ViewType& getView() const {
            // Get index of ElementType in ElementTypes...
            constexpr unsigned index = tagIndex_m.index<ElementType>();

            return std::get<index>(data_m)->getView();
        }

        const std::array<ViewType, sizeof...(ElementTypes)> getAllViews() const {
            std::array<ViewType, sizeof...(ElementTypes)> views = {std::get<TagIndex<ElementTypes...>::template index<ElementTypes>()>(data_m)->getView()...};
            return views;
        }

    private:

        static constexpr std::array<unsigned, sizeof...(DOFNums)> numDOFs_m = {DOFNums...}; // Number of DOFs for each field
        
        FieldTuple data_m; // Fields with DOFs for each element type with DOFs

        std::array<std::shared_ptr<FieldLayout<Dim>>, sizeof...(ElementTypes)> layout_m;   // Layouts for each element type with DOFs

        int nghost_m;

        Mesh_t* mesh_m; // Pointer to the mesh

        Layout_t* VertexLayout_m; // Pointer to the layout of vertices
        Layout_t* ElementLayout_m; // Pointer to the layout of highest order elements (Dim-dimensional elements)

        TagIndex<ElementTypes...> tagIndex_m; // Helper to get index of ElementType in ElementTypes...
    };
}   // namespace ippl


#include "FEMContainer.hpp"

#endif  // IPPL_FEMCONTAINER_H