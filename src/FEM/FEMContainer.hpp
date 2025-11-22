// Class FEMContainer
// This class holds a collection of DOFs (degrees of freedom) for a finite element mesh.
// The DOFs are stored in a multi-dimensional allay, split by their dimension (vertices, edges, faces, etc.).
// This allows for easy boundary condition application and field operations.

namespace ippl {

    template <typename T, unsigned Dim, typename EntityTypes, typename DOFNums>
    FEMContainer<T, Dim, EntityTypes, DOFNums>::FEMContainer() {}

    template <typename T, unsigned Dim, typename EntityTypes, typename DOFNums>
    FEMContainer<T, Dim, EntityTypes, DOFNums>::FEMContainer(Mesh_t& m, const Layout_t& l, int nghost) {
        initialize(m, l, nghost);
    }

    template <typename T, unsigned Dim, typename EntityTypes, typename DOFNums>
    void FEMContainer<T, Dim, EntityTypes, DOFNums>::initialize(Mesh_t& m, const Layout_t& l, int nghost) {

        // Set mesh and ghostcells
        nghost_m = nghost;
        mesh_m = &m;
        
        // Get domain and communicator of the layout
        NDIndex<Dim> domain = l.getDomain();

        mpi::Communicator comm = l.comm;
        auto parallel = l.isParallel();

        // loop trough all the entity types
        [&]<std::size_t... i>(std::index_sequence<i...>) {
            // Fold over all indices
            (([&]() {                
                // Get the dimension of this entity type
                std::array<bool, Dim> dir = std::tuple_element_t<i, EntityTypes>{}.getDir();
                
                // Create local subdomain
                NDIndex<Dim> subDomain = domain;
                
                for (unsigned int d = 0; d < Dim; ++d) {
                    subDomain[d] = dir[d] ? subDomain[d].cut(1) : subDomain[d];
                }
                
                // Create a sub field layout for this element type and direction
                layout_m[i].comm = comm;
                layout_m[i].initialize(domain, subDomain, parallel, l.isAllPeriodic_m);
                
                // Initialize field for this element type and direction
                std::get<i>(data_m).initialize(*mesh_m, layout_m[i], nghost);
            })(), ...);
        }(std::make_index_sequence<std::tuple_size_v<EntityTypes>>{});
    }

    template <typename T, unsigned Dim, typename EntityTypes, typename DOFNums>
    FEMContainer<T, Dim, EntityTypes, DOFNums>& FEMContainer<T, Dim, EntityTypes, DOFNums>::operator=(T value) {
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            (([&]() {
                constexpr std::size_t arraySize = std::get<Is>(DOFNums{}).value;
                DOFArray<T, arraySize> arr(value);
                std::get<Is>(data_m) = arr;
            }()), ...);
        }(std::make_index_sequence<std::tuple_size_v<decltype(data_m)>>{});
       
        return *this;
    }

    template <typename T, unsigned Dim, typename EntityTypes, typename DOFNums>
    FEMContainer<T, Dim, EntityTypes, DOFNums>& FEMContainer<T, Dim, EntityTypes, DOFNums>::operator+=(T value) {
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            (([&]() {
                constexpr std::size_t arraySize = std::get<Is>(DOFNums{}).value;
                DOFArray<T, arraySize> arr(value);
                std::get<Is>(data_m) += arr;
            }()), ...);
        }(std::make_index_sequence<std::tuple_size_v<decltype(data_m)>>{});
       
        return *this;
    }

    // TODO:
    // Add copy functionality
    /*
    template <typename T, unsigned Dim, typename EntityTypes, typename DOFNums>
    FEMContainer<T, Dim, EntityTypes, DOFNums>& FEMContainer<T, Dim, EntityTypes, DOFNums>::operator=(const FEMContainer<T, Dim, EntityTypes, DOFNums>& other) {
    }
    */

    template <typename T, unsigned Dim, typename EntityTypes, typename DOFNums>
    void FEMContainer<T, Dim, EntityTypes, DOFNums>::fillHalo() {
        std::apply([&](auto&... fields) {
            ((fields.fillHalo()), ...);
        }, data_m);
    }

    template <typename T, unsigned Dim, typename EntityTypes, typename DOFNums>
    void FEMContainer<T, Dim, EntityTypes, DOFNums>::accumulateHalo() {
        std::apply([&](auto&... fields) {
            ((fields.accumulateHalo()), ...);
        }, data_m);
    }
}