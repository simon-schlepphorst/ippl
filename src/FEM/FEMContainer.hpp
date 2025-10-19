// Class FEMContainer
// This class holds a collection of DOFs (degrees of freedom) for a finite element mesh.
// The DOFs are stored in a multi-dimensional allay, split by their dimension (vertices, edges, faces, etc.).
// This allows for easy boundary condition application and field operations.

namespace ippl {

    inline std::vector<unsigned int> kth_combination(int n, int k, int dir) {

        // In case of vertex level, return empty vector
        if (k == 0) {
            return std::vector<unsigned int>{};
        }

        // Result vector with the chosen axes
        std::vector<unsigned int> result;

        // current axis
        int x = 0;

        // loop over the the number of axes we need for current element
        for (int i = 0; i < k; ++i) {

            // loop over possible axis, in case we already chose one, only consider axes with higher indices
            for (int j = x; j < n; ++j) {
                
                // how many combinations are possible if we choose j as the i-th axis
                int count = detail::binomialCoefficient(n - j - 1, k - i - 1);

                // In case we have enough possible combinations, choose j as the i-th axis
                if (dir < count) {
                    result.push_back(j);
                    x = j + 1;
                    break;
                } 

                // If we don't have enough possible combinations, move to check the next axis
                // Reduce the direction by the number of combinations we just checked
                else {
                    dir -= count;
                }
            }
        }
        return result;
    }

    template <typename T, unsigned Dim, unsigned... DOFNums>
    FEMContainer<T, Dim, DOFNums...>::FEMContainer() {}

    template <typename T, unsigned Dim, unsigned... DOFNums>
    FEMContainer<T, Dim, DOFNums...>::FEMContainer(Mesh_t& m, Layout_t& l, int nghost) {
        initialize(m, l, nghost);
    }

    template <typename T, unsigned Dim, unsigned... DOFNums>
    void FEMContainer<T, Dim, DOFNums...>::initialize(Mesh_t& m, Layout_t& l, int nghost) {
        // Set mesh and ghostcells
        nghost_m = nghost;
        mesh_m = &m;
        
        // Get domain and communicator of the layout
        NDIndex<Dim> domain = l.getDomain();

        mpi::Communicator comm = l.comm;
        auto parallel = l.isParallel();

        // Number of elements already defined
        unsigned int NElements = 0;

        // loop through all element types (vertices, edges, faces, etc.)
        for (unsigned n = 0; n < DOFsPerType.size(); ++n) {
            // Calculate the number of possible directions for this element type
            unsigned nDirs = detail::binomialCoefficient(Dim, n);

            for (unsigned dir = 0; dir < nDirs; ++dir) {
                // Set the number of degrees of freedom
                numDOFs_m[NElements] = DOFsPerType[n];
                
                // Get the direction in which we have less elements in the subdomain
                std::vector<unsigned int> axes = kth_combination(Dim, n, dir);
                
                // Create local subdomain
                NDIndex<Dim> subDomain = domain;
                for (unsigned int axis : axes) {
                    subDomain[axis] = subDomain[axis].cut(1);
                }
                
                // Create a sub field layout for this element type and direction
                layout_m[NElements] = std::make_shared<SubFieldLayout<Dim>>(comm, domain, subDomain, parallel, l.isAllPeriodic_m);
                
                if(DOFsPerType[n] == 0) {
                    data_m[NElements] = nullptr;
                    continue;
                }

                // Initialize field for this element type and direction
                data_m[NElements] = std::make_shared<Field_t>(*mesh_m, *layout_m[NElements], nghost);

                // Add created element to the total number of elements
                NElements += 1;
            }
        }
    }

    template <typename T, unsigned Dim, unsigned... DOFNums>
    FEMContainer<T, Dim, DOFNums...>& FEMContainer<T, Dim, DOFNums...>::operator=(T value) {
        for (const auto& field : data_m) {
            if (field != nullptr) {
                *field = value;
            }
        }
        return *this;
    }

    template <typename T, unsigned Dim, unsigned... DOFNums>
    FEMContainer<T, Dim, DOFNums...>& FEMContainer<T, Dim, DOFNums...>::operator=(const FEMContainer<T,Dim, DOFNums...>& other) {
        if (this != &other) {
            nghost_m = other.nghost_m;
            mesh_m = other.mesh_m;
            numDOFs_m = other.numDOFs_m;
            layout_m = other.layout_m;
            data_m = other.data_m;
        }
        return *this;
    }

    template <typename T, unsigned Dim, unsigned... DOFNums>
    void FEMContainer<T, Dim, DOFNums...>::fillHalo() {
        for (const auto& field : data_m) {
            field != nullptr ? field->fillHalo() : void();
        }
    }

    template <typename T, unsigned Dim, unsigned... DOFNums>
    void FEMContainer<T, Dim, DOFNums...>::accumulateHalo() {
        for (const auto& field : data_m) {
            field != nullptr ? field->accumulateHalo() : void();
        }
    }

    template <typename T, unsigned Dim, unsigned... DOFNums>
    void FEMContainer<T, Dim, DOFNums...>::setHalo(T setValue) {
        for (const auto& field : data_m) {
            field != nullptr ? field->setHalo(setValue) : void();
        }
    }
}