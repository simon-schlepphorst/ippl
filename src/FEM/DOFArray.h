// struct wrapped around Kokkos Array to allow operators on Kokkos::Array

#ifndef IPPL_DOFARRAY_H
#define IPPL_DOFARRAY_H

namespace ippl {

    // Helper struct wrapped around Kokkos Array to allow operators on Kokkos::Array
    template <typename T, std::size_t N>
    struct DOFArray {
        Kokkos::Array<T, N> data;

        KOKKOS_INLINE_FUNCTION
        DOFArray() = default;

        KOKKOS_INLINE_FUNCTION
        DOFArray(T value) {
            for (std::size_t i = 0; i < N; ++i) {
                data[i] = value;
            }
        }

        KOKKOS_INLINE_FUNCTION
        T& operator[](int i) { return data[i]; }          // read/write access
        const T& operator[](int i) const { return data[i]; } // read-only access

        KOKKOS_INLINE_FUNCTION
        DOFArray& operator+=(const DOFArray& other) {
            for (std::size_t i = 0; i < N; ++i) {
                data[i] += other.data[i];
            }
            return *this;
        }

        KOKKOS_INLINE_FUNCTION
        DOFArray& operator+=(T value) {
            for (std::size_t i = 0; i < N; ++i) {
                data[i] += value;
            }
            return *this;
        }

        KOKKOS_INLINE_FUNCTION
        DOFArray operator+(T value) const {
            DOFArray result;
            for (std::size_t i = 0; i < N; ++i) {
                result.data[i] = data[i] + value;
            }
            return result;
        }

        KOKKOS_INLINE_FUNCTION
        DOFArray operator-(T value) const {
            DOFArray result;
            for (std::size_t i = 0; i < N; ++i) {
                result.data[i] = data[i] - value;
            }
            return result;
        }
        KOKKOS_INLINE_FUNCTION
        DOFArray operator=(T value) {
            for (std::size_t i = 0; i < N; ++i) {
                data[i] = value;
            }
            return *this;
        }
    };

} // namespace ippl
#endif // IPPL_DOFARRAY_H