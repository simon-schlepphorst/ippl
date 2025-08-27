# -----------------------------------------------------------------------------
# test/FEM/FEMContainer/CMakeLists.txt
#
# Integration tests for IPPL FEMContainer.
# -----------------------------------------------------------------------------

file(RELATIVE_PATH _relPath "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "🔧 Adding FEMContainer integration tests from: ${_relPath}")

file (RELATIVE_PATH _relPath "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message (STATUS "Adding index test found in ${_relPath}")

include_directories (
    ${CMAKE_SOURCE_DIR}/src
)

link_directories (
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Kokkos_DIR}/..
)

set (IPPL_LIBS ippl)
set (COMPILE_FLAGS ${OPAL_CXX_FLAGS})

add_executable (TestFEMContainer TestFEMContainer.cpp)
target_link_libraries (
    TestFEMContainer
    ${IPPL_LIBS}
    ${MPI_CXX_LIBRARIES}
)