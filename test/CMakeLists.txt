
include_directories(${CMAKE_SOURCE_DIR}/src)

# regression test: NUFFT plan batching (ntrans) correctness
add_executable(nufft_batch nufft_batch.cpp)
target_link_libraries(nufft_batch PRIVATE tomocam)
add_test(NAME nufft_batch_test COMMAND nufft_batch)

# adjoint check: <R x, y> == <x, R^T y> for project()/backproject()
add_executable(adjoint adjoint.cpp)
target_link_libraries(adjoint PRIVATE tomocam)
add_test(NAME adjoint_test COMMAND adjoint)

# gradient correctness: R^T(Rx-y) (direct project/backproject) vs.
# gradient() (nufft-cached) vs. gradient2() (Toeplitz/PSF)
add_executable(gradient gradient.cpp)
target_link_libraries(gradient PRIVATE tomocam)
add_test(NAME gradient_test COMMAND gradient)

# finite-difference gradient/divergence adjoint + laplacian sanity check
add_executable(finitdiff finitdiff.cpp)
target_link_libraries(finitdiff PRIVATE tomocam)
add_test(NAME finitdiff_test COMMAND finitdiff)

# GPU isotropic-TV shrinkage kernel vs. reference CPU formula
add_executable(shrinkage shrinkage.cpp)
target_link_libraries(shrinkage PRIVATE tomocam)
add_test(NAME shrinkage_test COMMAND shrinkage)

# GPU add_total_var2 (qGGMRF TV penalty, used by nagopt) vs. CPU reference
add_executable(total_var total_var.cpp)
target_link_libraries(total_var PRIVATE tomocam)
add_test(NAME total_var_test COMMAND total_var)

# cgsolver: SPD synthetic operator (I - mu*laplacian), known solution
add_executable(cgsolver cgsolver.cpp)
target_link_libraries(cgsolver PRIVATE tomocam)
add_test(NAME cgsolver_test COMMAND cgsolver)

# nagopt: synthetic diagonal quadratic, FISTA convergence check
add_executable(nagopt nagopt.cpp)
target_link_libraries(nagopt PRIVATE tomocam)
add_test(NAME nagopt_test COMMAND nagopt)

# split-Bregman TV denoiser smoke test (A = identity)
add_executable(bregman bregman.cpp)
target_link_libraries(bregman PRIVATE tomocam)
add_test(NAME bregman_test COMMAND bregman)

# manual benchmark: chunk-count overhead vs. bandwidth in dist_array_ops.h
# (not registered with add_test -- run manually, not part of ctest)
add_executable(bench_chunking bench_chunking.cpp)
target_link_libraries(bench_chunking PRIVATE tomocam)

# manual benchmark: chunk-count overhead for gradient2 (compute-bound,
# Toeplitz/FFT-based, unlike the memory-bound bench_chunking ops)
add_executable(bench_gradient2 bench_gradient2.cpp)
target_link_libraries(bench_gradient2 PRIVATE tomocam)

# manual benchmark: chunk-count overhead for function_value2
add_executable(bench_function_value2 bench_function_value2.cpp)
target_link_libraries(bench_function_value2 PRIVATE tomocam)

# manual benchmark: chunk-count overhead for add_total_var2 (halo-partitioned)
add_executable(bench_add_total_var2 bench_add_total_var2.cpp)
target_link_libraries(bench_add_total_var2 PRIVATE tomocam)

# manual benchmark: chunk-count overhead for function_value (direct NUFFT
# projection, non-Toeplitz sibling of function_value2)
add_executable(bench_function_value bench_function_value.cpp)
target_link_libraries(bench_function_value PRIVATE tomocam)
