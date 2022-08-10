/* -------------------------------------------------------------------------------
 * Tomocam Copyright (c) 2018
 *
 * The Regents of the University of California, through Lawrence Berkeley
 *National Laboratory (subject to receipt of any required approvals from the
 *U.S. Dept. of Energy). All rights reserved.
 *
 * If you have questions about your rights to use or distribute this software,
 * please contact Berkeley Lab's Innovation & Partnerships Office at
 *IPO@lbl.gov.
 *
 * NOTICE. This Software was developed under funding from the U.S. Department of
 * Energy and the U.S. Government consequently retains certain rights. As such,
 *the U.S. Government has been granted for itself and others acting on its
 *behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software
 *to reproduce, distribute copies to the public, prepare derivative works, and
 * perform publicly and display publicly, and to permit other to do so.
 *---------------------------------------------------------------------------------
 */


#include <cuda.h>


#include "dev_array.h"
#include "utils.cuh"

namespace tomocam {

    template <typename T>
    __global__ void _add_vecs(DeviceArray<T> a, DeviceArray<T> b, DeviceArray<T> res) {
        int idx = Index1D();
        if (idx < a.size()) res[idx] = a[idx] + b[idx];
    }

    template <typename T>
    __global__ void _subtract_vecs(DeviceArray<T> a, DeviceArray<T> b, DeviceArray<T> res) {
        int idx = Index1D();
        if (idx < a.size()) res[idx] = a[idx] - b[idx];
    }

    template <typename T>
    __global__ void _multiply_vecs(DeviceArray<T> a, DeviceArray<T> b, DeviceArray<T> res) {
        int idx = Index1D();
        if (idx < a.size()) res[idx] = a[idx] * b[idx];
    }

    template <typename T>
    __global__ void _divide_vecs(DeviceArray<T> a, DeviceArray<T> b, DeviceArray<T> res) {
        int idx = Index1D();
        if (idx < a.size()) res[idx] = a[idx] / b[idx];
    }

    template <typename T>
    __global__ void _scale_vecs(DeviceArray<T> a, T b, DeviceArray<T> res) {
        int idx = Index1D();
        if (idx < a.size()) res[idx] = a[idx] * b;
    }

    template <typename T>
    __global__ void _shift_vecs(DeviceArray<T> a, T b, DeviceArray<T> res) {
        int idx = Index1D();
        if (idx < a.size()) res[idx] = a[idx] + b;
    }

    template <typename T>
    void add_vectors(DeviceArray<T> a, DeviceArray<T> b, DeviceArray<T> res, cudaStream_t stream) {
       Grid grid(a.size());
       _add_vecs <<<grid.blocks(), grid.threads(), 0, stream>>> (a, b, res);
    }

    template <typename T>
    void subtract_vectors(DeviceArray<T> a, DeviceArray<T> b, DeviceArray<T> res, cudaStream_t stream) {
       Grid grid(a.size());
       _subtract_vecs <<<grid.blocks(), grid.threads(), 0, stream>>> (a, b, res);
    }

    template <typename T>
    void multiply_vectors(DeviceArray<T> a, DeviceArray<T> b, DeviceArray<T> res, cudaStream_t stream) {
       Grid grid(a.size());
       _multiply_vecs <<<grid.blocks(), grid.threads(), 0, stream>>> (a, b, res);
    }

    template <typename T>
    void divide_vectors(DeviceArray<T> a, DeviceArray<T> b, DeviceArray<T> res, cudaStream_t stream) {
       Grid grid(a.size());
       _divide_vecs <<<grid.blocks(), grid.threads(), 0, stream>>> (a, b, res);
    }

    template <typename T>
    void scale_vectors(DeviceArray<T> a, T b, DeviceArray<T> res, cudaStream_t stream) {
       Grid grid(a.size());
       _scale_vecs <<<grid.blocks(), grid.threads(), 0, stream>>> (a, b, res);
    }

    template <typename T>
    void shift_vectors(DeviceArray<T> a, DeviceArray<T> b, DeviceArray<T> res, cudaStream_t stream) {
       Grid grid(a.size());
       _shift_vecs <<<grid.blocks(), grid.threads(), 0, stream>>> (a, b, res);
    }
} // namespace
