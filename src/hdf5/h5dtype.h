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

#include <hdf5.h>

#ifndef H5DYPES_H
#define H5DYPES_H

namespace tomocam {
    namespace h5 {
        template <typename T>
        constexpr hid_t getH5Dtype() {
            if (std::is_same<T, float>::value) {
                return H5T_NATIVE_FLOAT;
            } else if (std::is_same<T, double>::value) {
                return H5T_NATIVE_DOUBLE;
            } else if (std::is_same<T, int>::value) {
                return H5T_NATIVE_INT;
            } else {
                throw std::runtime_error("Unsupported data type");
            }
        }

        template <typename T>
        struct cmplx_t {
            T re, im;
        };

        // hdf5 complex data type
        template <typename T>
        hid_t makeComplexType() {
            if (std::is_same<T, float>::value) {
                hid_t cmplx_id =
                    H5Tcreate(H5T_COMPOUND, sizeof(cmplx_t<float>));
                H5Tinsert(cmplx_id, "r", HOFFSET(cmplx_t<float>, re),
                    H5T_IEEE_F32LE);
                H5Tinsert(cmplx_id, "i", HOFFSET(cmplx_t<float>, im),
                    H5T_IEEE_F32LE);
                return cmplx_id;
            } else if (std::is_same<T, double>::value) {
                hid_t cmplx_id =
                    H5Tcreate(H5T_COMPOUND, sizeof(cmplx_t<double>));
                H5Tinsert(cmplx_id, "r", HOFFSET(cmplx_t<double>, re),
                    H5T_IEEE_F64LE);
                H5Tinsert(cmplx_id, "i", HOFFSET(cmplx_t<double>, im),
                    H5T_IEEE_F64LE);
                return cmplx_id;
            } else {
                throw std::runtime_error("Unsupported data type");
            }
        }

    } // namespace h5
} // namespace tomocam
#endif // H5DYPES_H
