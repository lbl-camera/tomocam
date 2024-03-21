#ifndef CIRCSHIFT__CUH
#define CIRCSHIFT__CUH

namespace tomocam {
    namespace gpu_ops {
        template <typename T>
        void circshift(T *out, T *in, int nrows, int ncols, int yshift, int xshift);

        template <typename T>
        inline void fftshift(T *out, T *in, int nrows, int ncols) {
            circshift(out, in, nrows, ncols, nrows / 2, ncols / 2);
        }

        template <typename T>
        inline void ifftshift(T *out, T *in, int nrows, int ncols) {
            circshift(out, in, nrows, ncols, (nrows + 1) / 2, (ncols + 1) / 2);
        }
    } // namespace gpu_ops
} // namespace tomocam

#endif // CIRCSHIFT__CUH
