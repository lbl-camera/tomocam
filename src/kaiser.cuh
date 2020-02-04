
namespace tomocam {
    __device__ __forceinline__
    float kaiser(float k, float W, float beta) {

        float weight =  0.f;
        float KMAX = (W - 1) * 0.5;
        if (fabsf(k) < KMAX) {
            float r   = 2 * k / W;
            float x   = cyl_bessel_i0(beta * sqrtf(1 - r * r));
            float y   = cyl_bessel_i0(beta);
            weight = x / y;
        }
        return weight;
    }

    __device__ __forceinline__
    float kaiser_fourier_trans(float x, float W, float beta) {
        const float PI = 3.14159265359f;

        float weight = 1.f;
        float t1 = (beta * beta) - powf(x * W * PI, 2);
        if (t1 > 0 ) {
            t1 = sqrtf(t1);
            float t2 = W / cyl_bessel_i0(beta);
            weight = (t2 * sinhf(t1)/t1);
        }
        return weight;
    }

} // namespace tomocam
