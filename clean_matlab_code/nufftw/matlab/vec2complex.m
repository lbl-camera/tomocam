function cplx = vec2complex(vec)

cplx = vec(1:2:end)+i*vec(2:2:end);