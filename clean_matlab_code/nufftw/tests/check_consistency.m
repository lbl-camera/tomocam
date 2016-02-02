% check_consistency - make sure the C++ and MATLAB nuFFT implementations
%   are correct, return the same results, and that the forward and adjoint
%   are consistent with each other

addpath('../../packages/mritut/nuFFT_tutorial/code')
addpath('../code/MATLAB')
load ../../packages/mritut/nuFFT_tutorial/data/noncartesian_phantom.mat
alpha = [1.3 2];
maxerr = [-2 -3];
[alpha, maxerr] = meshgrid(alpha, maxerr);
alpha = alpha(:);
maxerr = maxerr(:);
N = 126:129;


me = zeros(length(alpha),length(N));
mc = zeros(length(alpha),length(N));
ce = zeros(length(alpha),length(N));
cc = zeros(length(alpha),length(N));
for j = 1:length(N)
    disp(N(j))
    %eval(['!../bin/adjoint_nudft ../data/phantom_spiral.kd 2 ' int2str(N(j)) ' ' int2str(N(j)) ' ../out/imagedirect' int2str(N(j)) '.dd data/phantom_spiral.dd 1'])
    imagedirect = read_image(['../out/imagedirect' int2str(N(j)) '.dd']);
    for i = 1:length(alpha)
        nufft_st = init_nufft(kcoord, N(j), alpha(i), 10^(maxerr(i)));
        im = adjoint_nufft(kdata.*dcf, nufft_st);
        e = get_image_errors(imagedirect,im);
        me(i,j) = max(e(:));
        kdata2 = forward_nufft(im, nufft_st);
        mc(i,j) = (kdata.*dcf)'*kdata2-im(:)'*im(:);
        
        eval(['!../bin/adjoint_nufft ../data/phantom_spiral.kd 2 ' int2str(N(j)) ' ' int2str(N(j)) ' ../out/imagespm.dd ../data/phantom_spiral.dd 1 ' int2str(maxerr(i)) ' ' num2str(alpha(i)) ' spm'])
        imagespm = read_image('../out/imagespm.dd');
        e = get_image_errors(imagedirect,imagespm);
        ce(i,j) = max(e(:));
        eval(['!bin/forward_nufft data/phantom_spiral.kd 2 ' int2str(N(j)) ' ' int2str(N(j)) ' out/imagespm.dd out/phantom_spiral_spm.dd 1 ' int2str(maxerr(i)) ' ' num2str(alpha(i)) ' spm'])
        kdata2spm = vec2complex(readbin('out/phantom_spiral_spm.dd'));
        cc(i,j) = (kdata.*dcf)'*kdata2spm-imagespm(:)'*imagespm(:);
    end
end
