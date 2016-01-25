addpath ../bin
addpath ../matlab/
addpath matlab
addpath mex
load ../data/rt_spiral_03.mat

trajectory1 = k(:);
trajectory1 = [real(trajectory1) imag(trajectory1)];
trajectory1 = trajectory1';
trajectory1 = trajectory1(:);

dcf = w(:);
sqrtdcf = sqrt(dcf);

imp1 = nufft_implementation(1, [128 128], 1e-3, 2, 'spm', trajectory1, sqrtdcf);

kdata = d(:);
kdata_denscomp = complex2vec(kdata.*sqrtdcf);
imag_data1 = imp1.adjoint(1,kdata_denscomp);



% for i = 1:1000
%     m1(i) = java.lang.Runtime.getRuntime.freeMemory;
%     imp = nufft_implementation(1, [128 128], 1e-3, 2, 'spm', trajectory1, dcf);
% end
% m1(1001) = java.lang.Runtime.getRuntime.freeMemory;


trajectory2 = k(:);
trajectory2 = [imag(trajectory2) real(trajectory2)];
trajectory2 = trajectory2';
trajectory2 = trajectory2(:);

imp2 = nufft_implementation(1, [128 128], 1e-3, 2, 'spm', trajectory2, sqrtdcf);

imag_data2 = imp2.adjoint(1,kdata_denscomp);

figure
subplot(131)
imshowz(reshape(vec2complex(imag_data1),128,128))
subplot(132)
imshowz(reshape(vec2complex(imag_data2),128,128))

kdata_denscomp2 = imp1.forward(1, imag_data1);

img = vec2complex(imag_data1);
k1 = vec2complex(kdata_denscomp);
k2 = vec2complex(kdata_denscomp2);

k2'*k1-img'*img


imagedirect = read_image('../out/spiral2d_phantom_imagedirect.cdb');
e = get_image_errors(imagedirect,reshape(vec2complex(imag_data1),128,128));
subplot(133)
imshowz(e)
colorbar