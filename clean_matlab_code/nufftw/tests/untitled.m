%% check forward matches adjoint + accuracy

kd = readbin('data/phantom_spiral.kd');
w = kd(3:3:end);
d = vec2complex(readbin('data/phantom_spiral.dd'));

% nudft
%!bin/adjoint_nudft data/phantom_spiral.kd 2 128 128 out/imagedirect.dd data/phantom_spiral.dd 1
%!bin/forward_nudft data/phantom_spiral.kd 2 128 128 out/imagedirect.dd out/phantom_spiral_direct.dd 1
im = vec2complex(readbin('out/imagedirect.dd'));
d2 = vec2complex(readbin('out/phantom_spiral_direct.dd'));
(d.*w)'*d2-im(:)'*im(:)

% nufft-spm
%!bin/adjoint_nufft data/phantom_spiral.kd 2 128 128 out/imagespm.dd data/phantom_spiral.dd 1 -3 2 spm
%!bin/forward_nufft data/phantom_spiral.kd 2 128 128 out/imagespm.dd out/phantom_spiral_spm.dd 1 -3 2 spm
im2 = vec2complex(readbin('out/imagespm.dd'));
d2 = vec2complex(readbin('out/phantom_spiral_spm.dd'));
(d.*w)'*d2-im2(:)'*im2(:)
get_image_errors(im,im2);

% nufft-onthefly
%!bin/adjoint_nufft data/phantom_spiral.kd 2 128 128 out/imageonthefly.dd data/phantom_spiral.dd 1 -3 2 onthefly
%!bin/forward_nufft data/phantom_spiral.kd 2 128 128 out/imageonthefly.dd out/phantom_spiral_onthefly.dd 1 -3 2 onthefly
im2 = vec2complex(readbin('out/imageonthefly.dd'));
d2 = vec2complex(readbin('out/phantom_spiral_onthefly.dd'));
(d.*w)'*d2-im2(:)'*im2(:)
get_image_errors(im,im2);

% NFFT
%!bin/adjoint_NFFT data/phantom_spiral.kd 2 128 128 out/imagenfft.dd data/phantom_spiral.dd 1 -3 2
%!bin/forward_NFFT data/phantom_spiral.kd 2 128 128 out/imagenfft.dd out/phantom_spiral_nfft.dd 1 -3 2
im2 = vec2complex(readbin('out/imagenfft.dd'));
d2 = vec2complex(readbin('out/phantom_spiral_nfft.dd'));
(d.*w)'*d2-im2(:)'*im2(:)
get_image_errors(im,im2);

% direct NFFT
%!bin/adjoint_NFFT_direct data/phantom_spiral.kd 2 128 128 out/imagenfft_direct.dd data/phantom_spiral.dd 1 -3 2
%!bin/forward_NFFT_direct data/phantom_spiral.kd 2 128 128 out/imagenfft_direct.dd out/phantom_spiral_nfft_direct.dd 1 -3 2
im2 = vec2complex(readbin('out/imagenfft_direct.dd'));
d2 = vec2complex(readbin('out/phantom_spiral_nfft_direct.dd'));
(d.*w)'*d2-im2(:)'*im2(:)
get_image_errors(im,im2);

%% check times
%!bin/adjoint_nudft data/phantom_spiral.kd 2 128 128 out/imagedirect.dd data/phantom_spiral.dd 1
im = vec2complex(readbin('out/imagedirect.dd'));
alpha = 1.2:0.1:2;
maxerrpow = -2:-0.1:-4;
[alpha,maxerrpow] = meshgrid(alpha,maxerrpow);
alpha = alpha(:);
maxerrpow = maxerrpow(:);

n = length(alpha);
spmerr = zeros(n,2);
nffterr = zeros(n,2);
for i = 1:n
    eval(['!bin/adjoint_nufft data/phantom_spiral.kd 2 128 128 out/imagenew.dd data/phantom_spiral.dd 1 ' num2str(maxerrpow(i)) ' ' num2str(alpha(i)) ' spm 2> out.txt'])
    im2 = vec2complex(readbin('out/imagenew.dd'));
    spmerr(i,1) = maxabs(get_image_errors(im,im2));
    fid = fopen('out.txt','rt');
    fgetl(fid);
    fgetl(fid);
    spmerr(i,2) = fscanf(fid,'data transformation: %fs');
    fclose(fid);
    
    eval(['!bin/adjoint_NFFT data/phantom_spiral.kd 2 128 128 out/imagenew.dd data/phantom_spiral.dd 1 ' num2str(maxerrpow(i)) ' ' num2str(alpha(i)) ' 2> out.txt'])
    im2 = vec2complex(readbin('out/imagenew.dd'));
    nffterr(i,1) = maxabs(get_image_errors(im,im2));
    fid = fopen('out.txt','rt');
    fgetl(fid);
    fgetl(fid);
    nffterr(i,2) = fscanf(fid,'data transformation: %fs');
    fclose(fid);
end



%%
im = vec2complex(readbin('out/imagedirect.dd'));
alpha = 1.2:0.1:2;
maxerrpow = -2:-0.1:-4;
[maxerrpow,alpha] = meshgrid(maxerrpow,alpha);
alpha = alpha(:);
maxerrpow = maxerrpow(:);

n = length(alpha);
spmerr = zeros(n,2);
nffterr = zeros(n,2);
for i = 1:n
    %eval(['!bin/adjoint_nufft data/phantom_spiral.kd 2 128 128 out/imagenew.dd data/phantom_spiral.dd 1 ' num2str(maxerrpow(i)) ' ' num2str(alpha(i)) ' spm 2> out.txt'])
    
    im2 = vec2complex(readbin('out/imagenew.dd'));
    spmerr(i,1) = maxabs(get_image_errors(im,im2));
    fid = fopen('out.txt','rt');
    fgetl(fid);
    fgetl(fid);
    spmerr(i,2) = fscanf(fid,'data transformation: %fs');
    fclose(fid);
    
    eval(['!bin/adjoint_NFFT data/phantom_spiral.kd 2 128 128 out/imagenew.dd data/phantom_spiral.dd 1 ' num2str(maxerrpow(i)) ' ' num2str(alpha(i)) ' 2> out.txt'])
    im2 = vec2complex(readbin('out/imagenew.dd'));
    nffterr(i,1) = maxabs(get_image_errors(im,im2));
    fid = fopen('out.txt','rt');
    fgetl(fid);
    fgetl(fid);
    nffterr(i,2) = fscanf(fid,'data transformation: %fs');
    fclose(fid);
end




% migrate to mikgiant and try multithreading