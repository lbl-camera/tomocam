

addpath('../../packages/mritut/nuFFT_tutorial/code')
addpath('../code/MATLAB')
alpha = 1.2:0.2:2;%1.2:0.1:2;
maxerr = -1.5:-0.5:-4;%-1.5:-0.25:-5;
[alpha, maxerr] = meshgrid(alpha, maxerr);
alpha = alpha(:);
maxerr = maxerr(:);
N = 126;%:129;
Ntrials = 6;%10


enufft = zeros(length(alpha),length(N));
enfft = zeros(length(alpha),length(N));
tnufft = zeros(length(alpha),length(N),Ntrials);
tnfft = zeros(length(alpha),length(N),Ntrials);
for j = 1:length(N)
    disp(N(j))
    %eval(['!../bin/adjoint_nudft ../data/phantom_spiral.kd 2 ' int2str(N(j)) ' ' int2str(N(j)) ' ../out/imagedirect' int2str(N(j)) '.dd ../data/phantom_spiral.dd 1'])
    imagedirect = read_image(['../out/imagedirect' int2str(N(j)) '.dd']);
    for i = 1:length(alpha)
        for k = 1:Ntrials
            eval(['!../bin/adjoint_nufft ../data/phantom_spiral.kd 2 ' int2str(N(j)) ' ' int2str(N(j)) ' ../out/imagespm.dd ../data/phantom_spiral.dd 1 ' num2str(maxerr(i)) ' ' num2str(alpha(i)) ' spm 2> out.txt'])
            fid = fopen('out.txt','rt');
            fgetl(fid);
            fgetl(fid);
            tnufft(i,j,k) = fscanf(fid,'data transformation: %fs\n');
            fclose(fid);

            eval(['!../bin/adjoint_NFFT ../data/phantom_spiral.kd 2 ' int2str(N(j)) ' ' int2str(N(j)) ' ../out/imagenfft.dd ../data/phantom_spiral.dd 1 ' num2str(maxerr(i)) ' ' num2str(alpha(i)) ' 2> out.txt'])
            fid = fopen('out.txt','rt');
            fgetl(fid);
            fgetl(fid);
            tnfft(i,j,k) = fscanf(fid,'data transformation: %fs\n');
            fclose(fid);
        end
        imagespm = read_image('../out/imagespm.dd');
        e = get_image_errors(imagedirect,imagespm);
        enufft(i,j) = max(e(:));
        
        imagenfft = read_image('../out/imagenfft.dd');
        e = get_image_errors(imagedirect,imagenfft);
        enfft(i,j) = max(e(:));
    end
end