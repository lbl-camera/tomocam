%Test script for a single slice recon from BL 832 data 

addpath operators
addpath gpu
addpath gnufft
addpath Common

file_name ='/home/svvenkatakrishnan/data/20130807_234356_OIM121R_SAXS_5x.mat';
%ShepLogan_2560_2049_dose5000_noise_0_1.mat';
%projection = projection(1:4:2048,1:end-1);
%Padding data --TODO : Should happen automatically inside the main code 

load(file_name);
Nr=size(projection,2);


%Forward model params 
formodel.center = 1294;
formodel.pix_size = 1;
formodel.det_size = 1;
formodel.Npad = 3000;

%KB window params 
formodel.k_r=2;
formodel.beta =3*pi*1.0;

%Prior model params 
prior.reg_param = 0.25;

%Solver params
opts.maxIts           = 50;
opts.maxLSIts         = 150;
opts.gradTol          = 1e-30;
opts.weightTV         = prior.reg_param;
opts.weightLp         = 0.0;
opts.pNorm            = 1;
opts.qNorm            = 1;
opts.alpha            = 0.01;
opts.beta             = 0.6;%?
opts.mu               = 1e-12;%?

%
tic;
[recon]=MBIRTV(projection,ones(size(projection)),zeros(Nr,Nr),formodel,prior,opts);
toc;

recon_original_size = real(recon(formodel.Npad/2 - Nr/2:formodel.Npad/2 + Nr/2 -1 ,formodel.Npad/2 - Nr/2:formodel.Npad/2 + Nr/2 -1 ));

imagesc(recon_original_size.');axis image;colormap(gray);colorbar;
