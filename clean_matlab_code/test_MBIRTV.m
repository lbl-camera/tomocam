%Test script for a single slice recon from BL 832 data 
clc;
clear;
close all;

addpath operators
addpath gpu
addpath gnufft
addpath Common

file_name ='/home/svvenkatakrishnan/data/ShepLogan_2560_2049_dose5000_noise_0_1.mat';


%20130807_234356_OIM121R_SAXS_5x.mat';

 
load(file_name);

projection = projection(1:8:2048,1:end-1);

% imsize = 512/2;
% P=phantom(imsize);
% nangles = 512/4;
% projection = radon(P, 0:180/nangles:180-180/nangles);
% projection = projection((end-1)/2-imsize/2:(end-1)/2+imsize/2-1,:).';


Nr=size(projection,2);

%Forward model params 
formodel.center =1280;%imsize/2;%1280;%1294;
formodel.pix_size = 1;
formodel.det_size = 1;
formodel.Npad = 1000;%3200;%3200;
formodel.ring_corr = 0;

%KB window params 
formodel.k_r=2;
formodel.beta =3*pi*1.0;

%Prior model params 
prior.reg_param = 10;

%Solver params
opts.maxIts           = 150/3;
opts.maxLSIts         = 50;
opts.gradTol          = 1e-30;
opts.weightTV         = 1;%prior.reg_param;
opts.gammaTV          = prior.reg_param;
opts.gammaLp          = 0;
opts.weightLp         = 0;
opts.pNorm            = 1;
opts.qNorm            = 1;
opts.alpha            = 0.01;
opts.beta             = 0.6;%?
opts.mu               = 1e-12;%the rounding value used to make a differentiable TV regularizer

%
temp_weight = rand(size(projection));
temp_weight(temp_weight>0)=1;
temp_weight(temp_weight<=0)=0;
tic;
[recon,x0]=MBIRTV(projection,temp_weight,zeros(Nr,Nr),formodel,prior,opts);
toc;


recon_original_size = real(recon(formodel.Npad/2 - Nr/2:formodel.Npad/2 + Nr/2 -1 ,formodel.Npad/2 - Nr/2:formodel.Npad/2 + Nr/2 -1 ));
x0 = real(x0(formodel.Npad/2 - Nr/2:formodel.Npad/2 + Nr/2 -1 ,formodel.Npad/2 - Nr/2:formodel.Npad/2 + Nr/2 -1 ));
%imagesc(recon_original_size.');axis image;colormap(gray);colorbar;

recon_original_size = flipud(recon_original_size.');
x0 = flipud(x0.');

figure;imagesc(x0);axis image;colormap(gray);colorbar;title('NUFFT Back Proj');
figure;imagesc(recon_original_size);axis image;colormap(gray);colorbar;title('MBIR TV');
line_idx = int16(0.8*Nr);
figure;plot(x0(line_idx,:));hold on;plot(recon_original_size(line_idx,:),'r');legend('NUFFT Back Proj','MBIR TV');title(strcat('Reg=',num2str(prior.reg_param)));