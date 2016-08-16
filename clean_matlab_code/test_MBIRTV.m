%Test script for a single slice recon from BL 832 data 
clc;
clear;
close all;
addpath operators;
addpath gpu;
addpath gnufft;
addpath Common;
reset(gpuDevice(2));
gpuDevice(2);
file_name ='/home/svvenkatakrishnan/data/20130807_234356_OIM121R_SAXS_5x.mat';
%ShepLogan_2560_2049_dose5000_noise_1.mat';
%ShepLogan_2560_2049_dose5000_noise_0_5.mat';
grnd_truth = phantom(2560);%2560
grnd_truth(grnd_truth < 0)=0;
grnd_truth=grnd_truth*10e-3;
 
load(file_name);
% 
%projection = projection(1:2:2048,1:end-1);
%projection = projection(1:2:end,1:end);
weight =ones(size(projection));
%(noisy_data(1:2:end,1:end-1));
%weight = sqrt(weight./sum(weight(:)));

%% Ring addition
% img = zeros(size(projection));
% img(:,end/4)=1;
% img(:,3*end/4)=1;
% projection = projection + (.05*max(projection(:))).*img;

 num_angles=size(projection,1);
 angle_list = 0:180/num_angles:180-180/num_angles;
% 
%  imsize = 2560;
%  P=(phantom(imsize));
% nangles = 512;%512/4;
%  projection = radon(P, angle_list);
%  projection = projection((end-1)/2-imsize/2:(end-1)/2+imsize/2-1,:).';
%  weight = ones(size(projection));
% 
% 

Nr=size(projection,2);

%Forward model params 
formodel.center = 1294;%1280;%128;%1280;%1294;%1280;%1294;
formodel.pix_size = 1;
formodel.det_size = 1;
formodel.Npad = 3200;%4000;%512;%3000;%3200;%3200;
formodel.ring_corr = 1;
formodel.angle_list = angle_list;

%KB window params 
formodel.k_r=2;
formodel.beta =3*pi*1.0;

%Prior model params 
prior.reg_param =  2.5/8000;

%Solver params
opts.maxIts           = 500;%Max iterations of cost-function 
opts.maxLSIts         = 150;%max line-search iterations
opts.gradTol          = 1e-30;
opts.weightTV         = 1;%prior.reg_param;
opts.gammaTV          = prior.reg_param;
opts.gammaLp          = 0;
opts.weightLp         = 0;
opts.pNorm            = 1;
opts.qNorm            = 1;%q-value (between 1 and 2)
opts.alpha            = 0.01;
opts.beta             = 0.6;%?
opts.mu               = 1e-10;%the rounding value used to make a differentiable regularizer
opts.display          = 0;%Display output after each iteration

%
%temp_weight = rand(size(projection));
%temp_weight(temp_weight>0.0)=1;
%temp_weight(temp_weight<=0.0)=0;
%temp_weight = sqrt(temp_weight);

FBP =iradon(projection',angle_list,'hamming',0.08,Nr);

tic;
[recon,x0]=MBIRTV(projection.*weight,weight,zeros(size(FBP)),formodel,prior,opts);
toc;


recon_original_size = real(recon(formodel.Npad/2 - Nr/2:formodel.Npad/2 + Nr/2 -1 ,formodel.Npad/2 - Nr/2:formodel.Npad/2 + Nr/2 -1 ));
x0 = real(x0(formodel.Npad/2 - Nr/2:formodel.Npad/2 + Nr/2 -1 ,formodel.Npad/2 - Nr/2:formodel.Npad/2 + Nr/2 -1 ));
%imagesc(recon_original_size.');axis image;colormap(gray);colorbar;

recon_original_size = flipud(recon_original_size.')*Nr*Nr*pi*pi/2;%*Nr/(pi*pi);
x0 = flipud(x0.');

%rescale values for projector 
x0=x0.*(Nr/pi);
recon_original_size=recon_original_size;%*(Nr/(pi*pi));


figure;imagesc(grnd_truth);axis image;colormap(gray);colorbar;title('Ground truth');
%figure;imagesc(x0);axis image;colormap(gray);colorbar;title('NUFFT Back Proj');
figure;imagesc(FBP);axis image;colormap(gray);colorbar;title('FBP');
figure;imagesc(recon_original_size);axis image;colormap(gray);colorbar;title('MBIR TV');
line_idx = int16(0.8*Nr);
%figure;plot(grnd_truth(line_idx,:));hold on;plot(x0(line_idx,:),'-k');hold on;plot(recon_original_size(line_idx,:),'r');legend('Ground truth','NUFFT Back Proj','MBIR TV');title(strcat('Reg=',num2str(prior.reg_param)));
figure;plot(grnd_truth(line_idx,:),'g');hold on;plot(FBP(line_idx,:),'-k');hold on;plot(recon_original_size(line_idx,:),'r');legend('Ground truth','FBP','MBIR TV');title(strcat('Reg=',num2str(prior.reg_param)));
