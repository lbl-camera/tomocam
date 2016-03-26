%Script to test the NUFFT based forward and back-projection operators 

clear;
close all;

gpuDevice([]);
gpuDevice(2);

addpath operators
addpath gpu
addpath gnufft
addpath Common

%% Create a toy data set 

Ns_actual = 256;

nangles = 180;%180;
%Ns_pad = 4096;
center_actual = 128;%128;%sub pixels 
pix_size = 1;%um 
det_size = 1;%um 

%padding
Ns=512;
center = center_actual + (Ns/2 - Ns_actual/2);

signal = gpuArray(padmat(phantom(Ns_actual),[Ns,Ns]));


Dt=(180/nangles); %spacing in degrees
angle_list= 0:Dt:180-Dt;
[tt,qq]=meshgrid(angle_list,(1:(Ns))-floor((Ns+1)/2)-1);


%% Initialize the forward and back projectors. 
%Parameters associated with NUFFT - there are more "hidden" into the gnufft_init file

k_r=3;beta =2*pi*2;  %kernel size 2*kr+1
[Ns,nangles]=size(qq);
%[~,~,A,~]=gnufft_init_spmv_op_v2(Ns,qq,tt,beta,k_r,center,ones(size(qq)),pix_size,pix_size,Ns_actual);
[gnuqradon,gnuqiradon,P,opGNUFFT]=gnufft_init_op(Ns,qq,tt,beta,k_r,0);


%% Test the operators 

%%%%%%%%%% Forward-projection %%%%%% 
display('Projecting using NUFFT');
tic;
real_data=Ns.*pi/2.*A.gnuradon(signal);
toc;
%input_data=preprocessop.radon2q(real_data);

display('Projecting using Matlab gpu radon');
tic;
forward_proj_inbuilt = radon((rot90(signal)),angle_list);
forward_proj_inbuilt = forward_proj_inbuilt((end+1)/2-Ns/2:(end+1)/2+Ns/2-1,:);
toc;

figure;imagesc(signal);axis image;colorbar;title('Ground truth');
figure;imagesc(real(real_data).');title('Projection using NUFFT');
colorbar;

figure;imagesc(real(forward_proj_inbuilt).');
title('Projection using Matlab radon');
colorbar;

figure;plot(real(real_data(:,end/2)));
hold on;plot(real(forward_proj_inbuilt(:,end/2)),'r')
title('Projection at  angle');
legend('NUFFT proj.','Matlab radon');

%%%%%%%% Back-projection %%%%%%%%%
display('Back-Projecting using NUFFT');
tic;
test_backproj = A.gnuiradon(real_data);
toc;

display('Projecting using Matlab iradon');
tic;
back_proj_inbuilt=iradon(forward_proj_inbuilt,angle_list,Ns);
toc;

figure;imagesc(real(test_backproj));axis image;colorbar;
title('Back projection using NUFFT');

figure;imagesc(rot90(back_proj_inbuilt,-1));axis image;colorbar;
title('Back projection using iradon');

