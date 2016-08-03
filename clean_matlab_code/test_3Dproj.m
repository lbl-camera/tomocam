clear;
close all;

reset(gpuDevice(2));
gpuDevice(2);

addpath operators
addpath gpu
addpath gnufft
addpath Common

num_slice = 5;
Ns_actual = 2560;

nangles = 512;
%Ns_pad = 4096;
center_actual = 1280;%sub pixels 
pix_size = 1;%um 
det_size = 1;%um 

%padding
Ns=3624;
center = center_actual + (Ns/2 - Ns_actual/2);

%temp=zeros(Ns_actual,Ns_actual);
%temp(end/4-5:end/4+5,end/2-5:end/2+5)=1;
signal =gpuArray(repmat((padmat(phantom(Ns_actual),[Ns,Ns])),1,1,num_slice));
%gpuArray(repmat((padmat(temp,[Ns,Ns])),1,1,num_slice));

Dt=(180/nangles); %spacing in degrees
angle_list= 0:Dt:180-Dt;

%% Matlab

%temp=zeros(Ns_actual,Ns_actual);
%temp(end/4-5:end/4+5,end/2-5:end/2+5)=1;
signal_ML = gpuArray(repmat(phantom(Ns_actual),1,1,num_slice));
%gpuArray(repmat(temp,1,1,num_slice));


%%%%%%%%%% Forward-projection %%%%%% 
if false
display('Projecting using Matlab');
projection_ML = gpuArray(zeros(Ns+1,nangles,num_slice));
tic;
for i=1:num_slice
    projection_ML(:,:,i)=radon(squeeze(signal_ML(:,:,i)),angle_list);
end
toc;

%%%%%%%% Back-projection %%%%%%%%%
display('Back-Projecting using Matlab');
test_backproj_ML=gpuArray(zeros(Ns_actual,Ns_actual,num_slice));
tic;
for i=1:num_slice
    test_backproj_ML(:,:,i) = iradon(squeeze(projection_ML(:,:,i)),angle_list,'none',Ns_actual);
end
toc;
end

%% Debugging

[tt,qq]=meshgrid(angle_list,(1:(Ns))-floor((Ns+1)/2)-1);
k_r=3;beta =4*pi;  %kernel size 2*kr+1
delta_r=1;
delta_xy=1;

%[~,~,P,opGNUFFT]=gnufft_init_op_v2(Ns,qq,tt,beta,k_r,center,ones(size(qq)),delta_r,delta_xy,Ns);
%[~,~,P,opGNUFFT]=gnufft_init_op_v2(Ns,qq,tt,beta,k_r,center,ones(size(qq)),delta_r,delta_xy,Ns);

[~,~,P,opGNUFFT]=gnufft_init_spmv_op_v3(Ns,qq,tt,beta,k_r,center,ones(size(qq)),delta_r,delta_xy,Ns);
[~,~,Ps,opGNUFFT]=gnufft_init_spmv_op_v3(Ns,qq,tt,beta,k_r,center,ones(size(qq)),delta_r,delta_xy,Ns);
%[~,~,Ps,opGNUFFT]=gnufft_init_spmv_op_v2(Ns,qq,tt,beta,k_r,center,ones(size(qq)),delta_r,delta_xy,Ns);

%[A,P]=forwarmodel_v2(qq,tt,center,pix_size,det_size);

%%%%%%%%%% Forward-projection %%%%%% 
display('Projecting using NUFFT');
projection = gpuArray(zeros(Ns,nangles,num_slice));
tic;
for i=1:num_slice
    %projection(:,:,i)=(Ns.*pi/2).*P.image2radon(squeeze(signal(:,:,i)));
    projection(:,:,i)=(Ns.*pi/2).*P.gnuradon(signal(:,:,i));
end
toc;

%%%%%%%% Back-projection %%%%%%%%%
display('Back-Projecting using NUFFT');
test_backproj=gpuArray(zeros(Ns,Ns,num_slice));
tic;
for i=1:num_slice
    test_backproj(:,:,i) = P.gnuiradon(projection(:,:,i));
end
toc;

%% plot and comparison

if false
figure;
imagesc(real(squeeze(projection_ML(:,:,1))));colormap(gray);colorbar;
title('Matlab projection');
figure;
imagesc(real(squeeze(test_backproj_ML(:,:,1))));colormap(gray);colorbar;
title('Matlab back-projection');

end

figure;
imagesc(real(squeeze(projection(:,:,1))));colormap(gray);colorbar;
title('NUFFT projection');


figure;
%imagesc(real(squeeze(test_backproj(:,:,1))));colormap(gray);colorbar;
imagesc(abs(squeeze(test_backproj(:,:,1))));cax=caxis;caxis([0 cax(2)]);colormap(gray);colorbar;
title('NUFFT back-projection');
