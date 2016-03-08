clear;
close all;

gpuDevice([]);
%gpuDevice(2);

addpath operators
addpath gpu
addpath gnufft
addpath Common

num_slice = 64;
Ns_actual = 2560;

nangles = 512;
%Ns_pad = 4096;
center_actual = 1280;%sub pixels 
pix_size = 1;%um 
det_size = 1;%um 

%padding
Ns=3000;
center = center_actual + (Ns/2 - Ns_actual/2);

signal = gpuArray(repmat((padmat(phantom(Ns_actual),[Ns,Ns])),1,1,num_slice));
%phantom(Ns);
%padmat(generateAngiogram(Ns/2,Ns/2),[Ns,Ns]);

Dt=(180/nangles); %spacing in degrees
angle_list= 0:Dt:180-Dt;
[tt,qq]=meshgrid(angle_list,(1:(Ns))-floor((Ns+1)/2)-1);

[A,preprocessop]=forwarmodel_v2(qq,tt,center,pix_size,det_size);


%% Debugging
%%%%%%%%%% Forward-projection %%%%%% 
display('Projecting using NUFFT');
projection = gpuArray(zeros(Ns,nangles,num_slice));
tic;
for i=1:num_slice
    projection(:,:,i)=Ns.*pi/2.*preprocessop.image2radon(squeeze(signal(:,:,i)));
end
toc;

%%%%%%%% Back-projection %%%%%%%%%
display('Back-Projecting using NUFFT');
test_backproj=gpuArray(zeros(Ns,Ns,num_slice));
tic;
for i=1:num_slice
    test_backproj(:,:,i) = preprocessop.radon2image(squeeze(projection(:,:,i)));
end
toc;
