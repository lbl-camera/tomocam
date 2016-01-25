clear;
close all;

gpuDevice([]);
gpuDevice(2);

addpath operators
addpath gpu
addpath gnufft
addpath Common

Ns=2560;
nangles=512;

signal = padmat(phantom(Ns/2),[Ns,Ns]);
%padmat(generateAngiogram(Ns/2,Ns/2),[Ns,Ns]);

 Dt=(180/nangles); %spacing in degrees
 angle_list= 0:Dt:180-Dt;
[tt,qq]=meshgrid(angle_list,(1:(Ns))-floor((Ns+1)/2)-1);

 [A,preprocessop]=forwarmodel(qq,tt);

%%
% note that real data is 
real_data=preprocessop.image2radon(signal);
input_data=preprocessop.radon2q(real_data);


%debugging 

figure;imagesc(signal);axis image;colorbar;title('Ground truth');

figure;imagesc(real(real_data).');title('Projection using NUFFT');
colorbar;

forward_proj_inbuilt = radon(gpuArray(rot90(signal)),angle_list);
forward_proj_inbuilt = forward_proj_inbuilt((end+1)/2-Ns/2:(end+1)/2+Ns/2-1,:);
figure;imagesc(real(forward_proj_inbuilt).');
title('Projection using Matlab radon');
colorbar;


%P.gnuradon(reshape(data.signal,[Ns,Ns]));
%data.b=P.opprefilter(real_data(:),2);

%data.signalSize=[Ns Ns];
%data = completeOps(data);

%TV = opDifference(data.signalSize);

  
%%  
x0=data.reconstruct(A.M(input_data,2));

x=x0(:);
msk1=padmat(ones(Ns*3/4),[1 1]*Ns);
x=x.*msk1(:);
subplot(1,2,1);
cropimg=@(img) img(Ns/4+(1:Ns/2),Ns/4+(1:Ns/2));
imagesc(cropimg((abs(x0)+.1).^.5)); axis image
tic;
for i=1:1
    x = solveTV(data.M, data.B, TV, data.b, x, opts);
    y = data.reconstruct(x);
    tm=toc/i;
    subplot(1,2,2);
    imagesc((abs(cropimg(y))+.1).^.5);axis image
    title(sprintf('Iteration %d, timeperiter=%g',i*opts.maxIts,tm));
    drawnow;
    pause
end
%%
 ttime=toc;
 fprintf('total time=%g\n',ttime);