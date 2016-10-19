clear;
close all;

reset(gpuDevice(2));
gpuDevice(2);

addpath operators
addpath gpu
addpath gnufft
addpath Common

num_slice = 10;
Ns_actual = 256;%2560;

nangles = 256;%2048;%Ns_pad = 4096;
center_actual = 128+10;%1280;%sub pixels 

pix_size = 1;%um 
det_size = 1;%um 

%padding
Ns=512;%3624
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
	test_backproj_ML(:,:,i) = iradon(squeeze(projection_ML(:,:,i)),angle_list,'Hamming',0.3,Ns_actual);
end
toc;
end

%% Debugging

[tt,qq]=meshgrid(angle_list,(1:(Ns))-floor((Ns+1)/2)-1);
k_r=3;
beta =4*pi*1;  %kernel size 2*kr+1
delta_r=1;
delta_xy=1;

%[~,~,P,opGNUFFT]=gnufft_init_op_v2(Ns,qq,tt,beta,k_r,center,ones(size(qq)),delta_r,delta_xy,Ns);
%[~,~,P,opGNUFFT]=gnufft_init_op_v2(Ns,qq,tt,beta,k_r,center,ones(size(qq)),delta_r,delta_xy,Ns);

[P]=gnufft_init_spmv_op_v3(Ns,qq,tt,beta,k_r,center,ones(size(qq)),delta_r,delta_xy,Ns);
bwidth=real((sum(P.kblut.^2.*(1:numel(P.kblut)))/sum(P.kblut.^2)/numel(P.kblut))*16);%16 ?


%[Ps]=gnufft_init_spmv_op_v3(Ns,qq,tt,beta,k_r,center,ones(size(qq)),delta_r,delta_xy,Ns);
%[~,~,Ps,opGNUFFT]=gnufft_init_spmv_op_v2(Ns,qq,tt,beta,k_r,center,ones(size(qq)),delta_r,delta_xy,Ns);

%[A,P]=forwarmodel_v2(qq,tt,center,pix_size,det_size);

%%%%%%%%%% Forward-projection %%%%%% 
display('Projecting using NUFFT');
projection = gpuArray(zeros(Ns,nangles,num_slice));
% run once to warm up
i=1; projection(:,:,i)=P.gnuradon(signal(:,:,i));    
% now time it
tic;
for i=1:num_slice
    %projection(:,:,i)=(Ns.*pi/2).*P.image2radon(squeeze(signal(:,:,i)));
     projection(:,:,i)=P.gnuradon(signal(:,:,i));
end
t_gnuradon=toc;

%%%%%%%% Back-projection %%%%%%%%%
display('Back-Projecting using NUFFT');
test_backproj=gpuArray(zeros(Ns,Ns,num_slice));
% run once to warm up
i=1;test_backproj(:,:,i) = P.gnuiradon(projection(:,:,i));
% now time it
tic;

for i=1:num_slice
%	test_backproj(:,:,i) = (Ns_actual)*(pi/2)*P.gnuiradon(projection(:,:,i));
    test_backproj(:,:,i) = P.gnuiradon(projection(:,:,i));
end
t_gnuiradon=toc;
%test_backproj=test_backproj/bwidth;

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
ss_gnuradon=sprintf('NUFFT projection, angles=%d, Ns=%d, nslices=%d, time=%g',nangles,Ns,num_slice,t_gnuradon);
title(ss_gnuradon);
fprintf([ss_gnuradon '\n']);


figure;
%imagesc(real(squeeze(test_backproj(:,:,1))));colormap(gray);colorbar;
imagesc(abs(squeeze(test_backproj(:,:,1))));cax=caxis;caxis([0 cax(2)]);colormap(gray);colorbar;
ss_gnuiradon=sprintf('NUFFT back-projection, angles=%d, Ns=%d, nslices=%d, time=%g',nangles,Ns,num_slice,t_gnuiradon);
title(ss_gnuiradon);
fprintf([ss_gnuiradon '\n']);


col=@(x) x(:); 
normratio=norm(col(test_backproj(:,:,1)))/norm(col(signal(:,:,1)));
cratio=sum(col(conj(test_backproj(:,:,1)).*signal(:,:,1)))./sum(col(abs(signal(:,:,1).^2)));

% testing the 0 frequency only...:
normratio1=sum(col(test_backproj(:,:,1)))/sum(col(signal(:,:,1)));
ss_ratios=sprintf('x0, x1=iradon radon x0, \n sum(x1)/sum(x0)=%g, (x1` * x0)/||x0||^2=%g , ||x1||/||x0||=%g\n',normratio1,cratio,normratio);
fprintf(ss_ratios)

% 1/2/(sum(P.kblut.*(1:numel(P.kblut)))/sum(P.kblut)/numel(P.kblut))
% comparing with width..

 

