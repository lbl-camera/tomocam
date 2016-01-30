function [recon]=MBIRTV(projection,weight,init,forward_model,prior_model,opts)
%Function to call the TV MBIR code 
%Input : projection : A 2-D array with num_angles X n_det entries
%containing the projection data
%        weight : A 2-D array with num_angles X n_det entries
%containing the noise weight data
%        init : An initial image reconstruction 
%        forward_model : A structure having the following entries :
%                      center : Center of rotation in units of pixels from
%                      the left end of the detector 
%                      pix_size : pixel size in um 
%                      det_size : detector pixel size in um 
%         prior_model : A structure having the following entries :
%                        reg_value  :  Value of the regularization constant
%       angles : a list of angles used (1 X num_angles array)
%
g=gpuDevice(3);
reset(g);
%gpuDevice(2);



% Set solver parameters


%Code starts here
[nangle,Ns]=size(projection');
Dt=180/nangle;

[tt,qq]=meshgrid(0:Dt:180-Dt,(1:(Ns))-floor((Ns+1)/2)-1);
[~,~,P,opGNUFFT]=gnufft_init_spmv_op_v2(Ns,qq,tt,forward_model.beta,forward_model.k_r,forward_model.center,forward_model.pix_size,forward_model.det_size);
opFPolyfilter = opFPolyfit(nangle,Ns,P.opprefilter);

data.signal = gpuArray(init);
data.M=opFoG(opGNUFFT);
data.M=opFoG(opFPolyfilter,opGNUFFT);
real_data = gpuArray(projection);
data.b=P.opprefilter(real_data(:),2);
data=completeOps(data);
TV = opDifference(data.signalSize);
x0=data.reconstruct(data.M(data.b,2));
x=x0(:);
%msk1=padmat(ones(Ns/2),[1 1]*Ns);
%x=x.*msk1(:);
x = solveTV(data.M, data.B, TV, data.b, x, opts);
recon = data.reconstruct(x);