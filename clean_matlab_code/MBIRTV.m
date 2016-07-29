function [recon,x0]=MBIRTV(projection,weight,init,forward_model,prior_model,opts)
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
%                      k_r : The radius of the Kaiser-Bessel interpolator
%                      beta : the drop off value of the K.-B kernel
%                      Npad : The size of the up-sampled image
%                      ring_corr : A binary flag for ring correction , 0=>
%                      no correction and 1 => correction
%                      angle_list : An array of angles at whic hprojection
%                      is obtained
%         prior_model : A structure having the following entries :
%                        reg_value  :  Value of the regularization constant
%       angles : a list of angles used (1 X num_angles array)
%
%g=gpuDevice(2);
%reset(g);
%gpuDevice(2);


[Ntheta,Nr]=size(projection);
projection = (padmat(projection,[Ntheta forward_model.Npad]));
weight = (padmat(weight,[Ntheta forward_model.Npad],0));
init = (padmat(init,[forward_model.Npad, forward_model.Npad]));

forward_model.center = forward_model.center + (forward_model.Npad/2-Nr/2);


%Code starts here
[nangle,Ns]=size(projection);
%Dt=180/nangle;

[tt,qq]=meshgrid(forward_model.angle_list,(1:(Ns))-floor((Ns+1)/2)-1);
%[~,~,P,opGNUFFT]=gnufft_init_spmv_op_v2(Ns,qq,tt,forward_model.beta,forward_model.k_r,forward_model.center,weight,forward_model.pix_size,forward_model.det_size,Nr);
[~,~,P,opGNUFFT]=gnufft_init_op_v2(Ns,qq,tt,forward_model.beta,forward_model.k_r,forward_model.center,weight,forward_model.pix_size,forward_model.det_size,Nr);
opFPolyfilter = opFPolyfit(nangle,Ns);

data.signal = gpuArray(init);
data.M=opFoG(opGNUFFT);
if(forward_model.ring_corr)
    data.M=opFoG(opFPolyfilter,opGNUFFT);
end
real_data = gpuArray(projection.');
%data.b=P.opprefilter(real_data(:),2);
data.b=real_data(:);
data=completeOps(data);
TV = opDifference(data.signalSize);
x0=fliplr(init.');%init;
%x0=data.reconstruct(data.M(data.b,2));
x=x0(:);
%msk1=padmat(ones(Ns/2),[1 1]*Ns);
%x=x.*msk1(:);
x = solveOGM(data.M, data.B, TV, data.b, x, opts);
%solveTV(data.M, data.B, TV, data.b, x, opts);

recon = data.reconstruct(x);


recon=recon./forward_model.Npad;%Scaling the reconstruction assuming unit sized pixels
x0 = x0./forward_model.Npad;
