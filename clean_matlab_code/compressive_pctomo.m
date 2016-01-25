tooldir='~/matlab/toolboxes/';
%
%tooldir='/Users/smarchesini/matlab/toolboxes/';

addpath([tooldir,'spgl1-1.7/']);
addpath([tooldir,'sparco-1.2/operators/']);
addpath([tooldir,'sparco-1.2/tools/']);
irtdir=[tooldir 'irt/'];
addpath([irtdir,'nufft/']);
addpath([irtdir,'nufft/table']);
addpath([irtdir 'utilities']);
addpath([irtdir 'systems']);


%read sinogram
dnam_out='/data/ALS/tomography/balls2010/processed/';
sinogramd=[dnam_out 'sinograms1/'];

%get file names
[imgf, imnum]=getfilesn(dnam_out);
% center of rotation in the sinogram;
f0shift=[    0  -114];
f0shift=[    0  -74];

%get sinogram
iisection=500 %section #10

sg=getsinog( dnam_out,imgf,iisection)';
sg=circshift(sg,(f0shift));
%

imagesc([sg; fliplr(sg(2:end,:))]');

colormap hot
%remove 180 deg
% sg=sg(1:end-1,:);
%%
%Polynomial fit along each vertical stripe
ntheta=size(sg,1);
vander4=@(x) [x.^4 x.^3 x.^2 x.^1 x.^0]';
A=vander4(linspace(-1,1,ntheta)');
% fit polynomial and subtract
Pfilt4=eye(size(A,2))-A'*(A*A')^(-1)*A;

%calculate filtered sinogram
sinog0=sg;
sinog4=Pfilt4*sg;

%divide polynomial into blocks=background intervals (40)
yy1=1:size(sg,1);yy1=(yy1-1);
yy0=(yy1<40);
yy1=(yy1)/39.*yy0;
yy2=[yy0 ;yy1];
yy3=zeros(size(yy2,1)*720/40,size(yy2,2));
for ii=0:17;
    yy3(ii*2+(1:2),:)=circshift(yy2,[0,40*ii]);
end
A=yy3;

%this looks better
Pfilt_block=eye(size(A,2))-A'*(A*A')^(-1)*A;

%%
%matlab's inverse radon
% sinog=sg';
 sinog=(Pfilt4*sg)';
%sinog=(Pfilt_block*sg)';

theta=linspace(0,180,size(sinog,2));
tic
img1 = iradon(sinog, theta,'linear','Ram-Lak',1,size(sinog,1));
t2=toc;
fprintf('matlab iradon, timing=%g\n',t2);

imagesc(abs(img1).^.2)


%%
% nufft, this is the stuff we want on the GPU.
sinog=sinog;

%1D FFT sinogram
[nx,nt]=size(sinog);
ww2f=fftshift(fft(fftshift(sinog,1)),1)/sqrt(nx);

%set parameters for nufft
%%%%%%%%%%%%%%%%%%
% not sure what all the parameters are, but 'table' is for large matrices
%%%%%%%%%%%%%%%%%%%%

%set angles
[om ang] = ndgrid(((1:nx)-nx/2-1)/nx*2*pi, theta/180*pi); % [nr+1, na]
omega = [col(om.*cos(ang)) col(om.*sin(ang))];

%set gridding parameters
clear N;
N(1:2)=nx;
J = [3 3];
nufft_args = {N, J, 2*N, N/2, 'table', 2^10, 'minmax:kb'};
mask=true(N);
%     [xx,yy,rr]=make_grid(N,1);
%mask=rr<nx/2.5;
%     mask=true(size(
% Gn = Gnufft_n(mask, {omega, nufft_args{:}});
% Gn = nufft_init({omega, nufft_args{:}});
Gn = nufft_init(omega, nufft_args{:}); %nufft toolbox

%%
%nuFFT timing
tic;
% xcp=Gn'*(ww2f(:));

xcp=nufft_adj(ww2f(:),Gn);

t3=toc;
fprintf('nufft, timing=%g\n',t3);

img2=real(flipud(reshape(xcp,size(mask))'));

% img2=cropmat(real(flipud(xcp')),size(img1));

figure(2)
imagesc(abs(img2).^.2)
%  mask = true(N);
% tic
%%
%

[nx,nt]=size(sinog);

P.op.fft=opFFTC(nx,nt);
P.op.nufft=opFatrix(Gn);
info=P.op.nufft([],0);

% nufftdims=

P.op.real         = opReal(info{2});
% P.op.pfit=opPfit((A'*A)^(-1)*A');
% P.op.pfilt=opPfith((eye(nt)-A*(A'*A)^(-1)*A')'); %remove it all
P.op.pfilt=opPfith(Pfilt4,sinog); %remove it all


% P.op.pfit=opPfit(A*(A'*A)^(-1)*A'); %remove it all
% P.M=opFoG(P.op.real, P.op.nufft);
P.M=opFoG(P.op.pfilt,P.op.fft,P.op.nufft,P.op.real);

% P.op.f=
% 
% w=P.op.pfilt(P.op.fft(P.M(q,1),2),2);

%prepare data:
P.b=sinog(:);
% P.signal=P.M(ww1f,2);
P.signal=P.M(P.b,2);

% b0=P.M(ww1f,2);
P=completeOps(P);

%%
img3=flipud(reshape(P.signal,size(img1)))';

%
% imagesc(abs(ifft(reshape(P.b,nx,nt))));
% 
% q=P.op.real(P.op.nufft(P.b,2),2);
% q=P.M(P.b,2);
% imagesc(abs(reshape(q,nx,nx)))
% w=P.op.pfilt(P.op.fft(P.M(q,1),2),2);
% %sinogram 
% imagesc(abs(reshape(w,nx,nt)))

%%

opts = spgSetParms('optTol',1e-4,'iterations',10);
tau = 2000000;   % Initial one-norm of solution
sigma = 0; % set to 0 for a basis pursuit solution
%   z = spgl1(P.A, P.b, tau, sigma, [], opts);
[z, r, g, info] = spgl1(P.A, P.b, tau, sigma, [], opts);

% xcp1=zeros(size(mask));
% xcp1(mask)=P.reconstruct(z);
figure(10);
img3=flipud(reshape(z,size(img1)))';
imagesc(abs(img3).^.2);
colormap hot;

% figure(11);
% img4= img3 >5;
% imagesc(abs(img4).^.2);
% colormap hot;
% img3=cropmat(real(flipud(z')),size(img1));
