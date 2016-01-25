function [gnuradon,gnuiradon,qtXqxy,qxyXqt,op,P,PT,gDq]=gnufft_initx(Ns,qq,tt,beta,k_r,uniqueness)
% function [gnuradon,gnuiradon,qtXqxy,qxyXqt]=gnufft_init(Ns,q,t,beta,k_r)
% 
% returns radon  and inverse radon trasnform operators (GPU acceler ated)
% 
% input: Ns, x-y grid size 
%        q,t (polar coordinates q, theta)
%        beta, kaiser-bessel parameter, 
%        k_r, kernel width
% Output:
%        radon and inverse radon operators, geometry is fixed and embedded
%        also gridding and inverse gridding operators, with fixed geometry
%        radon and inverse radon wrap around some FFTs 
%
%
%
% Stefano Marchesini,  LBNL 2013


%%
% Preload the Bessel kernel (real components!)
[kblut,KB,KB1,KB2D]=KBlut(k_r,beta,256);
scale = single((256-1)/k_r);

grid = [Ns,Ns];
scaling=1;
% % Normalization (density compensation factor)
Dq=KBdensity1(qq'.*scaling,tt',KB,k_r,Ns)';
gDq=gpuArray(single(Dq));
grmask=gpuArray(abs(qq)<size(qq,1)/4*3/2);

%gDq=gpuArray(1./(abs(qq)+1));

% anti-aliased deapodization factor, (the FT of the kernel, cropped):
dpz=deapodization(Ns,KB);
gdpz=gpuArray(single(dpz));

% normalize by KB factor
cnorm=gpuArray(single(sum(sum(KB2D((1:Ns)'-Ns/2,(1:Ns)-Ns/2)))));

 
% polar to cartesian to matrix 
[yi,xi]=pol2cart(tt*pi/180,scaling.*qq);
xi = xi+floor((Ns+1)/2)+1;
yi = yi+floor((Ns+1)/2)+1;


%  q-radon <-- q-cartesian 
gkblut=gpuArray(single(kblut));
gxi=gpuArray(single(xi));
gyi=gpuArray(single(yi));

qtXqxy=@(Gqxy) polarsample(gxi,gyi,Gqxy,grid,gkblut,scale,k_r);


nangles=size(tt,1);

%
xint=int32(xi);
yint=int32(yi);

xf=-(xi-double(xint)); %fractional shift
yf=-(yi-double(yint)); %fractional shift

% matrix from non-uniform samples to grid
nrow=prod(grid);
ncol=numel(xi);

% stencil vectors
nkr=2*k_r+1;
kstencil=int32(gpuArray.linspace(-k_r,k_r,nkr));
kkrx=reshape(kstencil,[ 1 1 2*k_r+1 1]);
kkry=reshape(kstencil,[ 1 1 1 2*k_r+1]);

%replicate over -k_r:k_r
xii=int32(bsxfun(@plus,xint,kkrx));
yii=int32(bsxfun(@plus,yint,kkry));
%valii=bsxfun(@times,KB1((-k_r:k_r)),KB((-k_r:k_r),0)');
%
% pre-compute kernel value for every shift
gval=single(bsxfun(@times,KB1(bsxfun(@plus,xf,double(kkrx))),KB1(bsxfun(@plus,yf,double(kkry)))));

% index of where every point lands  on the image
grow=bsxfun(@plus,(xii-1)*Ns,yii); %index of where every point lands
%index of where every point comes from
gcol=repmat(int32(reshape(1:numel(qq),size(qq))),[ 1 1 nkr nkr]);
%
% remove stencils if they land outside the frame
iin=find(bsxfun(@and,(xii>0) & ( xii<Ns+1) , (yii>0) & (yii<Ns+1)));
if numel(iin)<numel(gval);
 grow=grow(iin);
 gcol=gcol(iin);
 gval=gval(iin);
end
% %%

%get the array and transpose
P=gcsparse(gcol,grow,complex(gval),nrow,ncol,1);
PT=gcsparse(grow,gcol,complex(gval),ncol,nrow,1); 
%%

%grmask=gpuArray(abs(qq)<Ns/6);

% (qx,qy) <-> (q, theta) : cartesian <-> non-uniform samples
qxyXqtn =@(Gqt) reshape(P*(Gqt(:)./gDq(:)),[Ns Ns]);

qxyXqt =@(Gqt) qxyXqtn(Gqt)/cnorm;

xx=gpuArray(single(0:Ns-1));
f2shift=bsxfun(@times,(-1).^xx,(-1).^xx'); %fftshift factor


%  q-cartesian <-- q-radon  
%qxyXqt =@(Gqt) reshape(P*(Gqt(:)./gDq(:)),[Ns Ns]);
   
    
% fftshift factors
xx=gpuArray(single(0:Ns-1));
%f2shift=(-1).^(xx+yy); %fftshift factor
f2shift=bsxfun(@times,(-1).^xx,(-1).^xx'); %fftshift factor
nangles=size(tt,2);
fftshift1D=(-1).^xx';

% real (r) to fourier (q) -- cartesian (xy)
qxyXrxy=@(Grxy) (f2shift.*fft2(Grxy.*(gdpz).*f2shift));%deapodized

rxyXqxy=@(Gqxy) f2shift.*ifft2((Gqxy.*f2shift)).*(gdpz); %deapodized


% real (r) to fourier (q) -- radon space (r/q-theta)
rtXqt=@(Gqt) ifftshift(bsxfun(@times,fftshift1D,ifft(Gqt)),1).*grmask;
  qtXrt=@(Grt) ifftshift( bsxfun(@times,fft(Grt),fftshift1D),1);

% radon transform: (x y) to (qx qy) to (q theta) to (r theta):
gnuradon=@(G) rtXqt(qtXqxy(qxyXrxy(G)));
% inverse radon transform: (r theta) to (q theta) to (qx qy) to (x y)
gnuiradon=@(GI) rxyXqxy(qxyXqt(qtXrt(GI)));


op = @(x,mode) opRadon_intrnl(x,mode);

function y =opRadon_intrnl(x,mode)
checkDimensions(nangles*Ns,Ns*Ns,x(:),mode);
if mode == 0
   y = {nangles*Ns,Ns*Ns,[1,1,1,1],{'GNURADON'}};
   elseif mode == 1
       y=gnuradon(reshape(x,grid));
       y=y(:);
else
      y=gnuiradon(reshape(x,[Ns nangles]));
      y=y(:);
end

end

end


           
