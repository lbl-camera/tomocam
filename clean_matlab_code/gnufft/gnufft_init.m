function [gnuradon,gnuiradon,qtXqxy,qxyXqt,op,P,PT,gDq]=gnufft_init(Ns,qq,tt,beta,k_r,uniqueness)
% function [gnuradon,gnuiradon,qtXqxy,qxyXqt]=gnufft_init(Ns,q,t,beta,k_r)
% 
% returns radon  and inverse radon trasnform operators (GPU accelerated)
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

% Preload the Bessel kernel (real components!)
if nargin<6
    uniqueness=false;
end

%[kblut,KB]=
[kblut,KB,KB1,KB2D]=KBlut(k_r,beta,256);
%KBnorm=gpuArray(single(sum(sum(KB2D((1:Ns)'-Ns/2,(1:Ns)-Ns/2)))));
KBnorm=gpuArray(single(sum(sum(KB2D((-k_r:k_r)',(-k_r:k_r))))));
kblut=kblut/KBnorm;

scaling=1;

% % Normalization (density compensation factor)
Dq=KBdensity1(qq'.*scaling,tt',KB,k_r,Ns)';%'
%Dq=1./(abs(qq)+1);

grmask=gpuArray(abs(qq)<size(qq,1)/4*3/2);

% deapodization factor, (the FT of the kernel):
dpz=deapodization(Ns,KB);
% polar to cartesian, centered
[xi,yi]=pol2cart(tt*pi/180,scaling.*qq);
xi = xi+floor((Ns+1)/2);
yi = yi+floor((Ns+1)/2);
grid = [Ns,Ns];

%%
%uniqueness
if uniqueness
[~,s2u,u2s]=unique([xi(:),yi(:)],'rows');
grow=u2s;
gcol=1:numel(u2s);
  
PT=gcsparse(grow,gcol,complex(single(gcol*0+1)),numel(u2s),numel(s2u),1); 
P=gcsparse(gcol,grow,complex(single(gcol*0+1)),numel(s2u),numel(u2s),1); 

%
xi=xi(s2u); 
yi=yi(s2u);
Dq=accumarray(u2s,Dq(:));
%gDq=real(P*single(complex(gDq)));
tred=numel(u2s)-numel(s2u)
end

% tiling, geometry:

if ~exist('polarbinmtx.mat','file');
 [s_per_b,b_dim_x,b_dim_y,s_in_bin,b_offset,b_loc,b_points_x,b_points_y] = polarbin1(xi,yi,grid,4096*4,k_r);
 save('polarbinmtx.mat','s_per_b','b_dim_x','b_dim_y','s_in_bin','b_offset','b_loc','b_points_x','b_points_y');
 else
   load('polarbinmtx.mat');
end



%push to GPU
gxi=gpuArray(single(xi));
gyi=gpuArray(single(yi));
gkblut=gpuArray(single(kblut));
gs_per_b=gpuArray(s_per_b);
gs_in_bin=gpuArray(s_in_bin);
gb_dim_x= gpuArray(b_dim_x);
gb_dim_y= gpuArray(b_dim_y);
gb_offset=gpuArray(b_offset);
gb_loc=   gpuArray(b_loc);

% gs_per_b=gpuArray(int64(s_per_b));
% gs_in_bin=gpuArray(int64(s_in_bin));
% gb_dim_x= gpuArray(int64(b_dim_x));
% gb_dim_y= gpuArray(int64(b_dim_y));
% gb_offset=gpuArray(int64(b_offset));
% gb_loc=   gpuArray(int64(b_loc));

gb_points_x=gpuArray(single(b_points_x));
gb_points_y=gpuArray(single(b_points_y));
gDq=gpuArray(single(Dq));
gdpz=gpuArray(single(dpz));

grid = int64([Ns,Ns]);
scale = single((256-1)/k_r);

if uniqueness
    % q-radon to q cartesian with density compensation (1/Dq)
    qxyXqt =@(Gqt) polargrid_cub((P*Gqt)./gDq,grid,gs_per_b,...
        gb_dim_x, gb_dim_y, gs_in_bin, gb_offset, gb_loc,...
        gb_points_x,gb_points_y,gkblut,scale);
    
    % q-cartesian to q-radon
    qtXqxy=@(Gqxy) reshape(PT*polarsample(gxi,gyi,Gqxy,grid,gkblut,scale,k_r),[Ns,nangles]);
else
    
    % q-radon to q cartesian with density compensation (1/Dq)
    qxyXqt =@(Gqt) polargrid_cub(Gqt./gDq,grid,gs_per_b,...
        gb_dim_x, gb_dim_y, gs_in_bin, gb_offset, gb_loc,...
        gb_points_x,gb_points_y,gkblut,scale);
    
    % q-cartesian to q-radon
    qtXqxy=@(Gqxy) polarsample(gxi,gyi,Gqxy,grid,gkblut,scale,k_r);
    
end
xx=gpuArray(single(0:Ns-1));
%f2shift=(-1).^(xx+yy); %fftshift factor
f2shift=bsxfun(@times,(-1).^xx,(-1).^xx'); %fftshift factor
nangles=size(tt,2);
fftshift1D=(-1).^xx';

% real (r) to fourier (q) -- cartesian (xy)
%qxyXrxy=@(Grxy) fftshift(fft2(ifftshift(Grxy.*(gdpz))));%deapodized
%qxyXrxy=@(Grxy) fftshift(f2shift.*fft2(Grxy.*(gdpz)));%deapodized
qxyXrxy=@(Grxy) (f2shift.*fft2(Grxy.*(gdpz).*f2shift));%deapodized

%rxyXqxy=@(Gqxy) ifftshift(ifft2(fftshift((Gqxy)))).*(gdpz); %deapodized
%rxyXqxy=@(Gqxy) ifftshift(f2shift.*ifft2((Gqxy))).*(gdpz); %deapodized
rxyXqxy=@(Gqxy) f2shift.*ifft2((Gqxy.*f2shift)).*(gdpz); %deapodized


% real (r) to fourier (q) -- radon space (r/q-theta)
%rtXqt=@(Gqt) fftshift(ifft(ifftshift(Gqt,1)),1);
%rtXqt=@(Gqt) ifftshift(ifft(fftshift(Gqt,1)),1).*grmask;
%rtXqt=@(Gqt) ifftshift(ifft(Gqt),1).*grmask;

rtXqt=@(Gqt) ifftshift(bsxfun(@times,fftshift1D,ifft(Gqt)),1).*grmask;
%rtXqt=@(Gqt) bsxfun(@times,fftshift1D,ifft(bsxfun(@times,fftshift1D,Gqt))).*grmask;


%qtXrt=@(Grt) ifftshift( fft(fftshift(Grt,1)),1);
%qtXrt=@(Grt) ifftshift( bsxfun(@times,fft(Grt),fftshift1D),1);
qtXrt=@(Grt) ifftshift( bsxfun(@times,fft(Grt),fftshift1D),1);
%qtXrt=@(Grt) bsxfun(@times,fft(bsxfun(@times,Grt,fftshift1D)),fftshift1D);

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


           
