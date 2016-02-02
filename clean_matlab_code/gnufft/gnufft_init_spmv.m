function [gnuradon,gnuiradon,qtXqxy,qxyXqt]=gnufft_init_spmv(Ns,qq,tt,beta,k_r)
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
% Stefano Marchesini,  LBNL 2013

%%
% Preload the Bessel kernel (real components!)
[kblut,KB,KB1,KB2D]=KBlut(k_r,beta,256);

scaling=1;
% % Normalization (density compensation factor)
Dq=KBdensity1(qq'.*scaling,tt',KB,k_r,Ns)';
gDq=gpuArray(single(Dq));

%gDq=gpuArray(1./(abs(qq)+1));

% anti-aliased deapodization factor, (the FT of the kernel, cropped):
dpz=deapodization(Ns,KB);
gdpz=gpuArray(single(dpz));

% normalize by KB factor
cnorm=gpuArray(single(sum(sum(KB2D((1:Ns)'-Ns/2,(1:Ns)-Ns/2)))));

 
% polar to cartesian
[yi,xi]=pol2cart(tt*pi/180,scaling.*qq);

grid = [Ns,Ns];
nangles=size(tt,1);

xi = xi+floor((Ns+1)/2)+1;
yi = yi+floor((Ns+1)/2)+1;

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

% qxyXqt =@(Gqt) reshape(P*(Gqt(:)),[Ns Ns]);
qtXqxy=@(Gqxy) reshape(PT*Gqxy(:),size(qq));
% (qx,qy) <-> (x, y) : 2D 
qxyXrxy=@(Grxy) f2shift.*(fft2(Grxy.*gdpz.*f2shift));%deapodized
%rxyXqxy=@(Gqxy) ifftshift(ifft2(fftshift((Gqxy)))).*gdpz; %deapodized
rxyXqxy=@(Gqxy) f2shift.*(ifft2(f2shift.*((Gqxy)))).*gdpz; %deapodized

% (r,theta) <-> (q,theta) :  radon <-> Fourier slice
rtXqt=@(Gqt) ifftshift(ifft(fftshift(Gqt,1)),1);
qtXrt=@(Grt) ifftshift( fft(fftshift(Grt,1)),1);

% radon transform: (x y) to (qx qy) to (q theta) to (r theta):
gnuradon=@(G) rtXqt(qtXqxy(qxyXrxy(G)));
% inverse radon transform: (r theta) to (q theta) to (qx qy) to (x y)
gnuiradon=@(GI) rxyXqxy(qxyXqt(qtXrt(GI)));
 
return
end
%%
% % 
% % % first sort by row (where it lands)
% % [row_sort,uns2s]=sort(row(:));
% % col_sort=coli(uns2s); %reorder rows
% % valm_sort=valm(uns2s); %reorder values
% % 
% % % now turn it into compact CSR
% % ptr=[0;(find(diff(row_sort)));numel(row)]; % pointer for CSR
% % rowu=row_sort(ptr(2:end)); %output row value
% %  growu=gpuArray(int32(rowu));
% % S.type = '()'; S.subs = {gpuArray(int32(growu))};
% % unique2all=@(gg,gx) subsasgn(gg,S,gx); 
% 
% 
% Gq=gpuArray(single(G(1:size(qq,1),2000+(1:size(qq,2)))));
% 
% gnugrid=@(x) P*x; 
% 
% %guns2s=gpuArray(int32(uns2s));
% %gu2unsort=gpuArray(int32(u2unsort));
% 
% % % push to gpu
% % gptr=gpuArray(int32(ptr));
% % gcol=gpuArray(int32(col_sort-1));
% % gval=gpuArray(single(valm_sort));
% % 
% % % CSR will return only the rows that have values
% % growu=gpuArray(int32(rowu));
% % S.type = '()'; S.subs = {gpuArray(int32(growu))};
% % unique2all=@(gg,gx) subsasgn(gg,S,gx); 
% 
% %
% % gnugrid=@(gg,gx) subsasgn(gg,S,gspmv(gval,gcol,gptr,gx)); 
% % gnugrid=@(gg,gx) unique2all(gg,gspmv(gval,gcol,gptr,gx)); 
% %%
% % %transpose
% % [col_sort,uns2s]=sort(coli(:));
% % row_sort=row(uns2s); %reorder rows
% % valm_csort=valm(uns2s); %reorder values
% % 
% % % now turn it into CSR
% % cptr=[0;(find(diff(col_sort)));numel(row)]; % pointer for CSR
% % colu=row_sort(cptr(2:end)); %output row value
% % 
% % % push to gpu
% % gcptr=gpuArray(int32(ptr));
% % gccol=gpuArray(int32(col_sort-1));
% % gcval=gpuArray(single(valm_csort));
% % gcolu=gpuArray(int32(colu));
% % 
% % Sc.type = '()'; Sc.subs = {gpuArray(int32(gcolu))};
% % col_unique2all=@(gg,gx) subsasgn(gg,Sc,gx); 
% % gnucgrid=@(gg,gx) col_unique2all(gg,gspmv(gcval,gccol,gcptr,gx)); 
% % 
% % 
% 
% %gg=gy(
% 
% %gx=gpuArray(single(x));
% 
% 
% 
% % valii=bsxfun(@times,KB1(dxi+(-k_r:k_r)),KB1(dyi+(-k_r:k_r),0)');
% % ff=@(dxi,dyi) bsxfun(@times,KB1(xf+(-k_r:k_r)),KB1(dyi+(-k_r:k_r))');
% 
% 
% 
% % for each non uniform point, we average over 
% 
% nsample=numel(qq);
% ngrid=Ns*Ns;
% 
% % tiling, geometry:
% [s_per_b,b_dim_x,b_dim_y,s_in_bin,b_offset,b_loc,b_points_x,b_points_y] = polarbin1(xi,yi,grid,4096*4,k_r);
% 
% %push to GPU
% gxi=gpuArray(single(xi));
% gyi=gpuArray(single(yi));
% gkblut=gpuArray(single(kblut));
% %dpz=gsingle(1);
% 
% gs_per_b=gpuArray(int64(s_per_b));
% gs_in_bin=gpuArray(int64(s_in_bin));
% gb_dim_x= gpuArray(int64(b_dim_x));
% gb_dim_y= gpuArray(int64(b_dim_y));
% gb_offset=gpuArray(int64(b_offset));
% gb_loc=   gpuArray(int64(b_loc));
% gb_points_x=gpuArray(single(b_points_x));
% gb_points_y=gpuArray(single(b_points_y));
% gDq=gpuArray(single(Dq));
% gdpz=gpuArray(single(dpz));
% 
% grid = int64([Ns,Ns]);
% scale = single((256-1)/k_r);
% 
% % q-radon to q cartesian with density compensation (1/Dq)
% c=polargrid_cusp(gxi,gyi,(1+1i*eps)./(gDq),(grid),gs_per_b, gb_dim_x, gb_dim_y, gs_in_bin, gb_offset, gb_loc,b_points_x,gb_points_y,gkblut,scale);
% 
% gDq=gDq*norm(dpz(:))/norm(c(:));
%  
% 
% qxyXqt =@(Gqt) polargrid_cusp(gxi,gyi,Gqt./gDq,grid,gs_per_b,...
%     gb_dim_x, gb_dim_y, gs_in_bin, gb_offset, gb_loc,...
%     gb_points_x,gb_points_y,gkblut,scale);
%  
% % q-cartesian to q-radon
% qtXqxy=@(Gqxy) polarsample(gxi,gyi,Gqxy,grid,gkblut,scale,k_r);
% 
% % real (r) to fourier (q) -- cartesian (xy)
% qxyXrxy=@(Grxy) ifftshift(fft2(fftshift(Grxy.*gdpz)));%deapodized
% rxyXqxy=@(Gqxy) ifftshift(ifft2(fftshift((Gqxy)))).*gdpz; %deapodized
% %rxyXqxy=@(Gqxy) ifftshift(ifft2(fftshift((Gqxy)))); %deapodized
% 
% 
% % real (r) to fourier (q) -- radon space (r/q-theta)
% rtXqt=@(Gqt) ifftshift(ifft(fftshift(Gqt,1)),1);
% qtXrt=@(Grt) ifftshift( fft(fftshift(Grt,1)),1);
% 
% % radon transform: (x y) to (qx qy) to (q theta) to (r theta):
% gnuradon=@(G) rtXqt(qtXqxy(qxyXrxy(G)));
% % inverse radon transform: (r theta) to (q theta) to (qx qy) to (x y)
% gnuiradon=@(GI) rxyXqxy(qxyXqt(qtXrt(GI)));
%  
% 
% 
% op = @(x,mode) opRadon_intrnl(nsample,ngrid,gnuradon,gnuiradon,x,mode);
%  
% end


function y =opRadon_intrnl(m,n,gnuradon,gnuiradon,x,mode)
checkDimensions(m,n,x,mode);
if mode == 0
   y = {m,n,[1,1,1,1],{'RADON'}};
   elseif mode == 1
       y=gnuradon(x);
else
      y=gnuiradon(x);
end

end
           
