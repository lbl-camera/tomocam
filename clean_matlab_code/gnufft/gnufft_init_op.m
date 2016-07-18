function [gnuqradon,gnuqiradon,P,op,opprefilter]=gnufft_init_op(Ns,qq,tt,beta,k_r,uniqueness)
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

if nargin<6
    uniqueness=false;
end


%fftshift factor
xx=gpuArray(single(0:Ns-1));
fftshift2D=bsxfun(@times,(-1).^xx,(-1).^xx');
%fftshift1D=(-1).^xx';
fftshift1Dop=@(a) bsxfun(@times,(-1).^xx',a);

% Preload the Bessel kernel (real components!)
[kblut,KB,~,KB2D]=KBlut(k_r,beta,256);
KBnorm=gpuArray(single(sum(sum(KB2D((-k_r:k_r)',(-k_r:k_r))))));
kblut=kblut/KBnorm;

% % Normalization (density compensation factor)
Dq=KBdensity1(qq',tt',KB,k_r,Ns)';
% mask
P.grmask=gpuArray(abs(qq)<size(qq,1)/4*3/2);

% deapodization factor, (the FT of the kernel):
dpz=deapodization(Ns,KB);
% polar to cartesian, centered
scaling=1;
[xi,yi]=pol2cart(tt*pi/180,scaling.*qq);
xi = xi+floor((Ns+1)/2);
yi = yi+floor((Ns+1)/2);
grid = [Ns,Ns];

% sum points before stencils
if uniqueness
    [~,s2u,u2s]=unique([xi(:),yi(:)],'rows');
    grow=u2s;
    gcol=1:numel(u2s);
    
    PPT=gcsparse(grow,gcol,complex(single(gcol*0+1)),numel(u2s),numel(s2u),1);
    PP=gcsparse(gcol,grow,complex(single(gcol*0+1)),numel(s2u),numel(u2s),1);
    
    %
    xi=xi(s2u);
    yi=yi(s2u);
    Dq=accumarray(u2s,Dq(:));
    %gDq=real(P*single(complex(gDq)));
    tred=numel(u2s)-numel(s2u)
end


% tiling, geometry:
[s_per_b,b_dim_x,b_dim_y,s_in_bin,b_offset,b_loc,b_points_x,b_points_y] = polarbin1(xi,yi,grid,4096*4,k_r);

%push to GPU
gxi=gpuArray(single(xi));
gyi=gpuArray(single(yi));
gkblut=gpuArray(single(kblut));

gs_per_b=gpuArray(int64(s_per_b));
gs_in_bin=gpuArray(int64(s_in_bin));
gb_dim_x= gpuArray(int64(b_dim_x));
gb_dim_y= gpuArray(int64(b_dim_y));
gb_offset=gpuArray(int64(b_offset));
gb_loc=   gpuArray(int64(b_loc));
gb_points_x=gpuArray(single(b_points_x));
gb_points_y=gpuArray(single(b_points_y));
P.gDq=gpuArray(single(Dq));
P.gdpz=gpuArray(single(dpz));

grid = int64([Ns,Ns]);
scale = single((256-1)/k_r);


% real (r) to fourier (q) -- cartesian (xy)
P.qxyXrxy=@(Grxy) (fftshift2D.*fft2(Grxy.*(P.gdpz).*fftshift2D))/Ns;%deapodized
P.rxyXqxy=@(Gqxy) fftshift2D.*ifft2((Gqxy.*fftshift2D)).*(P.gdpz)*Ns; %deapodized

% real (r) to fourier (q) -- radon space (r/q-theta)
%P.rtXqt=@(Gqt) ifftshift(bsxfun(@times,fftshift1D,ifft(Gqt)),1).*P.grmask;
%P.qtXrt=@(Grt) ifftshift( bsxfun(@times,fftshift1D,fft(Grt)),1);
%P.rtXqt=@(Gqt) fftshift1Dop(ifft(fftshift1Dop(Gqt))).*P.grmask;
P.rtXqt=@(Gqt) fftshift1Dop(ifft(fftshift1Dop(Gqt))).*P.grmask;

P.qtXrt=@(Grt) fftshift1Dop(fft(fftshift1Dop(Grt)));


if uniqueness
    % q-radon to q cartesian with density compensation (1/Dq)
    
    P.qxyXqt =@(Gqt) polargrid_cusp(gxi,gyi,(PP*Gqt)./gDq,grid,gs_per_b,...
        gb_dim_x, gb_dim_y, gs_in_bin, gb_offset, gb_loc,...
        gb_points_x,gb_points_y,gkblut,scale);
    
    % q-cartesian to q-radon
    P.qtXqxy=@(Gqxy) reshape(PPT*polarsample(gxi,gyi,Gqxy,grid,gkblut,scale,k_r),[Ns,nangles]);
else
    
    % q-cartesian to q-radon
    P.qtXqxy=@(Gqxy) polarsample(gxi,gyi,Gqxy,grid,gkblut,scale,k_r);
    %GDq;
    
    % q-radon to q cartesian
%    P.qxyXqt =@(Gqt) polargrid_cusp(gxi,gyi,Gqt,grid,gs_per_b,...
%        gb_dim_x, gb_dim_y, gs_in_bin, gb_offset, gb_loc,...
%        gb_points_x,gb_points_y,gkblut,scale);

% sample value
% grid dim 
% samples per bin 
% bindimx 
% bindimy
% samples in bin
% bin start offset
% bin location
% bin points x 
% bin points y 
% klut
% klut scale 
%     P.qxyXqt =@(Gqt) polargrid_cub(Gqt,grid,gs_per_b,...
%         gb_dim_x, gb_dim_y, gs_in_bin, gb_offset, gb_loc,...
%         gb_points_x,gb_points_y,gkblut,scale);    
%        P.qxyXqt =@(Gqt) polargrid_cusp(Gqt,        grid,gs_per_b, gb_dim_x, gb_dim_y, gs_in_bin, gb_offset, gb_loc, gb_points_x, gb_points_y, gkblut, scale);
        P.qxyXqt =@(Gqt) polargrid_cusp(gxi,gyi,Gqt, grid, gs_per_b, gb_dim_x, gb_dim_y, gs_in_bin, gb_offset, gb_loc, gb_points_x, gb_points_y, gkblut, scale);

end

% qradon transform: (x y) to (qx qy) to (q theta) with density compensation:
gnuqradon=@(G) P.qtXqxy(P.qxyXrxy(G))./P.gDq;
% inverse qradon transform: (q theta) to (qx qy) to (x y)
gnuqiradon=@(GI) P.rxyXqxy(P.qxyXqt(GI));

% radon transform: (x y) to (qx qy) to (q theta) to (r theta):
P.gnuradon=@(G) P.rtXqt(P.qtXqxy(P.qxyXrxy(G)));
P.gnuiradon=@(GI) P.rxyXqxy(P.qxyXqt(P.qtXrt(GI)));


% data from radon (r theta) to (q theta) with density compensation
P.datafilt=@(GR) P.qtXrt(GR)./P.gDq;
P.datafiltt=@(GR) P.rtXqt(GR.*P.gDq); % and back
%gnuradona=@(G) rtXqt(gnuqradon(G)./gDQ);

% inverse radon transform: (r theta) to (q theta) to (qx qy) to (x y)
%gnuiradon=@(GI) gnuqiradon(qtXrt(GI));


nangles=size(tt,2);

op = @(x,mode) opRadon_intrnl(x,mode);
opprefilter = @(x,mode) opPrefilter_intrnl(x,mode);

    function y =opRadon_intrnl(x,mode)
        checkDimensions(nangles*Ns,Ns*Ns,x(:),mode);
        if mode == 0
            y = {nangles*Ns,Ns*Ns,[1,1,1,1],{'GNURADON'}};
        elseif mode == 1
            y=gnuqradon(reshape(x,grid));
            y=y(:);
        else
            y=gnuqiradon(reshape(x,[Ns nangles]));
            y=y(:);
        end
        
    end

    function y =opPrefilter_intrnl(x,mode)
        checkDimensions(nangles*Ns,nangles*Ns,x,mode);
        if mode == 0
            y = {nangles*Ns,nangles*Ns,[1,1,1,1],{'PREFILTER'}};
        elseif mode == 1
            %        y = (reshape(x,length(x)/nangles,nangles));
            %        P.datafilt=@(GR) P.qtXrt(GR)./P.gDq;
            %P.datafiltt=@(GR) P.rtXqt(GR.*P.gDq); % and back
            y= P.datafiltt((reshape(x,length(x)/nangles,nangles)));
            %       y=gnuqradon(reshape(x,grid));
            y=y(:);
        else
            y= P.datafilt((reshape(x,length(x)/nangles,nangles)));
            %       y=gnuqradon(reshape(x,grid));
            y=y(:);
        end
        
    end

end


           
