function [gnuqradon,gnuqiradon,P,op]=gnufft_init_spmv_op_v2(Ns,qq,tt,beta,k_r,center,weight,delta_r,delta_xy,Nr_orig)
% function [gnuradon,gnuiradon,qtXqxy,qxyXqt]=gnufft_init(Ns,q,t,beta,k_r)
%
% returns radon  and inverse radon transform operators (GPU accelerated)
%
% input: Ns, x-y grid size
%        qq,tt (polar coordinates q, theta)
%        beta, kaiser-bessel parameter,
%        k_r, kernel width
%        center - center of rotation in units of pixel index 
%        delta_r - lenght of detector pixel in units of micron - TODO 
%        delta_xy - length of object voxel size - assuming square/cubic voxels 
%        Nr_orig : The length of the original detector in pixels. Used to
%        mask out values during FFT. 
%Output:
%        radon and inverse radon operators, geometry is fixed and embedded
%        also gridding and inverse gridding operators, with fixed geometry
%        radon and inverse radon wrap around some FFTs
%
%
%
% Stefano Marchesini,  LBNL 2013
% Modifications by S.V. Venaktakrishnan, LBNL 2016
% if nargin<6
%     uniqueness=false;
% end

%Set constants for the file 
KBLUT_LENGTH = 256;
SCALING_FACTOR = 1.7;%What is this ? 
OVERSAMPLING = 2; %New addition - venkat

nangles=size(tt,2);

%fftshift factor
xx=gpuArray(single(0:Ns-1));
fftshift2D=bsxfun(@times,(-1).^xx,(-1).^xx');
%fftshift1D=(-1).^xx';
fftshift1Dop_old=@(a) bsxfun(@times,(-1).^xx',a);
fftshift1Dop=@(a) bsxfun(@times,exp(-1i*(center*2*pi/Ns).*xx'),a);
fftshift1Dop_inv=@(a) bsxfun(@times,exp(1i*(center*2*pi/Ns).*xx'),a);

% Preload the Bessel kernel (real components!)
[kblut,KB,KB1,KB2D]=KBlut(k_r,beta,KBLUT_LENGTH); 

KBnorm=gpuArray(single(sum(sum(KB2D((-k_r:k_r)',(-k_r:k_r))))));
kblut=kblut/KBnorm*SCALING_FACTOR; %scaling fudge factor
%TODO : Remove fudge factors - Venkat 
figure;plot(kblut);title('KB window');

% % Normalization (density compensation factor)
Dq=KBdensity1(qq',tt',KB,k_r,Ns)';
% <------mask
%P.grmask=gpuArray(abs(qq)<size(qq,1)/4*3/2);%TODO : What are these numbers ? Venkat 
%P.grmask=gpuArray(abs(qq)<size(qq,1)*3/2);
P.grmask =gpuArray(padmat(ones(Nr_orig,size(qq,2)),[Ns size(qq,2)]));

% deapodization factor, (the FT of the kernel):
dpz=deapodization_v2(Ns,KB,Nr_orig); %TODO : Buggy for large window sizes 
% gdpz=gpuArray(single(dpz));

% polar to cartesian, centered
[xi,yi]=pol2cart(tt*pi/180,1*qq);
xi = xi+floor((Ns+1)/2);
yi = yi+floor((Ns+1)/2);
grid = [Ns,Ns];

% push parameters to gpu
gxi=gpuArray(single(xi));
gyi=gpuArray(single(yi));
gkblut=gpuArray(single(kblut));

P.gDq=gpuArray(single(Dq));
P.gdpz=gpuArray(single(dpz));
P.weight = gpuArray(single(weight));
grid = int64([Ns,Ns]);
scale = single((KBLUT_LENGTH-1)/k_r); %TODO : What is 256 ? - Venkat 1/25/2016

% normalize by KB factor
%cnorm=gpuArray(single(sum(sum(KB2D((1:Ns)'-Ns/2,(1:Ns)-Ns/2)))));
cnorm=gpuArray(single(1)); %TO DO : ?? Venkat

% SPMV stuff
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


% pre-compute kernel value for every shift
gval=single(bsxfun(@times,KB1(bsxfun(@plus,xf,double(kkrx))),KB1(bsxfun(@plus,yf,double(kkry)))));

% index of where every point lands  on the image

%replicate over -k_r:k_r
xii=int32(bsxfun(@plus,xint,kkrx));
yii=int32(bsxfun(@plus,yint,kkry));

%index of where every point lands
grow=bsxfun(@plus,(yii)*Ns,xii+1); 
 %jkx=floor((grow-1)/Ns)+1;jky=mod(grow-1,Ns)+1;
%index of where every point comes from
gcol=repmat(int32(reshape(1:numel(qq),size(qq))),[ 1 1 nkr nkr]);

% remove stencils if they land outside the frame
iin=find(bsxfun(@and,(xii>0) & ( xii<Ns) , (yii>1) & (yii<Ns)));
if numel(iin)<numel(gval);
    grow=grow(iin);     gcol=gcol(iin);    gval=gval(iin);
end
% %%
% incorporate the density compensation factor into SPMV
gval1=complex(gval)./P.gDq(gcol);

%build the spmv array
RT=gcsparse(gcol,grow,gval1,nrow,ncol,1);
% we'll do the transpose with polarsample, which is faster...
% PT=gcsparse(grow,gcol,complex(gval),ncol,nrow,1);
%
% end of SPMV

% real (r) to fourier (q) -- cartesian (xy)
gdpz1=(P.gdpz).*fftshift2D;
P.qxyXrxy=@(Grxy) (fftshift2D.*fft2(Grxy.*gdpz1))/Ns;%deapodized
P.rxyXqxy=@(Gqxy) ifft2((Gqxy.*fftshift2D)).*(gdpz1)*Ns; %deapodized

%P.qxyXrxy=@(Grxy) (fftshift2D.*fft2(Grxy.*(P.gdpz).*fftshift2D))/Ns;%deapodized
%P.rxyXqxy=@(Gqxy) fftshift2D.*ifft2((Gqxy.*fftshift2D)).*(P.gdpz)*Ns; %deapodized

% real (r) to fourier (q) -- radon space (r/q-theta)
P.rtXqt=@(Gqt) fftshift1Dop_old(ifft(fftshift1Dop(Gqt))).*P.grmask;
P.qtXrt=@(Grt) fftshift1Dop_inv(fft(fftshift1Dop_old(Grt)));

% q-cartesian to q-radon
P.qtXqxy=@(Gqxy) polarsample(gxi,gyi,Gqxy,grid,gkblut,scale,k_r);
%%
% q-radon to q cartesian
% (qx,qy) <-> (q, theta) : cartesian <-> non-uniform samples
% gridding using SPMV:
P.qxyXqt =@(Gqt) reshape(RT*(Gqt(:)),[Ns Ns])/cnorm;

% radon transform: (x y) to (qx qy) to (q theta) to (r theta):
P.gnuradon=@(G) P.rtXqt(P.qtXqxy(P.qxyXrxy(G)));

%Iradon to go from (r theta) to (q theta) to (qx qy) to (x y): ? - Venkat:
%1/25/2016
P.gnuiradon=@(GI) P.rxyXqxy(P.qxyXqt(P.qtXrt(GI)));

% fast partial radon transform:
% remove 1D ffts in (q theta) to (r theta) transform by
% precomputing partial transform of our data from (r theta) to (q-theta)
P.datafilt=@(GR) P.qtXrt(GR)./P.gDq;
P.datafiltt=@(G) P.rtXqt(G.*P.gDq); % and back

% qradon transform: (x y) to (qx qy) to (q theta) with density compensation:
%gnuqradon=@(G) P.qtXqxy(P.qxyXrxy(G))./P.gDq;
gnuqradon=@(G) P.qtXqxy(P.qxyXrxy(G))./P.gDq;
% inverse qradon transform: (q theta) to (qx qy) to (x y)
gnuqiradon=@(GI) P.rxyXqxy(P.qxyXqt(GI.*P.gDq));

op = @(x,mode) opRadon_intrnl(x,mode);


P.opprefilter = @(x,mode) opPrefilter_intrnl(x,mode);
%P.opprefilter = @(x,mode) opIdentity_intrnl(x,mode);


    function y =opRadon_intrnl(x,mode)
        checkDimensions(nangles*Ns,Ns*Ns,x(:),mode);
        if mode == 0
            y = {nangles*Ns,Ns*Ns,[1,1,1,1],{'GNURADON'}};
        elseif mode == 1
            %y=gnuqradon(reshape(x,grid));
            y=P.gnuradon(reshape(x,grid)).*P.weight.';
            y=y(:);
        else
            y=P.gnuiradon(reshape(x,[Ns nangles]).*P.weight.');
%                        y=gnuqiradon(reshape(x,[Ns nangles]));
            y=y(:);
        end
        
    end

    function y =opPrefilter_intrnl(x,mode)
        checkDimensions(nangles*Ns,nangles*Ns,x,mode);
        if mode == 0
            y = {nangles*Ns,nangles*Ns,[1,1,1,1],{'PREFILTER'}};
        elseif mode == 1
            y= P.datafiltt((reshape(x,Ns,nangles)));
            %       y=gnuqradon(reshape(x,grid));
            y=y(:);
        else
            y= P.datafilt((reshape(x,Ns,nangles)));
            %       y=gnuqradon(reshape(x,grid));
            y=y(:);
        end
    end

    function y =opIdentity_intrnl(x,mode)
        checkDimensions(nangles*Ns,nangles*Ns,x,mode);
        if mode == 0
            y = {nangles*Ns,nangles*Ns,[1,1,1,1],{'IDENTITY'}};
        else
            y=x;
        end
        
    end
end