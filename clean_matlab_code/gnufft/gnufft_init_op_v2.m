function [gnuqradon,gnuqiradon,P,op]=gnufft_init_op_v2(Ns,qq,tt,beta,k_r,center,weight,delta_r,delta_xy,Nr_orig)
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
% S. V. Venkatakrishnan, LBNL 2016

%Set constants for the file 
KBLUT_LENGTH = 256;
SCALING_FACTOR = 1.7;%What is this ? 

nangles=size(tt,2);

%fftshift factor
xx=gpuArray(single(0:Ns-1));
fftshift2D=bsxfun(@times,(-1).^xx,(-1).^xx');
%fftshift1D=(-1).^xx';

fftshift1Dop_old=@(a) bsxfun(@times,(-1).^xx',a);
fftshift1Dop=@(a) bsxfun(@times,exp(-1i*(center*2*pi/Ns).*xx'),a);
fftshift1Dop_inv=@(a) bsxfun(@times,exp(1i*(center*2*pi/Ns).*xx'),a);

% Preload the Bessel kernel (real components!)
[kblut,KB,~,KB2D]=KBlut(k_r,beta,KBLUT_LENGTH); 

KBnorm=gpuArray(single(sum(sum(KB2D((-k_r:k_r)',(-k_r:k_r))))));
kblut=kblut/KBnorm *SCALING_FACTOR;

% % Normalization (density compensation factor)
Dq=KBdensity1(qq',tt',KB,k_r,Ns)';

% mask
P.grmask =gpuArray(padmat(ones(Nr_orig,size(qq,2)),[Ns size(qq,2)]));

% deapodization factor, (the FT of the kernel):
dpz=deapodization_v2(Ns,KB,Nr_orig); %TODO : Buggy for large window sizes 

% polar to cartesian, centered
scaling=1;
[xi,yi]=pol2cart(tt*pi/180,scaling.*qq);
xi = xi+floor((Ns+1)/2);
yi = yi+floor((Ns+1)/2);
grid = [Ns,Ns];

%push to GPU
gxi=gpuArray(single(xi));
gyi=gpuArray(single(yi));
gkblut=gpuArray(single(kblut));

P.gDq=gpuArray(single(Dq));
P.gdpz=gpuArray(single(dpz));
P.weight = gpuArray(single(weight));

grid = int64([Ns,Ns]);
scale = single((KBLUT_LENGTH-1)/k_r); %TODO : What is 256 ? - Venkat 1/25/2016

% real (r) to fourier (q) -- cartesian (xy)
P.qxyXrxy=@(Grxy) (fftshift2D.*fft2(Grxy.*(P.gdpz).*fftshift2D))/Ns;%deapodized
P.rxyXqxy=@(Gqxy) fftshift2D.*ifft2((Gqxy.*fftshift2D)).*(P.gdpz)*Ns; %deapodized

% real (r) to fourier (q) -- radon space (r/q-theta)
%P.rtXqt=@(Gqt) fftshift1Dop(ifft(fftshift1Dop(Gqt))).*P.grmask;
%P.qtXrt=@(Grt) fftshift1Dop(fft(fftshift1Dop(Grt)));
P.rtXqt=@(Gqt) fftshift1Dop_old(ifft(fftshift1Dop(Gqt))).*P.grmask;
P.qtXrt=@(Grt) fftshift1Dop_inv(fft(fftshift1Dop_old(Grt)));

% q-cartesian to q-radon
P.qtXqxy=@(Gqxy) polarsample(gxi,gyi,Gqxy,grid,gkblut,scale,k_r);
P.qxyXqt=@(Gqt) polarsample_transpose(gxi,gyi,Gqt./P.gDq,grid,gkblut,scale,k_r);%Changed to account for density compensation factor - Venkat
%P.qxyXqt=@(Gqt) polarsample_transpose(gxi,gyi,Gqt,grid,gkblut,scale,k_r);

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

nangles=size(tt,2);

op = @(x,mode) opRadon_intrnl(x,mode);
opprefilter = @(x,mode) opPrefilter_intrnl(x,mode);

P.opprefilter = @(x,mode) opPrefilter_intrnl(x,mode);

    function y =opRadon_intrnl(x,mode)
        checkDimensions(nangles*Ns,Ns*Ns,x(:),mode);
        if mode == 0
            y = {nangles*Ns,Ns*Ns,[1,1,1,1],{'GNURADON'}};
        elseif mode == 1
            y=P.gnuradon(reshape(x,grid)).*P.weight.';
            y=y(:);
        else
            y=P.gnuiradon(reshape(x,[Ns nangles]).*P.weight.');
            y=y(:);
        end
        
    end

    function y =opPrefilter_intrnl(x,mode)
        checkDimensions(nangles*Ns,nangles*Ns,x,mode);
        if mode == 0
            y = {nangles*Ns,nangles*Ns,[1,1,1,1],{'PREFILTER'}};
        elseif mode == 1
            y= P.datafiltt((reshape(x,length(x)/nangles,nangles)));
            y=y(:);
        else
            y= P.datafilt((reshape(x,length(x)/nangles,nangles)));
            y=y(:);
        end
        
    end

end