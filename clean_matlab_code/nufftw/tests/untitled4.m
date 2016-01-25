%W = 3.84326171875;
%alpha = 2;
%maxerr = 1e-3;
load noncartesian_phantom.mat
kdata = kdata.*dcf;
N = 127; % output image size
imagedirect = read_image('~/Documents/nuFFTW/out/imagedirect127.dd');
W = 3.7982177734375;
alpha = 1.3;
maxerr = 1e-2; % maximum aliasing error allowed
nufft_st = init_nufft(kcoord, N, alpha, W, maxerr);

if mod(nufft_st.G,2) == 0
    [a,b] = meshgrid(1:nufft_st.G,1:nufft_st.G);
    mask = (-1) .^ (a+b);
    mask1 = (-1).^(1:nufft_st.G);
else
    a = -(0:(nufft_st.G-1))*pi/nufft_st.G;
    a(2:2:end) = a(2:2:end)+pi;
    [a2,a3] = meshgrid(a,a);
    mask = exp(1i*(a2+a3));
    mask1 = exp(1i*mask1);
end
convmat = nufft_st.convmat .* repmat(mask(:),1,size(kcoord,1));
g2 = convmat*kdata;
g2 = reshape(g2,nufft_st.G,nufft_st.G);
imno = ifft2(g2);
imno = imno(nufft_st.nstart+(1:nufft_st.N),nufft_st.nstart+(1:nufft_st.N));


impulse = zeros(size(kdata));
impulse(1) = 1;
gridded_impulse = convmat * impulse;
gridded_impulse = reshape(gridded_impulse, nufft_st.G, nufft_st.G);
deap = ifft2(gridded_impulse);
deap = deap(nufft_st.nstart+(1:nufft_st.N),nufft_st.nstart+(1:nufft_st.N));
image_deap = imno ./ deap;

G = ceil(alpha*N);
beta = pi*sqrt(W^2/alpha^2*(alpha-0.5)^2-0.8);
W2G = W/2/G;
S = sqrt(0.37/maxerr)/alpha;
k_kb = (-W2G-(1/S/G)):(1/S/G):(W2G+(1/S/G));
KB = G/W*besseli(0,beta*sqrt(1-(2*k_kb*G/W).^2));
a = abs(k_kb) > W2G;
KB(a) = 0;

ksx = -0.5:(1/G):0.5;
nxy = abs(ksx)<=W2G;
conv_x = interp1(k_kb, KB, ksx(nxy));
g = zeros(1,G);
g(nxy) = conv_x;
g = g.*mask1;
d = ifft(g);
[d11,d12] = meshgrid(d,d);
d1 = d11.*d12;
d1 = d1(nufft_st.nstart+(1:nufft_st.N),nufft_st.nstart+(1:nufft_st.N));
im1 = imno./d1;

x1 = -floor(G/2):ceil(G/2-1);
d2 = sinc(sqrt((pi*W*x1/G).^2-beta^2)/pi);
d2 = d2.*mask1;
[d11,d12] = meshgrid(d2,d2);
d2 = d11.*d12;
d2 = d2(nufft_st.nstart-1+(1:nufft_st.N),nufft_st.nstart-1+(1:nufft_st.N));
im2 = imno./d2;

figure
plot(abs(deap(64,:)),'.-')
hold on
plot(abs(d1(64,:)),'r.-')
plot(abs(d2(64,:)),'g.-')
figure
imshowz([get_image_errors(imagedirect,image_deap) get_image_errors(imagedirect,im1) get_image_errors(imagedirect,im2)])

%plot(x1,abs(d))
%hold on
% plot(x1,abs(d2),'r')
% x = -2*floor(G/2):2*ceil(G/2-1);
% d3 = sinc(sqrt((pi*W*x/G).^2-beta^2)/pi);
% plot(x,abs(d3),'g')
% d4 = d3;
% d4(1:165) = d4(1:165)+d3(167:end);
% d4((end-165+1):end) = d4((end-165+1):end)+d3(1:165);
% plot(x,abs(d4),'m')
% ns = ceil((331-166)/2);
% d5 = d4(ns+(1:G));
% plot(x1,abs(d5),'k')
% still need to multiply by sinc




%%
KB(ceil(S*G)) = 0;
f = ifftshift(ifft(KB));
figure
plot(abs(f))




KB(ceil(S*G)) = 0;
b2 = ifft(KB.*(-1).^(1:ceil(S*G)));
ns = ceil((ceil(S*G)-N)/2);
b2 = b2(ns+(1:N));
b2 = b2 .* sinc((-floor(N/2):ceil(N/2-1))/ceil(S*G)).^2;
[b3,b4] = meshgrid(b2,b2);
bb = b3.*b4;

KB(ceil(S*G)) = 0;
[kb1,kb2] = meshgrid(KB);
kb = kb1.*kb2;
kkb = ifft2(kb);
kkb = ifftshift(kkb);
ns = ceil((ceil(S*G)-N)/2);
kkb = kkb(ns+1+(1:N),ns+1+(1:N));
get_image_errors(abs(deap),abs(kkb));
%x = -floor(nufft_st.N/2):ceil(nufft_st.N/2-1); % image pixel coordinates
% sincmask = sinc(x/ceil(S*G)).^2;
% sincmask = sincmask'*sincmask;
% kkb = kkb .* sincmask;
%get_image_errors(abs(deap),abs(kkb));
%if mod(N,2) == 0
    x = -floor(N/2):ceil(N/2-1); % image pixel coordinates
    mask = (-1).^(ones(size(x))'*x+x'*ones(size(x)));
%else
   a = -(0:(N-1))*pi/N;
   a(2:2:end) = a(2:2:end)+pi;
   [a2,a3] = meshgrid(a,a);
   mask = exp(1i*(a2+a3));
%end
%mask = (-1).^(ones(size(x))'*x+x'*ones(size(x)));
kkb1 = kkb.*mask;
get_image_errors(abs(deap),abs(kkb1));
figure
imshowz(get_image_errors(deap,kkb1))



if mod(nufft_st.G,2) == 0
    [a,b] = meshgrid(1:nufft_st.G,1:nufft_st.G);
    mask = (-1) .^ (a+b);
else
    a = -(0:(nufft_st.G-1))*pi/nufft_st.G;
    a(2:2:end) = a(2:2:end)+pi;
    [a2,a3] = meshgrid(a,a);
    mask = exp(1i*(a2+a3));
end
convmat = nufft_st.convmat .* repmat(mask(:),1,size(kcoord,1));
%if mod(nufft_st.N,2) == 0
    x = -floor(nufft_st.N/2):ceil(nufft_st.N/2-1); % image pixel coordinates
    mask = (-1).^(ones(size(x))'*x+x'*ones(size(x)));
%else
 %   a = -(0:(nufft_st.N-1))*pi/nufft_st.N;
 %   a(2:2:end) = a(2:2:end)+pi;
  %  [a2,a3] = meshgrid(a,a);
   % mask = exp(1i*(a2+a3));
%end
deapmat = nufft_st.deapmat .* mask;
g2 = convmat*kdata;
g2 = reshape(g2,nufft_st.G,nufft_st.G);
imno = ifft2(g2);
imno = imno(nufft_st.nstart+(1:nufft_st.N),nufft_st.nstart+(1:nufft_st.N));
image_deap2 = imno ./ deapmat;
get_image_errors(imagedirect,image_deap2);







image_nodeap = ifftshift(ifft2(gridded_data));


gridded_data1 = vec2complex(readbin('code/out.dd'));
gridded_data1 = reshape(gridded_data1,167,167);
get_image_errors(gridded_data,gridded_data1);

oversampled_image1 = vec2complex(readbin('code/out1.dd'));
oversampled_image1 = reshape(oversampled_image1,167,167);
oversampled_image1 = ifftshift(oversampled_image1);
get_image_errors(image_nodeap,oversampled_image1);


gridded_data2 = vec2complex(readbin('code/out.dd'));
gridded_data2 = reshape(gridded_data2,167,167);
x = 1:167;
[x,y] = meshgrid(x,x);
x = (-1).^(x+y);
gridded_data3 = gridded_data2.*x;
get_image_errors(gridded_data,gridded_data3);

oversampled_image2 = vec2complex(readbin('code/out1.dd'));
oversampled_image2 = reshape(oversampled_image2,167,167);
get_image_errors(ifft2(gridded_data2),oversampled_image2);
