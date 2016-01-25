function [e1,e2]=untitled8
N = 128;
maxerr = 1e-3;

alpha = 1.2:0.1:2;
W = 3.5:0.2:7;
beta_fract = 0.9:0.02:1.06;

x = -(N/2):(N/2);

for ialpha = 1:length(alpha)
    G = ceil(alpha(ialpha)*N);
    S = sqrt(0.37/maxerr)/alpha(ialpha);
    for iW = 1:length(W)
        for ibeta = 1:length(beta_fract)
            beta = pi*sqrt(W(iW)^2/alpha(ialpha)^2*(alpha(ialpha)-0.5)^2-0.8)*beta_fract(ibeta);
            for ix = 1:length(x)
                e1(ialpha,iW,ibeta,ix) = sqrt((2/3+1/3*cos(2*pi*x(ix)/S/G))/sinc(x(ix)/S/G)^4-1);
                e2(ialpha,iW,ibeta,ix) = get_e2(G,W(iW),beta,S,x(ix));
            end
        end
    end
end




function e2 = get_e2(G,W,beta,S,x)

e2 = 0;
for p = 1:(S-1)
    e2 = e2 + (get_cs(G,W,beta,S,x+G*p)/get_cs(G,W,beta,S,x)*sqrt(2/3+1/3*cos(2*pi*x/S/G))/sinc(x/S/G)^2)^2;
end
e2 = sqrt(e2);


function cs = get_cs(G,W,beta,S,x)

cs = sum(sinc(1/pi*sqrt((pi*W/G*(x-(-10:10)*S*G)).^2-beta^2)));
