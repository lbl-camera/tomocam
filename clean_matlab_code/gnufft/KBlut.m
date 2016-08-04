function [kblut, KB, KB1D,KB2D]=KBlut(k_r,beta,nlut)
% 

kk=linspace(0,k_r,nlut);
kblut = KB2( kk, k_r, beta);
scale = (nlut-1)/k_r;

kbcrop=@(x) (abs(x)<=k_r);     %crop outer values

KBnorm=norm(KB2((-k_r:k_r), 2*k_r, beta)'*KB2((-k_r:k_r), 2*k_r, beta));

%KBnorm

kblut=kblut/KBnorm;

ascale=@(x) abs(x)*scale;

KBI=@(x) ascale(x)-floor(ascale(x));
KB1D=@(x) (reshape(kblut(floor(ascale(x)).*kbcrop(x)+1),size(x)).*KBI(x)+...
	   reshape(kblut(ceil(ascale(x)).*kbcrop(x)+1),size(x)).*(1-KBI(x)))...
          .*kbcrop(x);
%KB2D=KB1D(x)*KB1D(y);
KB=@(x,y) KB1D(x).*KB1D(y);
KB2D=@(x,y) KB1D(x)*KB1D(y);

end
