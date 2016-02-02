function [kblut, KB, KB1D]=gKBlut(k_r,beta,nlut)
% 

kk=gpuArray.linspace(0,k_r,nlut);
kblut = KB2( kk, 2*k_r, beta);
scale = (nlut-1)/k_r;

kbcrop=@(x) (abs(x)<=k_r);     %crop outer values

KBI=@(x) abs(x)*scale-floor(abs(x)*scale);
KB1D=@(x) (reshape(kblut(floor(abs(x)*scale).*kbcrop(x)+1),size(x)).*KBI(x)+...
          reshape(kblut(ceil(abs(x)*scale).*kbcrop(x)+1),size(x)).*(1-KBI(x)))...
          .*kbcrop(x);
KB=@(x,y) KB1D(x).*KB1D(y);

%  
% % .*kblut(KBP(y))
% 
% % a convoluted way to avoid if statements:
% % s2ic=@(x,y) s2i(x.*kbcrop(x,y)+1,y.*kbcrop(x,y)+1);    %crop outer input, set index to 1
% 
% KBc = @(x) (abs(x)*scale-floor(abs(x)*scale));  %interpolation on table
% klbut(KBP(x))*
% 
% KB = @(x,y) (kbcrop(x,y).*...
%     (KBlut(s2ic(x,y)) .* ((1-KBc(x)).*(1-KBc(y))) )+...
%     (KBlut(s2ic(x,y)) .* (KBc(x).* (1-KBc(y))) ) + ...
%     (KBlut(s2ic(x,y)) .* ((1-KBc(x)).* KBc(y)) ) + ...
%     (KBlut(s2ic(x,y)) .* (KBc(x).* KBc(y)) )...
%     );
% 
% % %create 2D KB function
% % KBfunc = @(x,y) KB1(nj*2+1, x, beta)'*KB1(nj*2+1, y, beta);
% % 
% % % Preload the Bessel kernel (only care about real components!)
% % span = [0:0.1:nj+1];   %table lookup
% % KBlut = KBfunc(span, span); %compute table
% % KBlut = real(KBlut);        %keep real terms
% % % sz = ((nj+1)*10)+1;         %size of table
% % 
% % ns=numel(span);
% % %KB lookup
% % KBc = @(x) (abs(x)*10-floor(abs(x)*10));  %interpolation on table
% % KB2 = @(x,y) ...
% %     (KBlut(sz+floor(abs(x)*10), sz+floor(abs(x)*10)) .* ((1-KBc(x))'* (1-KBc(y))) )+...
% %     (KBlut(sz+ceil(abs(x)*10),  sz+floor(abs(x)*10)) .* (KBc(x)'* (1-KBc(y))) ) + ...
% %     (KBlut(sz+floor(abs(x)*10), sz+ceil(abs(x)*10)) .* ((1-KBc(x))'* KBc(y)) ) + ...
% %     (KBlut(sz+ceil(abs(x)*10),  sz+ceil(abs(x)*10)) .* (KBc(x)'* KBc(y)) );
% 
% 
% % %crop if x>nj, y>nj;
% % KB=@(x,y) KBcrop(x,y,nj,KB2);
% % 
% 
% end
% % function w = KB1(M, x, beta)
% % bes = abs(besseli(0, beta));
% % w = besseli(0, beta*sqrt(1-(2*x/M).^2)) / bes;
% % end
% %  
% % % function out=KBcrop(x,y,dx,KB2)
% % % 
% % % 
% % %  mm=((x>-dx)&(x<=dx)&(y>-dx)&(y<=dx));
% % % %dx=gsingle((sz-1)*2+1);
% % % % s2i=@(x,y) (x+(y-1)*dx);
% % % % out=@(x,y) KBlut((x+(y-1)*dx).*mm+(1-mm)).*mm;
% % %  out= KB2(x.*mm+(1-mm),y.*mm+(1-mm)).*mm;
% % % % out= KB2(x,y).*((x>0)&(x<=dx)&(y>0)&(y<=dx));
% % % 
% % % end
% % % 
