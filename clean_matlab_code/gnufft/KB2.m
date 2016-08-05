function w = KB2(x, k_r, beta)
%
% if( 1 ) 
%    bes = abs(besseli(0, beta));
    w = besseli(0, beta*sqrt(1-(x/k_r).^2)) ;
%    w=w/abs(besseli(0, beta));
    w=(w.*(x<=k_r));
%    kbcrop=@(x) (abs(x)<=k_r);     %crop outer values

% else
%     w = 1;
% end

