 function vo = mtimes(a, vi)
%	MRI "forward projection" y=A*x and backprojection x = (A')*y
%   Brad Sutton, Univ. Michigan, June 2002

if a.is.empty
	error empty
end



we = a.we;
NN = sqrt(numel(we));
we = reshape(we(:),NN,NN);


i = sqrt(-1);
tt = a.tt;

L = size(a.int,1);

if ~(size(tt) == size(a.int,2))
  sprintf('Number of time points not the same between P and Int')
  keyboard
end

tau = (max(tt)-min(tt)+eps)/(L-1);
Int = a.int;
TE = min(tt);
tt = tt-TE;

if ~a.is.transpose
        vi = exp(-i*we*TE).*reshape(vi(:),NN,NN);
        for ll = 1:L
            Wo = exp(-i*we*((ll-1)*tau));
            aa = Int(ll,:).';
            if ll == 1
               vo = aa.*[nufft2(Wo.*vi, a.st)];
            else
               vo = vo + aa.*[nufft2(Wo.*vi, a.st)];
            end            
        end
        if a.flgswt
	    vo = vo.*a.swt;
        end
	    
else
        if a.flgswt
	    vi = vi.*a.swt;  % Transpose of real sinc-weighting
        end
        for ll = 1:L
            Wo = exp(i*we*((ll-1)*tau));
            aa = Int(ll,:)';
            if ll == 1
                vo = Wo.*nufft2_adj(aa.*vi(:),a.st);
            else
                vo = vo + Wo.*nufft2_adj(aa.*vi(:),a.st);
            end  
	end
            vo = exp(i*we*TE).*vo;
	    vo = vo(:);
end





