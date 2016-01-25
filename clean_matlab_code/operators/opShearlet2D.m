function op = opShearlet2D(Nsx,Nsy,opinfo)
% opPolyfit(nangles,nscans, OPINFO) 
%


scales = 2;
shearLevels = [1 1];

col=@(x) x(:);

shearletSystem = SLgetShearletSystem2D(1,Nsx,Nsy,scales);

%coeffs = SLsheardec2D(signal,shearletSystem);
%Xrec = real(SLshearrec2D(coeffs,shearletSystem));

if nargin < 3
  opinfo = {'Shearlet', []};
elseif ischar(opinfo)
  opinfo = {'Shearlet', opinfo};
end

op = @(x,mode) opShearlet_intrnl(x,mode);


function y = opShearlet_intrnl(x,mode)
  checkDimensions(Nsx*Nsy,Nsx*Nsy*shearletSystem.nShearlets,x,mode);
if mode == 0
    y = {Nsx*Nsy,shearletSystem.nShearlets* Nsx*Nsy, [1,1,1,1],opinfo};
elseif mode ==2
    y=col(SLsheardec2D(reshape(x,[Nsx,Nsy]),shearletSystem));

else
    y = col(SLshearrec2D(...
    reshape(x,Nsx,Nsy, shearletSystem.nShearlets),shearletSystem));
        
end

end
end
