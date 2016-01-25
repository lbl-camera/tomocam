function op = opFmsk(Fmsk,F,opinfo)
% opPolyfit  Apply arbitrary matrix as operator.
%
%    opPolyfit(nangles,OPINFO) creates an operator that performs
%    matrix-vector multiplication with matrix A. Optional parameter
%    OPINFO can be used to override the default operator
%    information returned when querying the operator, or provide a 
%    string giving the interpretation of the matrix. When OPINFO is
%    a cell array the first entry is the operator name followed by
%    additional information for that operator type (this option is
%    mostly provided for internal use by other operators). 

[nscans,nangles]=size(Fmsk);


op = @(x,mode) opFmsk_intrnl(x,mode);

function y = opFmsk_intrnl(x,mode)
% nscans=floor(numel(x)/nangles);
  checkDimensions(nangles*nscans,nangles*nscans,x,mode);
if mode == 0
    y = {nangles*nscans,nangles*nscans,[1,1,1,1],{ 'FPolyFit'}};
else % it is a projector=transpose  
%    col=@(x) x(:);
    y = reshape(F(x,1),length(x)/nangles,nangles);
    y=y.*Fmsk;
    
%    y = reshape(F(x,1),length(x)/nangles,nangles)*Pfilt2;   
     y =F(y(:),2);    
 %   y=y(:);
end

end
end
