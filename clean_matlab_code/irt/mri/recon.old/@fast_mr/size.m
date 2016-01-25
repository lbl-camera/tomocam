function dim = size(ob)
%function dim = size(ob)
%       "size" method for Gtomo2 class
tmp = ob.st;

dim = [tmp.M (tmp.N1*tmp.N2)];

