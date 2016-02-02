  function arg = remove_spaces(arg)
%|function arg = remove_spaces(arg)
%| replace extra spaces at ends of matrix string array with zeros.
%| Copyright May 2000, Jeff Fessler, University of Michigan

for ii=1:size(arg,1)
	jj = size(arg,2);
	while (arg(ii,jj) == ' ')
		arg(ii,jj) = 0;
		jj = jj - 1;
		if jj < 1, error bug, end
	end
end
arg(:,end+1) = 0;

arg(arg(:,1) == 10, :) = []; % remove blank lines too
