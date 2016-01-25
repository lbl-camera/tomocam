 function data = load_ascii_skip_header(file)
%|function data = load_ascii_skip_header(file)
%| read ascii file, skipping lines that start with '#'
% Copyright 2007, Jeff Fessler, The University of Michigan

if nargin < 1, help(mfilename), error(mfilename), end

fid = fopen(file, 'r');
if fid == -1
	fail('problem opening "%s"', file)
end

data = [];
while 1
	line = fgetl(fid);
	if ~ischar(line)
		if isempty(data)
			fail('file %s ended too soon', file)
		else
			break % go to fclose
		end
	end
	if ~length(line) || line(1) == '#'
		continue
	end
	data(end+1,:) = sscanf(line, '%f');
end

tmp = fclose(fid);
if tmp ~= 0
	fail('problem with fclose for "%s"', file)
end
