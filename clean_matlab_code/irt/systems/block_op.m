  function out = block_op(Gb, varargin)
%|function out = block_op(Gb, varargin)
%|
%| Various operations on "block" objects and block data.
%|
%| There are (at least) three types of block objects possible in this toolbox.
%| 1. Cell arrays of matrices (or of some other matrix-like object).
%| 2. A Fatrix object with block capabilities (mtimes_block).
%| 3. The Gblock class (which is being phased out).
%|
%| This routine provides a common interface to all of them.
%|
%| arguments for object operations:
%|	'is'		is it a block object?
%|	'n'		nblock: # of blocks
%|	'ensure'	make it at least a 1-block block object (if needed)
%|
%| arguments for data operations:
%|	'ensure_block_data', data	rearrange arrays into cells
%|
%| Copyright 2005-6-19, Jeff Fessler, University of Michigan

if nargin < 2, help(mfilename), error(mfilename), end

switch length(varargin)
case 1
	out = block_op_ob(Gb, varargin{1});
case 2
	out = block_op_data(Gb, varargin{1}, varargin{2});
otherwise
	error 'bad args'
end


%
% block_op_data()
%
function out = block_op_data(Gb, arg, data)

if streq(arg, 'ensure_block_data')
	if iscell(data)
		out = data; % ok, already cell

	elseif isa(Gb, 'Fatrix')
		if ~isempty(Gb.handle_blockify_data)
			out = blockify_data(Gb, data);
			return
		end

		% attempt to be smart for backward compat!
		try
			nblock = block_op(Gb, 'n');
			starts = subset_start(nblock);
			out = cell(1,nblock);

			if ndims(data) == 3 % [ns,nt,na]
				data = reshaper(data, '2d'); % [ns*nt,na]
			end

			na = size(data,2);
			if nblock > na, error 'tried 2d but failed', end

			for iset=1:nblock
				istart = starts(iset);
				ia = istart:nblock:na;
				out{istart} = col(data(:,ia));
			end

		catch
			error([mfilename  ': failed to blockify data'])
		end
	else
		error 'unknown blockify case'
	end
else
	error 'bug'
end


%
% block_op_ob()
%
function out = block_op_ob(Gb, arg)

switch arg
case 'is'
	if iscell(Gb)
		out = true;
	elseif isa(Gb, 'Fatrix')
		out = ~isempty(Gb.nblock);
	elseif isa(Gb, 'Gblock')
		out = true;
	else
		out = false;
	end

case 'n'
	if ~block_ob(Gb, 'is'), error 'not a block object', end
	if iscell(Gb)
		out = length(Gb);
	elseif isa(Gb, 'Fatrix')
		out = Gb.nblock;
	elseif isa(Gb, 'Gblock')
		out = Gb.nblock;
	end

case 'ensure'
	if block_ob(Gb, 'is')
		out = Gb;
		return
	end

	if iscell(Gb)
		error 'bug: cell already is a block object'
	elseif isnumeric(Gb)
		Gb = Gsparse(Gb);
		Gb = Gblock(Gb, 1, 0);
	elseif isa(Gb, 'Fatrix') && ~isempty(Gb.nblock)
		out = Gb;
	else
		out = Gblock(Gb, 1, 0);
	end

otherwise
	fail(['unknown argument: ' arg])
end
