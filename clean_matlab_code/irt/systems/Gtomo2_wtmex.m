  function ob = Gtomo2_wtmex(arg1, varargin)
%|function ob = Gtomo2_wtmex(sg, ig, options)
%|function ob = Gtomo2_wtmex(file, options)
%|function ob = Gtomo2_wtmex(arg_pairs, options)
%|
%| Generate a tomographic system model based on aspire's wtfmex() routine.
%|
%| in
%|	sg	strum		sino_geom()
%|	ig	strum		image_geom()
%|
%| options
%|	'grouped'		row or col (default: row)
%|	'nthread'		pthreads for multiple processors (default: 1)
%|	'mask'	[nx,ny]		logical support mask, has precedence
%|	'pairs' {}		cell array of name/value pairs for aspire_pair()
%|
%| out
%|	ob	[nb*na,np]	Fatrix system "matrix", np = sum(ig.mask(:))
%|
%| Copyright 2006-3-3, Jeff Fessler, University of Michigan

if nargin == 1 && streq(arg1, 'test'), Gtomo2_wtmex_test, return, end
if nargin < 1, help(mfilename), error(mfilename), end

% options
arg.grouped = 'row';
arg.nthread = 1;
arg.mask = [];
arg.chat = 0;
arg.pairs = {};

%% warn '!!!!!!!! under repair may not work !!!!!!!!!!'

% given wtf or argument pairs
if ischar(arg1)

	arg = vararg_pair(arg, varargin);

	if exist(arg1, 'file') % file.wtf
		if ~isempty(arg.mask), error 'no mask option given .wtf', end
		arg.file = arg1;
%%		[nx ny nb na] = wtfmex('read', arg.file);
		[arg.buff nx ny nb na] = wtfmex('asp:read', arg.file, int32(arg.chat));

	else % arg_pair
		arg.arg = arg1;
		if isempty(arg.mask)
			mask_arg = {};
		else
			if ~islogical(arg.mask), error 'need logical mask', end
			mask_arg = uint8(arg.mask);
		end

%%		wtfmex('gensys', arg.arg', arg.grouped, mask_arg{:});
		arg.buff = wtfmex('asp:gensys', arg.arg', arg.grouped, mask_arg{:});

		nx = str2num(arg_get(arg.arg, 'nx'));
		ny = str2num(arg_get(arg.arg, 'ny'));
		nb = str2num(arg_get(arg.arg, 'nb'));
		na = str2num(arg_get(arg.arg, 'na'));
	end

%%	mask = wtfmex('mask') > 0;
	mask = wtfmex('asp:mask', arg.buff) > 0;
	arg.ig = image_geom('nx', nx, 'ny', ny, 'dx', -1, 'mask', mask);
	arg.sg = sino_geom('par', 'nb', nb, 'na', na);

% given sino_geom() and image_geom()
else

	arg.sg = arg1;
	arg.ig = varargin{1};
	varargin = {varargin{2:end}};
	arg = vararg_pair(arg, varargin);
	if ~isempty(arg.mask)
		arg.ig.mask = arg.mask;
	end

	arg.aspire_arg = aspire_pair(arg.sg, arg.ig, 'support', 'array', ...
 		arg.pairs{:}); % trick: pairs can override default support

%%	wtfmex('gensys', arg.aspire_arg', arg.grouped, uint8(arg.ig.mask));
	arg.buff = wtfmex('asp:gensys', arg.aspire_arg', arg.grouped, uint8(arg.ig.mask));

end

arg.power = 1;
arg.nd = arg.sg.nb * arg.sg.na;
arg.np = sum(arg.ig.mask(:));
dim = [arg.nd arg.np]; % trick: make it masked by default!

arg.nthread = int32(arg.nthread);

% wtfmex() method(s)
arg.stayman2_factors = @(G, wi) wtfmex('asp:stayman2', G.arg.buff, single(wi));

%
% build Fatrix object
%
ob = Fatrix(dim, arg, 'caller', 'Gtomo2_wtmex', ...
	'forw', @Gtomo2_wtmex_forw, 'back', @Gtomo2_wtmex_back, ...
	'free', @Gtomo2_wtmex_free, ...
	'mtimes_block', @Gtomo2_wtmex_mtimes_block);

if arg.chat
%%	wtfmex('print');
	wtfmex('asp:print', arg.buff);
end


%
% Gtomo2_wtmex_forw(): y = G * x
%
function y = Gtomo2_wtmex_forw(arg, x)
y = Gtomo2_wtmex_mtimes_block(arg, 0, x, 1, 1);


%
% Gtomo2_wtmex_back(): x = G' * y
% full backprojection
%
function x = Gtomo2_wtmex_back(arg, y)
x = Gtomo2_wtmex_mtimes_block(arg, 1, y, 1, 1);


%
% Gtomo2_wtmex_mtimes_block()
%
function y = Gtomo2_wtmex_mtimes_block(arg, is_transpose, x, istart, nblock)

if is_transpose
	fun = @Gtomo2_wtmex_block_back;
else
	fun = @Gtomo2_wtmex_block_forw;
end
y = embed_mult(fun, arg, is_transpose, x, istart, nblock, ...
	arg.ig.mask, arg.np, [arg.sg.nb arg.sg.na], 1);


%
% Gtomo2_wtmex_block_forw()
%
function y = Gtomo2_wtmex_block_forw(arg, dummy, x, istart, nblock)

if arg.power ~= 1, fail('power=%d not done', arg.power), end

if nblock == 1
%%	y = wtfmex('chat', arg.chat, 'mult', single(x));
	y = wtfmex('asp:forw', arg.buff, arg.nthread, single(x), int32(arg.chat));
else
%%	y = wtfmex('chat', arg.chat, 'proj,block', single(x), ...
	y = wtfmex('asp:proj,block', arg.buff, arg.nthread, single(x), ...
			int32(istart-1), int32(nblock), int32(arg.chat));

	% fix: extract the relevant columns - should do in wtfmex?
	ia = istart:nblock:arg.sg.na;
	y = y(:,ia,:);
end

y = double6(y);


%
% Gtomo2_wtmex_block_back()
%
function x = Gtomo2_wtmex_block_back(arg, dummy, y, istart, nblock)

if nblock == 1
	if arg.power == 2
%%		x = wtfmex('chat', arg.chat, 'back2', single(y));
		x = wtfmex('asp:back2', arg.buff, arg.nthread, single(y), int32(arg.chat));
	elseif arg.power == 1
%%		x = wtfmex('chat', arg.chat, 'back', single(y));
		x = wtfmex('asp:back', arg.buff, arg.nthread, single(y), int32(arg.chat));
	else
		fail('power=%d not done', arg.power)
	end
else
	if arg.power ~= 1, fail('power=%d not done', arg.power), end
%%	x = wtfmex('chat', arg.chat, 'back,block', single(y), ...
	x = wtfmex('asp:back,block', arg.buff, arg.nthread, single(y), ...
			int32(istart-1), int32(nblock), int32(arg.chat));
end
x = double6(x);


%
% Gtomo2_wtmex_free()
%
function Gtomo2_wtmex_free(arg)
printm 'freeing wtfmex static memory'
%% wtfmex('free');



%
% Gtomo2_wtmex_power()
%
function ob = Gtomo2_wtmex_power(ob, sup)
ob.arg.power = ob.arg.power * sup;
