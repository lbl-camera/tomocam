% test_all_mex
% make sure all mex files can execute
% this should just produce many "usage" messages

list = {
	'jf_mex'
	'dtft_mex'
	'exp_xform_mex'
	'mri_exp_mult_mex'

	'interp1_table_adj_mex'
	'interp1_table_mex'
	'interp2_table_adj_mex'
	'interp2_table_mex'
	'interp3_table_adj_mex'
	'interp3_table_mex'

	'penalty_mex'
	'rotmex'
	'wtfmex'
	'f3d_mex'
};

% check for UM-only mex files
if exist('dd_ge1_mex') == 3
	list{end+1} = 'dd_ge1_mex';
end
if exist('dd_ge2_mex') == 3
	list{end+1} = 'dd_ge2_mex';
end

passed = '';
failed = '';
for ii=1:length(list)
	mex = list{ii};
	try
		eval(mex)
		passed = [passed ' ' mex];
	catch
		failed = [failed ' ' mex];
	end
end

if ~isempty(failed)
	printf(['These mex files failed: ' failed])
	disp 'Sorry, you seem to have mex problems. :-('
	disp 'Probably you are a PC user and Windoze is not supported.'
	disp 'Or (in linux) there may be an annoying gcc library version issue.'
	disp 'Otherwise you probably have a path problem.'
	
	if ~isempty(passed)
		printf(['These mex files passed: ' passed])
		disp 'So perhaps some things will still work.'
	end

else
	disp '------------------------------------------------------------'
	printm(['All mex files passed:' passed])
	disp '------------------------------------------------------------'
	printm('All mex files appear to be properly executable,')
	disp 'and are properly displaying their internal help messages.'
end
