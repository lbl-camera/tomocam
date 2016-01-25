% irt_mex_make.m

if ispc
	mex penalty_mex.c mexarg.c 'penalty,diff.c' -DIs_pc -DMmex -outdir ../v7
else
	mex -v penalty_mex.c mexarg.c 'penalty,diff.c' -DMmex -outdir ../v7
end
