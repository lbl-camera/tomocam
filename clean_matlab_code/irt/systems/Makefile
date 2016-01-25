#	Makefile

dir1 = ../../../../src/matlab/alg/systems/@Gtomo_nufft

pees = $(dir1)/.print

plist = $(dir1)/*.m

en7=print,code
comm=$(en7) `linesplit $? | sort`; lpq; touch $@
$(dir1)/.print:	$(dir1)/Gtomo_nufft.m $(dir1)/mtimes.m
	$(comm)
