% test_all_systems

list = {
	'Fatrix_test_basic test'
	'fatrix_one test'
	'block_fatrix_test'
	'Gblock_test'
	'Gblur_test'
	'Gcascade test'
	'Gcone_test'
	'Gdsft_test'
	'Gnufft_test0'
	'Gnufft_test'
%	'Gtomo2_dscex test' % within Gwtmex_test
	'Gtomo2_moj_test'
	'Gtomo2_strip test'
	'Gtomo2_table_test'
	'Gtomo2_wtmex_test'
	'Gtomo3_test'
	'Gtomo3_test_adj'
	'Gtomo_nufft_test'
	'Gtomo_nufft_test_adj'
	'Gsparse_test'
	'wtf_read test'
};

im nan-fail
run_mfile_local(list)
