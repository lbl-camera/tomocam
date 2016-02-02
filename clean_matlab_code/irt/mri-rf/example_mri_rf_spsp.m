% example_mri_rf_spsp
%This script is an example of spectral-spatial pulse design for
%through-plane phase precompensatory slice selection for T2*-weighted
%functional MR, based on our publication:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Spectral-spatial RF pulse design script, based on "Spectral-spatial RF
%pulse design for through-plane phase precompensatory slice selection for
%T2*-weighted functional MRI", Chun-yu Yip et al, Magnetic Resonance in
%Medicine, 2009. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%Written by Chun-yu Yip, University of Michigan, Ann Arbor, 4/1/2009
%Current affiliation: Athinoula A. Martinos Center for Biomedical Imaging,
%Massachusetts General Hospital, Harvard Medical School, Charlestown, MA,
%USA. Email address: chunyuy@nmr.mgh.harvard.edu
%
%Running this script requires that the entire image reconstruction toolbox
%by Professor Jeffrey A. Fessler be installed first: 
%
%http://www.eecs.umich.edu/~fessler/code/index.html
%
%You can conveniently download the whole package at
%
%http://www.eecs.umich.edu/~fessler/irt/fessler.tgz
%
%and use the unix command 'tar' to retrieve the files and folders.
% Use the "setup.m" command in the toolbox to set up paths first.
%

%First load pulse design parameters into matlab structures kp, rfp, iop.
if ~isvar('kp'), printm 'Loading parameters...'
	[kp iop rfp] = example_spsp_param1();
end

%Design z-gradient waveform, based on parameters in kp.
if ~isvar('gz'), printm 'Designing z gradient waveform...'
	[kp gz kz kf] = compute_gz_spsp(kp);
end

%Design complex-valued RF waveform iteratively using conjugate gradient
if ~isvar('b'), printm 'Computing RF pulse waveform...'
	[b] = compute_rf_spsp_mgh(kp, rfp, gz, kz, kf);

	%Write computed waveforms to files for simulation and/or scanner.
	write2files_spsp(kp, rfp, iop, gz, b);
end

%Perform Bloch simulation in SPSP space
if ~isvar('mresult'), printm 'Performing Bloch simulation...'
	[mresult] = dosim7_spsp(kp, rfp, iop);
end