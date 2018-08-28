
mutable struct FBD
	pfibd::IBD
	pfpr::FPR
	plsbd::BD
end


function FBD(ntg, nt, nr, nts;
	       dobs=nothing, 
	       gobs=nothing, 
	       sobs=nothing, 
	       ) 
	pfibd=IBD(ntg, nt, nr, nts, gobs=gobs, dobs=dobs, sobs=sobs, 
		  fft_threads=true, fftwflag=FFTW.MEASURE,
		  verbose=false, sx_attrib=:positive, sx_fix_zero_lag_flag=true);

	pfpr=FPR(ntg, nr)

	plsbd=BD(ntg, nt, nr, nts, dobs=dobs, gobs=gobs, sobs=sobs, 
		 fft_threads=true, verbose=false, fftwflag=FFTW.MEASURE);

	return FBD(pfibd, pfpr, plsbd)

end


function fbd!(pa::FBD, io=stdout)

	# initialize
	DeConv.initialize!(pa.pfibd)

	# start with fibd
	fibd!(pa.pfibd, io, α=[Inf,0.0],tol=[1e-8,1e-3])

	# input g from fibd to fpr
	gobs = (iszero(pa.pfibd.om.g)) ? nothing : pa.pfibd.om.g # choose gobs for nearest receiver or not?
	update_cymat!(pa.pfpr; cymat=pa.pfibd.optm.cal.g, gobs=gobs)

	# perform fpr
	g=pa.plsbd.optm.cal.g
	Random.randn!(g)

	fpr!(g,  pa.pfpr, precon=:focus)

	# do regular bd
	bd!(pa.plsbd, io; tol=1e-3)

	return pa
end



function simple_problem()

	ntg=30
	nr=20
	tfact=20
	gobs=zeros(ntg, nr)
	Signals.toy_direct_green!(gobs, c=4.0, bfrac=0.20, afrac=1.0);
	#Signals.toy_direct_green!(gobs, c=4.0, bfrac=0.4, afrac=1.0);
	Signals.toy_reflec_green!(gobs, c=1.5, bfrac=0.35, afrac=-0.6);
	#Signals.toy_reflec_green!(gobs, c=1.5, bfrac=0.35, afrac=1.0);i
	nt=ntg*tfact
	nts=nt-ntg+1;
	sobs=randn(nts);
	return FBD(ntg, nt, nr, nts, gobs=gobs, sobs=sobs)
end
