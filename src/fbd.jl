
mutable struct FBD{T}
	pfibd::IBD{T}
	pfpr::FPR
	plsbd::BD{T}
end


function FBD(ntg, nt, nr, nts;
	       dobs=nothing, 
	       gobs=nothing, 
	       sobs=nothing, 
	       sxp=Sxparam(1,:positive)
	       ) 
	pfibd=IBD(ntg, nt, nr, nts, gobs=gobs, dobs=dobs, sobs=sobs, 
		  fft_threads=true, fftwflag=FFTW.MEASURE,
		  verbose=false, sxp=sxp, sx_fix_zero_lag_flag=true);

	pfpr=FPR(ntg, nr)

	plsbd=BD(ntg, nt, nr, nts, dobs=dobs, gobs=gobs, sobs=sobs, 
	  sxp=sxp,
		 fft_threads=true, verbose=false, fftwflag=FFTW.MEASURE);

	return FBD(pfibd, pfpr, plsbd)

end


function fbd!(pa::FBD, io=stdout; tasks=[:restart, :fibd, :fpr, :updateS])

	if(:restart ∈ tasks)
		# initialize
		DeConv.initialize!(pa.pfibd)
	end

	if(:fibd ∈ tasks)
		# start with fibd
		fibd!(pa.pfibd, io, α=[Inf,0.0],tol=[1e-10,1e-6])
	end

	# input g from fibd to fpr
	gobs = (iszero(pa.pfibd.om.g)) ? nothing : pa.pfibd.om.g # choose gobs for nearest receiver or not?
	update_cymat!(pa.pfpr; cymat=pa.pfibd.optm.cal.g, gobs=gobs)

	# perform fpr
	g=pa.plsbd.optm.cal.g
	Random.randn!(g)

	if(:fpr ∈ tasks)
		fpr!(g,  pa.pfpr, precon=:focus)
	end

	if(:updateS ∈ tasks)
		# update source according to the estimated g from fpr
		if(pa.plsbd.sxp.n == 2)
			update_stf!(pa.plsbd)
		else
			update!(pa.plsbd, pa.plsbd.sx.x, S(), optS)
		end
	end

	# regular lsbd: do a few more AM steps? might diverge..
	if(:lsbd ∈ tasks)
		bd!(pa.plsbd, io; tol=1e-5)
	end

	return nothing
end



function random_problem(;stf=false)

	ntg=3
	nr=50
	tfact=10
	gobs=randn(ntg, nr)
	nt=ntg*tfact
	nts=nt-ntg+1;
	sobs=randn(nts)
	sxp=Sxparam(1,:positive)
	if(stf)
		sobs=abs.(sobs);
		sxp=Sxparam(2,:positive)
	end
	return FBD(ntg, nt, nr, nts, gobs=gobs, sobs=sobs,sxp=sxp)
end


function simple_problem()

	ntg=30
	nr=20
	tfact=20
	gobs=zeros(ntg, nr)
	Signals.toy_direct_green!(gobs, c=4.0, bfrac=0.20, afrac=1.0);
	#Signals.toy_direct_green!(gobs, c=4.0, bfrac=0.4, afrac=1.0);
	Signals.toy_reflec_green!(gobs, c=1.5, bfrac=0.35, afrac=-0.6);
	#Signals.toy_reflec_green!(gobs, c=1.5, bfrac=0.35, afrac=1.0);
	nt=ntg*tfact
	nts=nt-ntg+1;
	sobs=(randn(nts));
	return FBD(ntg, nt, nr, nts, gobs=gobs, sobs=sobs,sxp=Sxparam(1,:positive))
end

#Source Time Functions are always positive
function simple_STF_problem()

	ntg=30
	nr=20
	tfact=20
	gobs=zeros(ntg, nr)
	Signals.toy_direct_green!(gobs, c=4.0, bfrac=0.20, afrac=1.0);
	#Signals.toy_direct_green!(gobs, c=4.0, bfrac=0.4, afrac=1.0);
	Signals.toy_reflec_green!(gobs, c=1.5, bfrac=0.35, afrac=-0.6);
	#Signals.toy_reflec_green!(gobs, c=1.5, bfrac=0.35, afrac=1.0);
	nt=ntg*tfact
	nts=nt-ntg+1;
	sobs=abs.(randn(nts));
	return FBD(ntg, nt, nr, nts, gobs=gobs, sobs=sobs, sxp=Sxparam(2,:positive))
end
