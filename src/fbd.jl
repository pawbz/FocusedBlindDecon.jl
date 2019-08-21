"""
This package defines a `P_fbd` type to represent the FBD model, and provides a set of methods to access its properties.
In order to generate an instance of `P_fbd`, use the following command, where the description of the arguments 
and keywords is given below.
```julia
pa=P_fbd(ntg, nt, nr, nts; dobs, gobs, sobs)
```
# Arguments
Let `(ntg,nr)=size(g)`, `(nt,nr)=size(d)` and `(nts,)=size(s)`, where

**`ntg`** is input dimension of the channel impulse responses `g`. 

**`nt`** is input dimension of the channel outputs `d`. 

**`nr`** is the number of receivers or channels.

**`nts`** is input dimension of the source.

# Keywords

**`dobs`** are the channel responses that will be factorized. Alternatively, the user may input the next two keywords i.e.,
`gobs` and `sobs` for synthetic experiments, where `dobs` are internally generated.

**`gobs`** (optional) are the true channel impulse responses, stored in `pa`. 

**`sobs`** (optional) similarly, it is the true source.
"""
mutable struct P_fbd{T}
	pfibd::IBD{T}
	pfpr::FPR
	plsbd::BD{T}
	g::Matrix{T}
	gxcorr::Matrix{T}
	s::Vector{T}
	sa::Vector{T}
end

function P_fbd(ntg, nt, nr, nts;
	       dobs=nothing, 
	       gobs=nothing, 
	       sobs=nothing, 
	       sxp=Sxparam(1,:positive),
	       fmin=0.0,
	       fmax=0.5,
	       size_check=false,
	       ) 
	pfibd=IBD(ntg, nt, nr, nts, gobs=gobs, dobs=dobs, sobs=sobs, 
		  fft_threads=true, fftwflag=FFTW.MEASURE,
		  verbose=false, sxp=sxp, 
		  sx_fix_zero_lag_flag=true, fmin=fmin, fmax=fmax, size_check=size_check);

	pfpr=FPR(ntg, nr, )

	plsbd=BD(ntg, nt, nr, nts, dobs=dobs, gobs=gobs, sobs=sobs, 
	  sxp=sxp,
		 fft_threads=true, verbose=false, fftwflag=FFTW.MEASURE, size_check=size_check);

	return P_fbd(pfibd, pfpr, plsbd, plsbd.optm.cal.g, pfibd.optm.cal.g, 
	      plsbd.optm.cal.s,	pfibd.optm.cal.s)

end

"""
One can use the `lsbd!` method to perform LSBD over a given instance of `P_fbd` i.e., `pa`.
LBSD is a least-squares fitting of `d` to optimize the `g` and `s`, which can be accessed via 
`pa[:g]` and `pa[:s]`, respectively.
```julia
lsbd!(pa)
heatmap(pa[:g], title="estimated impulse responses from LSBD")
```
"""
function lsbd!(pa::P_fbd, io=stdout; args...) 
	bd!(pa.plsbd, io; args...)
	return nothing
end


"""
One can use the `fibd!` method to perform FIBD over a given instance of `P_fbd` i.e., `pa`.
FIBD is a least-squares fitting of `xd` to optimize the `xg` and `sa`, which can be accessed via 
`pa[:xg]` and `pa[:sa]`, respectively.
```julia
fibd!(pa)
heatmap(pa[:xg], title="estimated interferometric impulse responses from FIBD")
```
"""
function fibd!(pa::P_fbd, io=stdout)
	fibd_tol=[1e-10,1e-6]
	initialize!(pa.pfibd)
	fibd!(pa.pfibd, io, α=[Inf],tol=[fibd_tol[1]])
	fibd!(pa.pfibd, io, α=[0.0],tol=[fibd_tol[2]])

	# input g from fibd to fpr
	gobs = (iszero(pa.pfibd.om.g)) ? nothing : pa.pfibd.om.g # choose gobs for nearest receiver or not?
	update_cymat!(pa.pfpr; cymat=pa.pfibd.optm.cal.g, gobs=gobs)
	return nothing
end


"""
After performing FIBD on a `P_fbd` instance `pa`, we can perform FPR to complete FBD.
These two code blocks should be equivalent. 
```julia
fibd!(pa)
fpr!(pa)
```
```julia
fbd!(pa)
```
The result of FPR i.e, `g` can be extracted using `pa[:g]`.
The corresponding source signature is stored in `pa[:s]`.
"""
function fpr!(pa::P_fbd)
	# input g from fibd to fpr
	gobs = (iszero(pa.pfibd.om.g)) ? nothing : pa.pfibd.om.g # choose gobs for nearest receiver or not?
	update_cymat!(pa.pfpr; cymat=pa.pfibd.optm.cal.g, gobs=gobs)

	# perform fpr
	g=pa.plsbd.optm.cal.g
	Random.randn!(g)

	fill!(pa.pfpr.g, 0.0)
	#update_f_index_loaded!(pa.pfpr)
	fpr!(g,  pa.pfpr, 
        precon=[:focus, :pr], 
        #precon=[:pr], 
	#index_loaded=pa.pfpr.index_loaded, 
        index_loaded=1, 
			show_trace=true, g_tol=1e-4)

	# update S
	update!(pa.plsbd, pa.plsbd.sx.x, S(), optS)
	return nothing
end


"""
Perform FIBD and FPR to complete FBD of an instance of `P_fbd` i.e., `pa`.
```julia
fbd!(pa)
plot(pa[:g], title="estimated impulse responses using FBD")
plot(pa[:s], title="estimated source using FBD")
```
"""
function fbd!(pa::P_fbd, io=stdout; tasks=[:restart, :fibd, :fpr])

	if(:restart ∈ tasks)
		# initialize
		initialize!(pa.pfibd)
	end

	if(:fibd ∈ tasks)
		fibd!(pa)
	end

	if(:fpr ∈ tasks)
		fpr!(pa)
	end

	#=
	if(:updateS ∈ tasks)
		# update source according to the estimated g from fpr
		if(STF_FLAG)
			update_stf!(pa.plsbd)
		else
			update!(pa.plsbd, pa.plsbd.sx.x, S(), optS)
		end
	end
	=#

	# regular lsbd: do a few more AM steps? might diverge..
	#if(:lsbd ∈ tasks)
	#	bd!(pa.plsbd, io; tol=1e-5)
	#end

	return nothing
end



function random_problem()

	ntg=3
	nr=50
	tfact=10
	gobs=randn(ntg, nr)
	nt=ntg*tfact
	nts=nt-ntg+1;
	sobs=randn(nts)
	sxp=Sxparam(1,:positive)
	if(STF_FLAG)
		sobs=abs.(sobs);
	end
	return P_fbd(ntg, nt, nr, nts, gobs=gobs, sobs=sobs,sxp=sxp)
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
	return P_fbd(ntg, nt, nr, nts, gobs=gobs, sobs=sobs,sxp=Sxparam(1,:positive))
end

function simple_bandlimited_problem(fmin=0.1, fmax=0.4)
	pa=simple_problem()

	dobs=pa.plsbd.om.d
	responsetype = Bandpass(fmin,fmax; fs=1)
	designmethod = Butterworth(8)
	# zerophase filter please
	dobs_filt=DSP.Filters.filtfilt(digitalfilter(responsetype, designmethod), dobs)

	# filter dobs
	pom=pa.plsbd.om
	return P_fbd(pom.ntg, pom.nt, pom.nr, pom.nts, dobs=dobs_filt, 
	    gobs=pom.g, sobs=pom.s,sxp=Sxparam(1,:positive), fmin=fmin, fmax=fmax)

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
	return P_fbd(ntg, nt, nr, nts, gobs=gobs, sobs=sobs, sxp=Sxparam(2,:positive))
end
