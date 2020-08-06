mutable struct BDSVI{T}
	dobs::AbstractArray{T,3}
	dcal::AbstractArray{T,3}
	ϕ::Array{T,2}
	x::Array{T,2}
	is0::Int
	bdr::BD{T}
	bds::BD{T}
	opt::Solver
end


function BDSVI(dobs, is0, opt::Solver)
	nt,nr,ns=size(dobs)
	ntslag=nt
	T=eltype(dobs)

	
	bdr=FBD.BD(nt, nt, nr, 2*ntslag-1, dobs=randn(T, nt,nr), slags=[ntslag-1,ntslag-1], 
	    	fft_threads=true, fftwflag=FFTW.MEASURE);
	bds=FBD.BD(2*ntslag-1, nt, ns, nt, dobs=randn(T, nt,ns), glags=[ntslag-1,ntslag-1], 
	   slags=[nt-1, 0],
	    fft_threads=true, fftwflag=FFTW.MEASURE);

	ϕ=zeros(T,2*ntslag-1,ns)
	x=zeros(T,nt,nr)

	if(typeof(opt) == UseGradNMF)
		dobs=abs.(dobs)
	end

	pa=BDSVI(dobs,zero(dobs),ϕ,x,is0,bdr,bds,opt)

	initialize!(pa)
	return pa
end

# initialize
function initialize!(pa::BDSVI)
	nt,nr,ns=size(pa.dobs)
	ntslag=nt

	if((ns-pa.is0)*(1-pa.is0) ≤ 0)
		for ir in 1:nr, it in 1:nt
			pa.x[it,ir]=pa.dobs[it,ir,pa.is0]
		end
	else
		randn!(pa.x)
	end

	randn!(pa.ϕ)
	if(typeof(pa.opt) == UseGradNMF)
		pa.ϕ=1.0 .+ abs.(randn(size(pa.ϕ)))
	end

	if((ns-pa.is0)*(1-pa.is0) ≤ 0)
		for it in 1:2*ntslag-1
			pa.ϕ[it,pa.is0]=0
		end
		pa.ϕ[ntslag,pa.is0]=1.0
	end

	# compute other phi
	if(typeof(pa.opt) == UseGradNMF)
		for i in 1:10  # just perform 10 iterations, change later?!
			fitϕ!(pa, pa.opt)
		end
	else
		fitϕ!(pa, UseIterativeSolvers())
	end
	return pa
end


# for each receiver, update x
function fitx!(pa::BDSVI, opt::Solver)

	nt,nr,ns=size(pa.dobs)
	J0=0.0
	for ir in 1:nr
		d=pa.bds.optm.obs.d
		dobs=view(pa.dobs,:,ir,:)
		for i in eachindex(d)
			d[i]=dobs[i]
		end
		for i in eachindex(pa.ϕ)
			pa.bds.optm.cal.g[i]=pa.ϕ[i]
		end

		# initial value
		xs=view(pa.x,:,ir)
		for i in eachindex(xs)
			pa.bds.optm.cal.s[i]=xs[i]
		end

		J1=FBD.update!(pa.bds, pa.bds.sx.x, FBD.S(), opt)

		J0 += J1

		for is in 1:ns, it in 1:nt
			pa.dcal[it,ir,is]=pa.bds.optm.cal.d[it,is]
		end


		copyto!(xs,pa.bds.sx.x)

	end
	return J0
end

function fitϕ!(pa::BDSVI, opt::Solver)

	nt,nr,ns=size(pa.dobs)
	J0=0.0
	for iss in 1:ns
		d=pa.bdr.optm.obs.d
		dobs=view(pa.dobs,:,:,iss)
		for i in eachindex(d)
			d[i]=dobs[i]
		end
		for i in eachindex(pa.x)
			pa.bdr.optm.cal.g[i]=pa.x[i]
		end

		# initial value of ϕ
		ϕs=view(pa.ϕ,:,iss)
		for i in eachindex(pa.bdr.optm.cal.s)
			pa.bdr.optm.cal.s[i]=ϕs[i]
		end


		if(((pa.is0-ns)*(1-pa.is0) ≤ 0) & (iss == pa.is0))
			ntm=div(length(ϕs),2)
			for it in vcat(1:ntm, ntm+2:length(ϕs))
				ϕs[it]=0.0
			end
			J1=0.
		else
			J1 = FBD.update!(pa.bdr, pa.bdr.sx.x, FBD.S(), opt)
			copyto!(ϕs,pa.bdr.sx.x)

		end
		J0 += J1


		for ir in 1:nr, it in 1:nt
			pa.dcal[it,ir,iss]=pa.bdr.optm.cal.d[it,ir]
		end

	end
	return J0
end





function fit!(pa::BDSVI, io=stdout; 
		     max_roundtrips=100, 
		     max_reroundtrips=1, 
		     ParamAM_func=nothing, 
		     roundtrip_tol=1e-6,
		     optim_tols=[1e-6, 1e-6], verbose=true,
		     log_file=false,
		     )

	global to, optG, optS
	reset_timer!(to)

	if(ParamAM_func===nothing)
		ParamAM_func=xx->ParamAM(xx, optim_tols=optim_tols,name="Blind Decon",
				    roundtrip_tol=roundtrip_tol, max_roundtrips=max_roundtrips,
				    max_reroundtrips=max_reroundtrips,
				    min_roundtrips=10,
				    verbose=verbose,
				    reinit_func=xxx->initialize!(pa),
				    #after_roundtrip_func=x->(write(io,string(TimerOutputs.flatten(to),"\n"))),
				    #after_roundtrip_func=x->(err!(pa, nothing)),
				    )
	end

	# create alternating minimization parameters
	f1=x-> fitϕ!(pa, pa.opt)
	f2=x-> fitx!(pa, pa.opt)
	paam=ParamAM_func([f1, f2])

	if(io===nothing)
		logfilename=joinpath(pwd(),string("XBDSVI",Dates.now(),".log"))
		io=open(logfilename, "a+")
	end

	go(paam, io)  # run alternative minimization

	# save log file
	#if(log_file)
	#	AMlogfile=joinpath(pwd(),string("AM",Dates.now(),".log"))
	#	CSV.write(AMlogfile, paam.log)
	#end

	write(io,string(to))
	write(io, "\n")

	if(io === nothing)
		close(io)
	end
end


