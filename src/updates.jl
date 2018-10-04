

# core algorithm using IterativeSolvers.jl
function update!(pa, x, attrib, ::UseIterativeSolvers)
	update_prepare!(pa, attrib)

	# prepare b
	for i in eachindex(pa.optm.dvec)
		pa.optm.dvec[i]=pa.optm.obs.d[i]
	end

	# create operator
	A=operator(pa, attrib);

	# put the value from model to x
	model_to_x!(x, pa, attrib)

	IterativeSolvers.lsmr!(x, A, pa.optm.dvec)

	x_to_model!(x, pa, attrib)

	J=func_grad!(nothing, x,  pa, attrib)

	update_finalize!(pa, attrib)
	 
	return J
end


# use Optim.jl (ConjugateGradients)
function update!(pa, x, attrib, ::UseOptim) 
	update_prepare!(pa, attrib)

	f =x->func_grad!(nothing, x,  pa, attrib) 
	g! =(storage, x)->func_grad!(storage, x,  pa, attrib)

	# put the value from model to x
	model_to_x!(x, pa, attrib)

	res = optimize(f, g!, 
		x, ConjugateGradient(alphaguess = LineSearches.InitialQuadratic()),
		Optim.Options(g_tol = 1e-8, iterations = 2000, show_trace = false))
	pa.verbose && println(res)

	x_to_model!(Optim.minimizer(res), pa, attrib)

	update_finalize!(pa, attrib)

	return Optim.minimum(res)
end


"""
* re_init_flag :: re-initialize inversions with random input or not?
"""
function update_all!(pa, io=stdout; 
		     max_roundtrips=100, 
		     max_reroundtrips=1, 
		     ParamAM_func=nothing, 
		     roundtrip_tol=1e-6,
		     optim_tols=[1e-6, 1e-6], verbose=true,
		     )

	global to, optG, optS
	reset_timer!(to)

	if(ParamAM_func===nothing)
		ParamAM_func=xx->Inversion.ParamAM(xx, optim_tols=optim_tols,name="Blind Decon",
				    roundtrip_tol=roundtrip_tol, max_roundtrips=max_roundtrips,
				    max_reroundtrips=max_reroundtrips,
				    min_roundtrips=10,
				    verbose=verbose,
				    reinit_func=xxx->initialize!(pa),
				    after_roundtrip_func=x->(write(io,string(TimerOutputs.flatten(to),"\n"))),
				    )
	end

	
	# create alternating minimization parameters
	f1=x->@timeit to "update S()" update!(pa, pa.sx.x, S(), optS)
	f2=x->@timeit to "update G()" update!(pa, pa.gx.x, G(), optG)
	paam=ParamAM_func([f1, f2])

	if(io===nothing)
		logfilename=joinpath(pwd(),string("XBD",Dates.now(),".log"))
		io=open(logfilename, "a+")
	end

	Inversion.go(paam, io)  # run alternative minimization

	write(io,string(to))
	write(io, "\n")

	# print errors
	err!(pa, io)
	write(io, "\n")
	if(io === nothing)
		close(io)
	end
end


