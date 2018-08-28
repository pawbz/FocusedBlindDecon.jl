


struct Use_Optim end
struct Use_Ipopt end
struct Use_IterativeSolvers end

# core algorithm
function update!(pa, x, attrib,)#::Use_IterativeSolvers) 
	update_prepare!(pa, attrib)

	# prepare b
	for i in eachindex(pa.optm.dvec)
		pa.optm.dvec[i]=pa.optm.obs.d[i]
	end

	# create operator
	A=operator(pa, attrib);

	# put the value from model to x
	model_to_x!(x, pa, attrib)

	IterativeSolvers.lsqr!(x, A, pa.optm.dvec, damp=0.0)

	x_to_model!(x, pa, attrib)

	J=func_grad!(nothing, x,  pa, attrib)

	update_finalize!(pa, attrib)
	 
	return J
end


# core algorithm
function update!(pa, x, attrib, ::Use_Optim) 
	update_prepare!(pa, attrib)

	f =x->func_grad!(nothing, x,  pa, attrib) 
	g! =(storage, x)->func_grad!(storage, x,  pa, attrib)

	# put the value from model to x
	model_to_x!(x, pa, attrib)

	"""
	Unbounded LBFGS inversion, only for testing
	"""
	res = optimize(f, g!, x, 
		ConjugateGradient(alphaguess = LineSearches.InitialQuadratic(),
		    ),
		Optim.Options(g_tol = 1e-8, iterations = 2000, show_trace = false))
	pa.verbose && println(res)

	x_to_model!(Optim.minimizer(res), pa, attrib)

	update_finalize!(pa, attrib)

	return Optim.minimum(res)
end

# core algorithm
function update!(pa, x, attrib, ::Use_Ipopt) 
	update_prepare!(pa, attrib)
	nx=length(x)
	eval_f=x->func_grad!(nothing, x,  pa, attrib) 
	eval_grad_f=(x, storage)->func_grad!(storage, x,  pa, attrib)

	function void_g(x, g) end
	function void_g_jac(x, mode, rows, cols, values) end

	prob = createProblem(nx, 
		      fill(-Inf, nx),
		      fill(Inf, nx),
		      #Array{Float64}(0), # lower_x
		      #Array{Float64}(0), # upper_x
		      0, Array{Float64}(0), Array{Float64}(0), 0, 0,
			eval_f, void_g, eval_grad_f, void_g_jac, nothing)

	addOption(prob, "hessian_approximation", "limited-memory")
	addOption(prob, "print_level", 0)
	#addOption(prob, "derivative_test", "first-order")
	#addOption(prob, "max_iter", 0)

	# put the value from model to x
	model_to_x!(x, pa, attrib)

	# put initial
	copyto!(prob.x,x)
		    
	res=solveProblem(prob)

	# take solution out
	copyto!(x,prob.x)
	x_to_model!(x, pa, attrib)

	update_finalize!(pa, attrib)

	return prob.obj_val
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

	global to
	reset_timer!(to)

	if(ParamAM_func===nothing)
		ParamAM_func=xx->Inversion.ParamAM(xx, optim_tols=optim_tols,name="Blind Decon",
				    roundtrip_tol=roundtrip_tol, max_roundtrips=max_roundtrips,
				    max_reroundtrips=max_reroundtrips,
				    min_roundtrips=10,
				    verbose=verbose,
				    reinit_func=xxx->initialize!(pa),
		#		    after_reroundtrip_func=x->(err!(pa);),
				    )
	end

	
	# create alternating minimization parameters
	f1=x->@timeit to "update G()" update!(pa, pa.gx.x, G())
	f2=x->@timeit to "update S()" update!(pa, pa.sx.x, S())
	paam=ParamAM_func([f1, f2])

	if(io===nothing)
		logfilename=joinpath(pwd(),string("XBD",Dates.now(),".log"))
		io=open(logfilename, "a+")
	end

	Inversion.go(paam, io)  # run alternative minimization

	println(to)

	# print errors
	err!(pa, io)
	write(io, "\n")
	if(io === nothing)
		close(io)
	end
end


