
# project x onto R⁺
function prox!(x)
	for i in eachindex(x)
		x[i] = max(x[i], 0.0)
	end
end

include("customoptimizer.jl")

# take a single step in the gradient direction using LineSearches.jl
function update!(pa, x, attrib, opt::Union{UseGradNMF,UseGrad})
	update_prepare!(pa, attrib)

	# prepare b
	for i in eachindex(pa.optm.dvec)
		pa.optm.dvec[i]=pa.optm.obs.d[i]
	end

	# put the value from model to x
	model_to_x!(x, pa, attrib)

	f =x1->func_grad!(nothing, x1,  pa, attrib) 
	g! = (g,x1)->func_grad!(g, x1,  pa, attrib)
	fg! = (g,x1)->func_grad!(g, x1,  pa, attrib)

	if(typeof(opt) == UseGradNMF)
		gdoptimize(f, g!, fg!, x, true)
	else
		gdoptimize(f, g!, fg!, x, false)
	end

	x_to_model!(x, pa, attrib)

	J=func_grad!(nothing, x,  pa, attrib)

	update_finalize!(pa, attrib)
	 
	return J
end




# take a single step in the gradient direction using LineSearches.jl
function update_old!(pa, x, attrib, ::UseGradNMF)
	update_prepare!(pa, attrib)

	# prepare b
	for i in eachindex(pa.optm.dvec)
	#	pa.optm.dvec[i]=pa.optm.obs.d[i]
	end

	# put the value from model to x
	model_to_x!(x, pa, attrib)

	xx = copy(x);
	gvec = similar(x);
	f =x->func_grad!(nothing, x,  pa, attrib) 
	f0=func_grad!(gvec, x,  pa, attrib)

	alpha=1.0
	min_stepsize=1e-10
	stepsize=1.0
	
	while alpha > min_stepsize
#		println("trying...",alpha)
		stepsize=alpha
		axpy!(-stepsize,gvec,xx)
	#	prox!(xx)
		if f(xx) < f0
			copyto!(x, xx)
			alpha *= 1.05
			break                
		else # the stepsize was too big; undo and try again only smaller                    
			copyto!(xx, x)
			alpha *= .7                  
			if alpha < min_stepsize               
				alpha = min_stepsize * 1.1    
				break         
			end      
		end
	end

	x_to_model!(x, pa, attrib)

	update_finalize!(pa, attrib)
	 
	return f(x)
end


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

	# put the value from model to x
	model_to_x!(x, pa, attrib)

	f =x->func_grad!(nothing, x,  pa, attrib) 
	g! =(storage, x)->func_grad!(storage, x,  pa, attrib)


	res = optimize(f, g!, 
		x, ConjugateGradient(alphaguess = LineSearches.InitialQuadratic()),
		Optim.Options(g_tol = 1e-8, iterations = 2000, show_trace = false))

	pa.verbose && println(res)

	# extract result
	xx=Optim.minimizer(res)
	x_to_model!(xx, pa, attrib) # put result in pa
	copyto!(x,xx) # put result in x

	update_finalize!(pa, attrib)

	return Optim.minimum(res)
end

# take a gradient step and then project
function update!(pa, x, attrib, ) 
	update_prepare!(pa, attrib)

	# put the value from model to x
	model_to_x!(x, pa, attrib)

	function f(x)
#		for xi in x
#			if(xi < 0)
#				return Inf
#			end
#		end
		# project on R+
		for i in eachindex(x)
			x[i] = max(x[i], 0.0)
		#	x[i] = xx[i]
		end

		return func_grad!(nothing, x,  pa, attrib) 
	end

	g! =(storage, x)->func_grad!(storage, x,  pa, attrib)

	res = optimize(f, g!, 
		x, 
		GradientDescent(linesearch = LineSearches.BackTracking()),
		Optim.Options(g_tol = 1e-12,
		iterations = 2, ))

	pa.verbose && println(res)

	# extract result
	xx=Optim.minimizer(res)
	x_to_model!(xx, pa, attrib) # put result in pa

	# project on R+
	for i in eachindex(x)
		x[i] = max(xx[i], 0.0)
#		x[i] = xx[i]
	end

	update_finalize!(pa, attrib)

	return f(x)
	#Optim.minimum(res)
end



#=
# use bounded Optim.jl (ConjugateGradients) (for positive Source Time Functions)
function update!(pa, x, attrib, ::UseOptimSTF) 

	if(typeof(attrib) ≠ S)
		error("bounded Optim only for S")
	end

	fill!(pa.sx.lower_x, 0.0)
	fill!(pa.sx.upper_x, Inf)


	update_prepare!(pa, attrib)


	f =x->func_grad!(nothing, x,  pa, attrib) 
	g! =(storage, x)->func_grad!(storage, x,  pa, attrib)
	println("FFFFAA", maximum(x))

	# put the value from model to x
	model_to_x!(x, pa, attrib)

	# check if x<0?
	for i in eachindex(x)
		if(x[i] < 0.0)
			x[i] = 0.0
		end
	end

	println("FFFF", maximum(x))


	res = optimize(f, g!, pa.sx.lower_x, pa.sx.upper_x, 
		x, Fminbox(ConjugateGradient(alphaguess = LineSearches.InitialQuadratic())),
		Optim.Options(g_tol = 1e-8, iterations = 2000, show_trace = false))
	pa.verbose && println(res)

	x_to_model!(Optim.minimizer(res), pa, attrib)

	update_finalize!(pa, attrib)

	return Optim.minimum(res)
end
=#



"""
* re_init_flag :: re-initialize inversions with random input or not?
"""
function update_all!(pa, io=stdout; 
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
	f1=x->@timeit to "update S()" update!(pa, pa.sx.x, S(), optS)
	f2=x->@timeit to "update G()" update!(pa, pa.gx.x, G(), optG)
	paam=ParamAM_func([f1, f2])

	if(io===nothing)
		logfilename=joinpath(pwd(),string("XBD",Dates.now(),".log"))
		io=open(logfilename, "a+")
	end

	go(paam, io)  # run alternative minimization

	# save log file
	if(log_file)
		AMlogfile=joinpath(pwd(),string("AM",Dates.now(),".log"))
		CSV.write(AMlogfile, paam.log)
	end

	write(io,string(to))
	write(io, "\n")

	# print errors
	err!(pa, io)
	write(io, "\n")
	if(io === nothing)
		close(io)
	end
end




"""
Given g (sign is ambiguous), update s such that s>0
"""
function update_stf!(pa::BD,)
	#(pa.sxp.n ≠ 2) && error("stf update not possible")
	# update source according to the estimated g from fpr
	update_prepare!(pa, S())
	update!(pa, pa.sx.x, S(), optS)
	update_finalize!(pa, S())
	# save functional and estimated source
	f1 = Misfits.error_squared_euclidean!(nothing, pa.optm.cal.d, pa.optm.obs.d, 
				       nothing, norm_flag=false)
	s1 = copy(pa.optm.cal.s)


	# what if the sign of g is otherwise
	rmul!(pa.optm.cal.g, -1.0)
	update_prepare!(pa, S())
	update!(pa, pa.sx.x, S(), optS)
	update_finalize!(pa, S())
	f2 = Misfits.error_squared_euclidean!(nothing, pa.optm.cal.d, pa.optm.obs.d, 
			       nothing, norm_flag=false)
	s2 = copy(pa.optm.cal.s)

	# sxp is not used here, because the estimated g after fpr has ambiguous sign
	# as a result, positivity constraint cannot be imposed on s
	if(f1<f2)
		copyto!(pa.optm.cal.s, s1)
		rmul!(pa.optm.cal.g, -1.0) # change sign back
	else
		# no need to change sign of g
		copyto!(pa.optm.cal.s, s2)
	end

	return nothing
end
