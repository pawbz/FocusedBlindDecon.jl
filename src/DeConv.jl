# blind deconvolution
__precompile__()

module DeConv
using Inversion
using Misfits
using Conv
using Grid
using Optim, LineSearches
using RecipesBase
using DataFrames
using StatsBase
using JLD
using CSV
using DSP.nextfastfft
using ProgressMeter

include("DataTypes.jl")
include("FPR.jl")
include("Phase.jl")
include("BD.jl")
include("IBD.jl")
include("Misfits.jl")
include("Doppler.jl")
include("Deterministic.jl")

function add_gprecon!(pa, gprecon)

	copy!(pa.gx.precon, gprecon)		
	copy!(pa.gx.preconI, pa.gx.precon)
	for i in eachindex(gprecon)
		if(!(iszero(gprecon[i])))
			pa.gx.preconI[i]=inv(pa.gx.precon[i])
		end
	end
end
function add_sprecon!(pa, sprecon)
	copy!(pa.sx.precon, sprecon)
	copy!(pa.sx.preconI, pa.sx.precon)
	for i in eachindex(sprecon)
		if(!(iszero(sprecon[i])))
			pa.sx.preconI[i]=inv(pa.sx.precon[i])
		end
	end
end
function add_gweights!(pa, gweights)
	(gweights===nothing) && (gweights=ones(pa.optm.cal.g))
	copy!(pa.gx.weights, gweights)
end



function update_func_grad!(pa; goptim=nothing, soptim=nothing, gαvec=nothing, sαvec=nothing)
	# they will be changed in this program, so make a copy 
	ssave=copy(pa.optm.cal.s);
	gsave=copy(pa.optm.cal.g);
	dcalsave=copy(pa.optm.cal.d);

	(goptim===nothing) && (goptim=[:ls])
	(gαvec===nothing) && (gαvec=ones(length(goptim)))

	(soptim===nothing) && (soptim=[:ls])
	(sαvec===nothing) && (sαvec=ones(length(soptim)))

	# dfg for optimization functions
	optim_funcg=Vector{Function}(length(goptim))
	optim_gradg=Vector{Function}(length(goptim))
	for iop in 1:length(goptim)
		if (goptim[iop]==:ls)
			optim_funcg[iop]= x->func_grad!(nothing, x,  pa) 
			optim_gradg[iop]=(storage, x)->func_grad!(storage, x,  pa)
		elseif(goptim[iop]==:weights)
			optim_funcg[iop]= x -> func_grad_g_weights!(nothing, x, pa) 
			optim_gradg[iop]= (storage, x) -> func_grad_g_weights!(storage, x, pa)
		elseif(goptim[iop]==:acorr_weights)
			optim_funcg[iop]= x -> func_grad_g_acorr_weights!(nothing, x, pa) 
			optim_gradg[iop]= (storage, x) -> func_grad_g_acorr_weights!(storage, x, pa)
		else
			error("invalid optim_funcg")
		end
	end
	pa.attrib_inv=:g
	# multi-objective framework
	paMOg=Inversion.ParamMO(noptim=length(goptim), ninv=length(pa.gx.x), αvec=gαvec,
			    		optim_func=optim_funcg,optim_grad=optim_gradg,
					x_init=randn(length(pa.gx.x),10))
	# create dfg for optimization
	pa.gx.func = x -> paMOg.func(x, paMOg)
	pa.gx.grad! = (storage, x) -> paMOg.grad!(storage, x, paMOg)
#	pa.dfg = OnceDifferentiable(x -> paMOg.func(x, paMOg),       
#			    (storage, x) -> paMOg.grad!(storage, x, paMOg), )


	# dfs for optimization functions
	optim_funcs=Vector{Function}(length(soptim))
	optim_grads=Vector{Function}(length(soptim))
	for iop in 1:length(soptim)
		if (soptim[iop]==:ls)
			optim_funcs[iop]=x->func_grad!(nothing, x,  pa) 
			optim_grads[iop]=(storage, x)->func_grad!(storage, x,  pa) 
		else
			error("invalid optim_funcs")
		end
	end

	pa.attrib_inv=:s
	# multi-objective framework
	paMOs=Inversion.ParamMO(noptim=length(soptim), ninv=length(pa.sx.x), αvec=sαvec,
			    		optim_func=optim_funcs,optim_grad=optim_grads,
					x_init=vcat(ones(1,10),randn(length(pa.sx.x)-1,10)))
#	pa.dfs = OnceDifferentiable(x -> paMOs.func(x, paMOs),         
#			    (storage, x) -> paMOs.grad!(storage, x, paMOs))
	pa.sx.func = x -> paMOs.func(x, paMOs)
	pa.sx.grad! =  (storage, x) -> paMOs.grad!(storage, x, paMOs)


	copy!(pa.optm.cal.s, ssave)
	copy!(pa.optm.cal.g, gsave)
	copy!(pa.optm.cal.d,dcalsave)

	return pa
end


# core algorithm
function update!(pa, x, f, g!; 
		 store_trace::Bool=false, 
		 extended_trace::Bool=false, 
	     f_tol::Float64=1e-8, g_tol::Float64=1e-30, x_tol::Float64=1e-30, iterations=2000)

	# initial w to x
	model_to_x!(x, pa)

	"""
	Unbounded LBFGS inversion, only for testing
	"""
	#nlprecon =  GradientDescent(linesearch=LineSearches.Static(alpha=1e-4,scaled=true))
	#oacc10 = OACCEL(nlprecon=nlprecon, wmax=10)
	res = optimize(f, g!, x, 
#		oacc10,
		ConjugateGradient(),
		       #BFGS(),
		       Optim.Options(g_tol = g_tol, f_tol=f_tol, x_tol=x_tol,
		       iterations = iterations, store_trace = store_trace,
		       extended_trace=extended_trace, show_trace = false))
	pa.verbose && println(res)

	x_to_model!(Optim.minimizer(res), pa)

	return res
end


"""
Remove preconditioners from pa
"""
function remove_gprecon!(pa; including_zeros=false)
	for i in eachindex(pa.gx.precon)
		if((pa.gx.precon[i]≠0.0) || including_zeros)
			pa.gx.precon[i]=1.0
			pa.gx.preconI[i]=1.0
		end
	end
end

"""
Remove weights from pa
"""
function remove_gweights!(pa; including_zeros=false)
	for i in eachindex(pa.gx.weights)
		if((pa.gx.weights[i]≠0.0) || including_zeros)
			pa.gx.weights[i]=1.0
		end
	end
end


"""
* re_init_flag :: re-initialize inversions with random input or not?
"""
function update_all!(pa; max_roundtrips=100, max_reroundtrips=10, ParamAM_func=nothing, roundtrip_tol=1e-6,
		     optim_tols=[1e-6, 1e-6], verbose=true, )

	if(ParamAM_func===nothing)
		ParamAM_func=x->Inversion.ParamAM(x, optim_tols=optim_tols,name="Blind Decon",
				    roundtrip_tol=roundtrip_tol, max_roundtrips=max_roundtrips,
				    max_reroundtrips=max_reroundtrips,
				    min_roundtrips=10,
				    verbose=verbose,
				    reinit_func=x->initialize!(pa),
				    after_reroundtrip_func=x->(err!(pa);),
				    )
	end

	
	# create alternating minimization parameters
	f1=x->update_s!(pa, pa.sx.x)
	f2=x->update_g!(pa, pa.gx.x)
	paam=ParamAM_func([f1, f2])

	# do inversion
	Inversion.go(paam)

	# print errors
	err!(pa)
	println(" ")
end

"""
update calsave only when error in d is low
"""
function update_calsave!(optm::OptimModel, calsave)
	f1=Misfits.error_squared_euclidean!(nothing, calsave.d, optm.obs.d, nothing, norm_flag=true)
	f2=Misfits.error_squared_euclidean!(nothing, optm.cal.d, optm.obs.d, nothing, norm_flag=true)
	if(f2<f1)
		copy!(calsave.d, optm.cal.d)
		copy!(calsave.g, optm.cal.g)
		copy!(calsave.s, optm.cal.s)
	end
end




include("Save.jl")
include("Plots.jl")


end # module
