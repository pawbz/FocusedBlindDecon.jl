# blind deconvolution
module DeConv


using Inversion
using Misfits
using Signals
using Conv
using Grid
using Optim
using Ipopt
using RecipesBase
using DataFrames
using StatsBase
using IterativeSolvers
using LinearMaps
using CSV
using DSP
using FFTW
using LinearAlgebra
using Dates


function hello()
	println("FFFFKKF")
end

include("DataTypes.jl")
include("FPR.jl")
include("Phase.jl")
include("BD.jl")
include("IBD.jl")
include("FBD.jl")
include("Misfits.jl")
include("Doppler.jl")
include("Deterministic.jl")




function add_gprecon!(pa, gprecon)

	copyto!(pa.gx.precon, gprecon)		
	copyto!(pa.gx.preconI, pa.gx.precon)
	for i in eachindex(gprecon)
		if(!(iszero(gprecon[i])))
			pa.gx.preconI[i]=inv(pa.gx.precon[i])
		end
	end
end
function add_sprecon!(pa, sprecon)
	copyto!(pa.sx.precon, sprecon)
	copyto!(pa.sx.preconI, pa.sx.precon)
	for i in eachindex(sprecon)
		if(!(iszero(sprecon[i])))
			pa.sx.preconI[i]=inv(pa.sx.precon[i])
		end
	end
end
function add_gweights!(pa, gweights)
	(gweights===nothing) && (gweights=ones(pa.optm.cal.g))
	copyto!(pa.gx.weights, gweights)
end


#=

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


	copyto!(pa.optm.cal.s, ssave)
	copyto!(pa.optm.cal.g, gsave)
	copyto!(pa.optm.cal.d,dcalsave)

	return pa
end
=#

struct Use_Optim end
struct Use_Ipopt end

# core algorithm
function update!(pa, x, ::Use_Optim) 

	f =x->func_grad!(nothing, x,  pa) 
	g! =(storage, x)->func_grad!(storage, x,  pa)

	# put the value from model to x
	model_to_x!(x, pa)

	"""
	Unbounded LBFGS inversion, only for testing
	"""
	res = optimize(f, g!, x, 
		ConjugateGradient(),
		Optim.Options(g_tol = 1e-8, iterations = 2000, show_trace = false))
	pa.verbose && println(res)

	x_to_model!(Optim.minimizer(res), pa)

	return res
end

# core algorithm
function update!(pa, x) 
	nx=length(x)
	eval_f=x->func_grad!(nothing, x,  pa) 
	eval_grad_f=(x, storage)->func_grad!(storage, x,  pa)

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
	model_to_x!(x, pa)

	# put initial
	copyto!(prob.x,x)
		    
	res=solveProblem(prob)
	println("ffff")

	#println(typeof(res), "\tFFFFFFFVVVVV\t", res)

	# take solution out
	copyto!(x,prob.x)

	return prob.obj_val
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
function update_all!(pa, io=stdout; 
		     max_roundtrips=100, 
		     max_reroundtrips=1, 
		     ParamAM_func=nothing, 
		     roundtrip_tol=1e-6,
		     optim_tols=[1e-6, 1e-6], verbose=true,
		     )

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
	f1=x->update_s!(pa, pa.sx.x)
	f2=x->update_g!(pa, pa.gx.x)
	paam=ParamAM_func([f1, f2])

	if(io===nothing)
		logfilename=joinpath(pwd(),string("XBD",Dates.now(),".log"))
		io=open(logfilename, "a+")
	end
	Inversion.go(paam, io)  # run alternative minimization

	# print errors
	err!(pa, io)
	write(io, "\n")
	if(io === nothing)
		close(io)
	end
end

"""
update calsave only when error in d is low
"""
function update_calsave!(optm::OptimModel, calsave)
	f1=Misfits.error_squared_euclidean!(nothing, calsave.d, optm.obs.d, nothing, norm_flag=true)
	f2=Misfits.error_squared_euclidean!(nothing, optm.cal.d, optm.obs.d, nothing, norm_flag=true)
	if(f2<f1)
		copyto!(calsave.d, optm.cal.d)
		copyto!(calsave.g, optm.cal.g)
		copyto!(calsave.s, optm.cal.s)
	end
end



function update_window!(paf::FourierConstraints, obs)
	pac=obs
	paf=paf
	Conv.pad!(pac.d, pac.dpad, pac.dlags[1], pac.dlags[2], pac.np2)
	mul!(pac.dfreq, pac.dfftp, pac.dpad)
	for i in eachindex(paf.window)
		paf.window[i]=complex(0.0,0.0)
	end

	nr=size(pac.dfreq,2)
	# stack the spectrum of data
	for j in 1:nr
		for i in eachindex(paf.window)
			paf.window[i] += (abs2(pac.dfreq[i,j]))
		end
	end

	normalize!(paf.window)
	paf.window = 10. * log10.(paf.window)

	# mute frequencies less than -40 dB
	for i in eachindex(paf.window)
		if(paf.window[i] ≤ -40)
			paf.window[i]=0.0
		else
			paf.window[i]=1.0
		end
	end
end

function apply_window_s!(s, pac, paf)
	Conv.pad!(s, pac.spad, pac.slags[1], pac.slags[2], pac.np2)
	mul!(pac.sfreq, pac.sfftp, pac.spad)
	for i in eachindex(pac.sfreq)
		pac.sfreq[i] *= paf.window[i]
	end
	mul!(pac.spad, pac.sifftp, pac.sfreq)
	Conv.truncate!(s, pac.spad, pac.slags[1], pac.slags[2], pac.np2)
end


function apply_window_g!(g, pac, paf)
	Conv.pad!(g, pac.gpad, pac.glags[1], pac.glags[2], pac.np2)
	mul!(pac.gfreq, pac.gfftp, pac.gpad)
	nr=size(pac.gfreq,2)
	for ir in 1:nr
		for i in size(pac.gfreq,1)
			pac.gfreq[i,ir] *= paf.window[i]
		end
	end
	mul!(pac.gpad, pac.gifftp, pac.gfreq)
	Conv.truncate!(g, pac.gpad, pac.glags[1], pac.glags[2], pac.np2)

end
include("Save.jl")
include("Plots.jl")


end # module
