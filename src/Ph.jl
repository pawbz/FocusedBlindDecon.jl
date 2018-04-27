


mutable struct ParamP
	ntg::Int64
	nt::Int64
	nr::Int64
	obs1::Conv.Param{Float64,2,2,1} # observed convolutional model
	cal1::Conv.Param{Float64,2,2,1} # calculated convolutional model
	obs2::Conv.Param{Float64,2,1,2} # observed convolutional model
	cal2::Conv.Param{Float64,2,1,2} # calculated convolutional model
	dg::Array{Float64,2}
	ds::Vector{Float64}
	ddcal::Array{Float64,2}
	attrib_inv::Symbol
	verbose::Bool
	xg::Vector{Float64}
	last_xg::Vector{Float64}
	g_func::Function
	g_grad!::Function
	xs::Vector{Float64}
	last_xs::Vector{Float64}
	s_func::Function
	s_grad!::Function
	err::DataFrames.DataFrame
end



"""
`gprecon` : a preconditioner applied to each Greens functions [ntg]
"""
function ParamP(ntg, nt, nr; 
	       fft_threads=false,
	       fftwflag=FFTW.ESTIMATE,
	       dobs=nothing,
	       saobs=nothing,
	       gobs=nothing,
	       sobs=nothing,
	       verbose=false,
	       ) 

	# use maximum threads for fft
	fft_threads &&  (FFTW.set_num_threads(Sys.CPU_CORES))

	# create models depending on mode
	# gi*sa
	obs1=Conv.Param(ssize=[2*nt-1], dsize=[2*(nt+ntg)-1,nr], gsize=[ntg,nr], 
	 slags=[nt-1, nt-1], dlags=[nt+ntg-1, nt+ntg-1], glags=[ntg-1, 0], fftwflag=fftwflag)
	cal1=deepcopy(obs1)

	obs2=Conv.Param(ssize=[2*(nt+ntg)-1, nr], gsize=[nt], dsize=[nt,nr], 
	 slags=[nt+ntg-1, nt+ntg-1], glags=[nt-1,0], dlags=[nt-1, 0], fftwflag=fftwflag)
	cal2=deepcopy(obs2)



	# initial values are random
	ds=zeros(cal2.g)
	dg=zeros(cal1.g)
	ddcal=zeros(cal1.d)
	
	# inversion variables allocation
	xg = zeros(length(cal1.g));
	xs = zeros(length(cal2.g));
	last_xg = randn(size(xg)) # reset last_x
	last_xs = randn(size(xs)) # reset last_x

	err=DataFrame(g=[], g_nodecon=[], s=[],d=[])

	attrib_inv=:g

	pa=ParamP(ntg,nt,nr,
	  obs1, cal1, obs2, cal2,
	dg,
	ds,
	ddcal,
	attrib_inv,
	verbose,
	xg,
	last_xg,
	x->randn(),
	x->randn(),
	xs,
	last_xs,
	x->randn(),
	x->randn(),
	err,
	)

	if(!(saobs===nothing))
		for i in eachindex(saobs)
			pa.obs1.s[i]=saobs[i]
			pa.cal1.s[i]=saobs[i]
		end
	else
		error("need saobs")
	end

	if(!(dobs===nothing))
		for i in eachindex(dobs)
			pa.obs2.d[i]=dobs[i]
			pa.cal2.d[i]=dobs[i]
		end
	else
		error("need dobs")
	end

	initialize!(pa)
	update_func_grad!(pa)

	# perfome forward operations
	(gobs===nothing) && (gobs=randn(size(pa.obs1.g) ))
	(sobs===nothing) && (sobs=randn(size(pa.obs2.g) ))

	for i in eachindex(pa.obs1.g)
		pa.obs1.g[i]=gobs[i]
	end
	for i in eachindex(pa.obs2.g)
		pa.obs2.g[i]=sobs[i]
	end


	# these should be same
	Conv.mod!(pa.obs1, :d) # 
	Conv.mod!(pa.obs2, :s) # 


	pa.attrib_inv= :g
	F!(pa, randn(ninv(pa) ))
	pa.attrib_inv= :s
	F!(pa, randn(ninv(pa) ))

	return pa
	
end



function update_func_grad!(pa::ParamP) 

	# dfg for optimization functions
	optim_funcg= x->func_grad!(nothing, x,  pa) 
	optim_gradg=(storage, x)->func_grad!(storage, x,  pa)

	pa.attrib_inv=:g
	# create dfg for optimization
	pa.g_func = optim_funcg
	pa.g_grad! = optim_gradg


	# dfs for optimization functions
	optim_funcs=x->func_grad!(nothing, x,  pa) 
	optim_grads=(storage, x)->func_grad!(storage, x,  pa) 

	pa.attrib_inv=:s
	pa.s_func = optim_funcs
	pa.s_grad! = optim_grads

	return pa
	
end

function ninv(pa::ParamP)
	if(pa.attrib_inv == :s)
		return length(pa.xs)
	else(pa.attrib_inv == :g)
		return length(pa.xg)
	end
end

"""
compute errors
update pa.err
print?
give either cal or calsave?
"""
function err!(pa::ParamP) 
	f = Misfits.error_squared_euclidean!(nothing, pa.cal1.d, pa.cal2.s, nothing, norm_flag=true)

	#push!(pa.err[:s],fs)
	push!(pa.err[:d],f)
	#push!(pa.err[:g],fg)
	#push!(pa.err[:g_nodecon],fg_nodecon)
	println("Blind Decon Errors\t")
	println("==================")
	show(pa.err)
end 


function model_to_x!(x, pa::ParamP)
	if(pa.attrib_inv == :s)
		for i in eachindex(x)
			x[i]=pa.cal2.g[i]
		end
	else(pa.attrib_inv == :g)
		for i in eachindex(x)
			x[i]=pa.cal1.g[i]
		end
	end
	return x
end


function x_to_model!(x, pa::ParamP)
	if(pa.attrib_inv == :s)
		for i in eachindex(x)
			# put same in all receivers
			pa.cal2.g[i]=x[i]
		end
	else(pa.attrib_inv == :g)
		for i in eachindex(x)
			pa.cal1.g[i]=x[i]
		end
	end
	return pa
end

function F!(pa::ParamP,	x::AbstractVector{Float64}  )
	if(pa.attrib_inv==:s)
		compute=(x!=pa.last_xs)
	elseif(pa.attrib_inv==:g)
		compute=(x!=pa.last_xg)
	else
		compute=false
	end

	if(compute)

		x_to_model!(x, pa) # modify pa.cal.s or pa.cal.g

		#pa.verbose && println("updating buffer")
		if(pa.attrib_inv==:s)
			copy!(pa.last_xs, x)
		elseif(pa.attrib_inv==:g)
			copy!(pa.last_xg, x)
		end

		Conv.mod!(pa.cal1, :d) # modify pa.cal.d
		Conv.mod!(pa.cal2, :s) # modify pa.cal.d
		return pa
	end
end

function func_grad!(storage, x::AbstractVector{Float64},pa::ParamP)

	# x to pa.cal.s or pa.cal.g 
	x_to_model!(x, pa)

	F!(pa, x) # forward

	if(storage === nothing)
		# compute misfit and δdcal
		f = Misfits.error_squared_euclidean!(nothing, pa.cal1.d, pa.cal2.s, nothing, norm_flag=false)
	else
		# note that ddcal are differencet for both cases
		if(pa.attrib_inv==:g)
			f = Misfits.error_squared_euclidean!(pa.ddcal, pa.cal1.d, pa.cal2.s, nothing, norm_flag=false)
		elseif(pa.attrib_inv==:s)
			f = Misfits.error_squared_euclidean!(pa.ddcal, pa.cal2.s, pa.cal1.d, nothing, norm_flag=false)
		end
		Fadj!(pa, x, storage, pa.ddcal)
	end
	return f

end



"""
Apply Fadj to 
x is not used?
"""
function Fadj!(pa::ParamP, x, storage, dcal)
	storage[:] = 0.
	if(pa.attrib_inv == :s)
		Conv.mod!(pa.cal2, :g, s=dcal, g=pa.ds)
		for j in 1:size(pa.ds,1)
			storage[j] = pa.ds[j]
		end

	else(pa.attrib_inv == :g)
		Conv.mod!(pa.cal1, :g, g=pa.dg, d=dcal)
		for i in eachindex(storage)
			storage[i]=pa.dg[i]
		end

	end
	return storage
end



function update_all!(pa::ParamP; max_roundtrips=100, max_reroundtrips=1, ParamAM_func=nothing, roundtrip_tol=1e-6,
		     optim_tols=[1e-6, 1e-6], verbose=true, )

	if(ParamAM_func===nothing)
		ParamAM_func=x->Inversion.ParamAM(x, optim_tols=optim_tols,name="Phase Retrieval",
				    roundtrip_tol=roundtrip_tol, max_roundtrips=max_roundtrips,
				    max_reroundtrips=max_reroundtrips,
				    min_roundtrips=10,
				    verbose=verbose,
				    reinit_func=x->initialize!(pa),
				    after_reroundtrip_func=x->(err!(pa); ),
				    )
	end

	
	# create alternating minimization parameters
	f1=x->update_s!(pa, pa.xs)
	f2=x->update_g!(pa, pa.xg)
	paam=ParamAM_func([f1, f2])

	# do inversion
	Inversion.go(paam)

	# print errors
	err!(pa)
	println(" ")
end


function initialize!(pa::ParamP)
	# starting random models
	for i in eachindex(pa.cal2.s)
		x=randn()
		pa.cal2.s[i]=x
	end
	for i in eachindex(pa.cal1.g)
		x=randn()
		pa.cal1.g[i]=x
	end
end



function update_g!(pa::ParamP, xg)
	pa.attrib_inv=:g    
	resg = update!(pa, xg,  pa.g_func, pa.g_grad!)
	fg = Optim.minimum(resg)
	return fg
end

function update_s!(pa::ParamP, xs)
	pa.attrib_inv=:s    
	ress = update!(pa, xs, pa.s_func, pa.s_grad!)
	fs = Optim.minimum(ress)
	return fs
end


function update!(pa::ParamP, x, f, g!; 
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

































