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

include("Phase.jl")

mutable struct Param
	ntg::Int64
	nt::Int64
	nr::Int64
	nra::Int64 # save actual nr here, as for mode==:ibd nr=nr*nr
	gobs::Array{Float64, 2} # save, before modifying it in obs
	sobs::Vector{Float64} # save, before modifying it in obs
	dobs::Array{Float64,2} # save before modifying it
	obs::Conv.Param{Float64,2,2,1} # observed convolutional model
	cal::Conv.Param{Float64,2,2,1} # calculated convolutional model
	calsave::Conv.Param{Float64,2,2,1} # save the best result
	dg::Array{Float64,2}
	ds::Vector{Float64}
	ddcal::Array{Float64,2}
	gprecon::Array{Float64,2}
	gpreconI::Array{Float64,2}
	gweights::Array{Float64,2}
	sprecon::Vector{Float64}
	spreconI::Vector{Float64}
	snorm_flag::Bool 			# restrict s along a unit circle during optimization
	snormmat::Matrix{Float64}             # stored outer product of s
	dsnorm::Vector{Float64}		# gradient w.r.t. normalized selet
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
	mode::Symbol
	g_acorr::Conv.Param{Float64,2,2,2}
	dg_acorr::Array{Float64,2}
	sproject::ForceAutoCorr
	saobs::Vector{Float64} # auto correlation of the source (known for mode :bda)
end



"""
`gprecon` : a preconditioner applied to each Greens functions [ntg]
"""
function Param(ntg, nt, nr; 
	       mode=:bd,
	       gprecon=nothing,
	       gweights=nothing,
	       goptim=nothing,
	       gαvec=nothing,
	       soptim=nothing,
	       sαvec=nothing,
	       sprecon=nothing,
	       snorm_flag=false,
	       fft_threads=false,
	       fftwflag=FFTW.PATIENT,
	       dobs=nothing, gobs=nothing, sobs=nothing, verbose=false, attrib_inv=:g,
	       saobs=nothing,
	       ) 

	# use maximum threads for fft
	fft_threads &&  (FFTW.set_num_threads(Sys.CPU_CORES))


	# create models depending on mode
	nra=nr
	if(mode ∈ [:bd, :bda])
		obs=Conv.Param(ssize=[nt], dsize=[nt,nr], gsize=[ntg,nr], slags=[nt-1, 0], fftwflag=fftwflag)
		cal=deepcopy(obs)
	elseif(mode==:ibd)
		nr=binomial(nr, 2)+nr
		obs=Conv.Param(ssize=[2*nt-1], dsize=[2*nt-1,nr], gsize=[2*ntg-1,nr], 
		 slags=[nt-1, nt-1], dlags=[nt-1, nt-1], glags=[ntg-1, ntg-1], fftwflag=fftwflag)
		cal=deepcopy(obs)
	end
	calsave=deepcopy(cal);

	# initial values are random
	ds=zeros(cal.s)
	dg=zeros(cal.g)
	ddcal=zeros(cal.d)
	
	# inversion variables allocation
	xg = zeros(length(cal.g));
	if(mode ∈ [:bd, :bda])
		xs = zeros(nt);
	elseif(mode==:ibd)
		xs = zeros(nt);
		xs[1] = 1.0
	end
	last_xg = randn(size(xg)) # reset last_x
	last_xs = randn(size(xs)) # reset last_x

	snorm_flag ?	(snormmat=zeros(nt, nt)) : (snormmat=zeros(1,1))
	snorm_flag ?	(dsnorm=zeros(nt)) : (dsnorm=zeros(1))

	err=DataFrame(g=[], g_nodecon=[], s=[],d=[])

	g_acorr=Conv.Param(gsize=[ntg,nra], dsize=[ntg,nra], ssize=[2*ntg-1,nra], slags=[ntg-1, ntg-1], fftwflag=fftwflag)
	dg_acorr=zeros(2*ntg-1, nra)

	if(mode == :bda)
		(saobs===nothing) && error("need saobs")
		sproject=DeConv.ForceAutoCorr(saobs, cal.np2)
	else
		saobs=zeros(2*nt-1)
		sproject=DeConv.ForceAutoCorr(saobs, cal.np2)
	end

	pa=Param(ntg,nt,nr,
		 nra,
	  zeros(ntg,nra), zeros(nt),
	  zeros(nt, nra),
	  obs,cal,calsave, dg,ds,ddcal,
	  zeros(cal.g), zeros(cal.g),
	  zeros(cal.g),
	  zeros(xs), zeros(xs),
	  snorm_flag,snormmat,
	  dsnorm,attrib_inv,verbose,xg,last_xg,x->randn(),x->randn(),xs,last_xs,x->randn(),x->randn(), err,
	  mode, g_acorr, dg_acorr, sproject, saobs)

	add_gprecon!(pa, gprecon)
	add_gweights!(pa, gweights)
	add_sprecon!(pa, sprecon)
 
	if(!(gobs===nothing))
		for i in eachindex(pa.gobs)
			pa.gobs[i]=gobs[i]
		end# save gobs, before modifying
	end

	if(!(sobs===nothing))
		for i in eachindex(pa.sobs)
			pa.sobs[i]=sobs[i]
		end# save gobs, before modifying
	end

	if(!(dobs===nothing))
		for i in eachindex(pa.dobs)
			pa.dobs[i]=dobs[i]
		end
	else # otherwise perform modelling
		(iszero(pa.gobs) || iszero(pa.sobs)) && error("need gobs and sobs")
		obstemp=Conv.Param(ssize=[nt], dsize=[nt,nra], gsize=[ntg,nra], 
		     slags=[nt-1, 0])
		copy!(obstemp.g, pa.gobs)
		copy!(obstemp.s, pa.sobs)
		Conv.mod!(obstemp, :d) # model observed data
		copy!(pa.dobs, obstemp.d)
	end

	if(mode ∈ [:bd, :bda])
		gobs=pa.gobs
		sobs=pa.sobs
		dobs=pa.dobs
	elseif(mode==:ibd)
		gobs=hcat(Conv.xcorr(pa.gobs)...)
		sobs=hcat(Conv.xcorr(pa.sobs)...)
		dobs=hcat(Conv.xcorr(pa.dobs)...) # do a cross-correlation 
	end

	# obs.g <-- gobs
	copy!(pa.obs.g, gobs)
	# obs.s <-- sobs
	replace_obss!(pa, sobs)
	# obs.d <-- dobs
	copy!(pa.obs.d, dobs) 

	initialize!(pa)
	update_func_grad!(pa,goptim=goptim,soptim=soptim,gαvec=gαvec,sαvec=sαvec)

	return pa
	
end

function add_gprecon!(pa::Param, gprecon=nothing)
	# create g precon
	(gprecon===nothing) && (gprecon=ones(pa.cal.g))

	copy!(pa.gprecon, gprecon)		
	copy!(pa.gpreconI, pa.gprecon)
	for i in eachindex(gprecon)
		if(!(iszero(gprecon[i])))
			pa.gpreconI[i]=inv(pa.gprecon[i])
		end
	end
end
function add_sprecon!(pa::Param, sprecon=nothing)
	# create g precon
	if(sprecon===nothing) 
		sprecon=ones(pa.xs)
		if(pa.mode==:ibd)
			sprecon[1]=0.0 # do not update zero lag
		end
	end
	copy!(pa.sprecon, sprecon)
	copy!(pa.spreconI, pa.sprecon)
	for i in eachindex(sprecon)
		if(!(iszero(sprecon[i])))
			pa.spreconI[i]=inv(pa.sprecon[i])
		end
	end
end
function add_gweights!(pa::Param, gweights=nothing)
	(gweights===nothing) && (gweights=ones(pa.cal.g))
	copy!(pa.gweights, gweights)
end



function update_func_grad!(pa; goptim=nothing, soptim=nothing, gαvec=nothing, sαvec=nothing)
	# they will be changed in this program, so make a copy 
	ssave=copy(pa.cal.s);
	gsave=copy(pa.cal.g);
	dcalsave=copy(pa.cal.d);

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
	paMOg=Inversion.ParamMO(noptim=length(goptim), ninv=length(pa.xg), αvec=gαvec,
			    		optim_func=optim_funcg,optim_grad=optim_gradg,
					x_init=randn(length(pa.xg),10))
	# create dfg for optimization
	pa.g_func = x -> paMOg.func(x, paMOg)
	pa.g_grad! = (storage, x) -> paMOg.grad!(storage, x, paMOg)
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
	paMOs=Inversion.ParamMO(noptim=length(soptim), ninv=length(pa.xs), αvec=sαvec,
			    		optim_func=optim_funcs,optim_grad=optim_grads,
					x_init=vcat(ones(1,10),randn(length(pa.xs)-1,10)))
#	pa.dfs = OnceDifferentiable(x -> paMOs.func(x, paMOs),         
#			    (storage, x) -> paMOs.grad!(storage, x, paMOs))
	pa.s_func = x -> paMOs.func(x, paMOs)
	pa.s_grad! =  (storage, x) -> paMOs.grad!(storage, x, paMOs)


	copy!(pa.cal.s, ssave)
	copy!(pa.cal.g, gsave)
	copy!(pa.cal.d,dcalsave)

	return pa
	
end


function ninv(pa)
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
function err!(pa::Param; cal=pa.cal) 
	xg_nodecon=hcat(Conv.xcorr(pa.dobs, lags=[pa.ntg-1, pa.ntg-1])...)
	xgobs=hcat(Conv.xcorr(pa.gobs)...) # compute xcorr with reference g
	if(pa.mode ∈ [:bd, :bda])
		fs = Misfits.error_after_normalized_autocor(cal.s, pa.obs.s)
		xgcal=hcat(Conv.xcorr(cal.g)...) # compute xcorr with reference g
		fg = Misfits.error_squared_euclidean!(nothing, xgcal, xgobs, nothing, norm_flag=true)
	elseif(pa.mode==:ibd)
		fg = Misfits.error_squared_euclidean!(nothing, cal.g, pa.obs.g, nothing, norm_flag=true)
		fs = Misfits.error_squared_euclidean!(nothing, cal.s, pa.obs.s, nothing, norm_flag=true)
	end
	fg_nodecon = Misfits.error_squared_euclidean!(nothing, xg_nodecon, xgobs, nothing, norm_flag=true)
	f = Misfits.error_squared_euclidean!(nothing, cal.d, pa.obs.d, nothing, norm_flag=true)

	push!(pa.err[:s],fs)
	push!(pa.err[:d],f)
	push!(pa.err[:g],fg)
	push!(pa.err[:g_nodecon],fg_nodecon)
	println("Blind Decon Errors\t")
	println("==================")
	show(pa.err)
end 


function model_to_x!(x, pa)
	if(pa.attrib_inv == :s)
		if(pa.mode ∈ [:bd, :bda])
			for i in eachindex(x)
				x[i]=pa.cal.s[i]*pa.sprecon[i]
			end
		elseif(pa.mode==:ibd)
			for i in eachindex(x)
				x[i]=pa.cal.s[i+pa.nt-1]*pa.sprecon[i] # just take any one receiver and positive lags
			end
			x[1]=1.0 # zero lag
		end
	else(pa.attrib_inv == :g)
		for i in eachindex(x)
			x[i]=pa.cal.g[i]*pa.gprecon[i] 		# multiply by gprecon
		end
	end
	return x
end


function x_to_model!(x, pa)
	if(pa.attrib_inv == :s)
		if(pa.mode ∈ [:bd, :bda])
			for i in 1:pa.nt
				# put same in all receivers
				pa.cal.s[i]=x[i]*pa.spreconI[i]
			end
			if(pa.snorm_flag)
				xn=vecnorm(x)
				scale!(pa.cal.s, inv(xn))
			end
		elseif(pa.mode==:ibd)
			pa.cal.s[pa.nt]=1.0 # fix zero lag
			for i in 1:pa.nt-1
				# put same in all receivers
				pa.cal.s[pa.nt+i]=x[i+1]*pa.spreconI[i+1]
				# put same in negative lags
				pa.cal.s[pa.nt-i]=x[i+1]*pa.spreconI[i+1]
			end
		end
	else(pa.attrib_inv == :g)
		for i in eachindex(pa.cal.g)
			pa.cal.g[i]=x[i]*pa.gpreconI[i]
		end
	end
	return pa
end

function F!(pa::Param,	x::AbstractVector{Float64}  )
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

		Conv.mod!(pa.cal, :d) # modify pa.cal.d
		return pa
	end
end

function func_grad!(storage, x::AbstractVector{Float64},pa)

	# x to pa.cal.s or pa.cal.g 
	x_to_model!(x, pa)

	F!(pa, x) # forward

	if(storage === nothing)
		# compute misfit and δdcal
		f = Misfits.error_squared_euclidean!(nothing, pa.cal.d, pa.obs.d, nothing, norm_flag=true)
	else
		f = Misfits.error_squared_euclidean!(pa.ddcal, pa.cal.d, pa.obs.d, nothing, norm_flag=true)
		Fadj!(pa, x, storage, pa.ddcal)
	end
	return f

end

"""
update calsave only when error in d is low
"""
function update_calsave!(pa)
	f1=Misfits.error_squared_euclidean!(nothing, pa.calsave.d, pa.obs.d, nothing, norm_flag=true)
	f2=Misfits.error_squared_euclidean!(nothing, pa.cal.d, pa.obs.d, nothing, norm_flag=true)
	if(f2<f1)
		copy!(pa.calsave.d, pa.cal.d)
		copy!(pa.calsave.g, pa.cal.g)
		copy!(pa.calsave.s, pa.cal.s)
	end
end

# add model based constraints here

# all the greens' functions have to be correlated

# exponential-weighted norm for the green functions
function func_grad_g_weights!(storage, x, pa)
	x_to_model!(x, pa)
	!(pa.attrib_inv == :g) && error("only for g inversion")
	if(!(storage === nothing)) #
		f = Misfits.error_weighted_norm!(pa.dg,pa.cal.g, pa.gweights) #
		for i in eachindex(storage)
			storage[i]=pa.dg[i]
		end
	else	
		f = Misfits.error_weighted_norm!(nothing,pa.cal.g, pa.gweights)
	end
	return f
end

# exponential-weighted norm for the green functions
function func_grad_g_acorr_weights!(storage, x, pa)
	x_to_model!(x, pa)
	!(pa.attrib_inv == :g) && error("only for g inversion")

	if(!(storage === nothing)) #
		f = Misfits.error_acorr_weighted_norm!(pa.dg,pa.cal.g, 
					 paconv=pa.g_acorr,dfds=pa.dg_acorr) #
		for i in eachindex(storage)
			storage[i]=pa.dg[i]
		end
	else	
		f = Misfits.error_acorr_weighted_norm!(nothing,pa.cal.g, 
					 paconv=pa.g_acorr,dfds=pa.dg_acorr)
	end

	return f
end
#  



"""
Apply Fadj to 
x is not used?
"""
function Fadj!(pa, x, storage, dcal)
	storage[:] = 0.
	if(pa.attrib_inv == :s)
		Conv.mod!(pa.cal, :s, d=dcal, s=pa.ds)
		if(pa.mode ∈ [:bd, :bda])
			# stack ∇s along receivers
			for j in 1:size(pa.ds,1)
				storage[j] = pa.ds[j]
			end
		elseif(pa.mode==:ibd)
			# stacking over +ve and -ve lags
			for j in 2:pa.nt
				storage[j] += pa.ds[pa.nt-j+1] # -ve lags
				storage[j] += pa.ds[pa.nt+j-1] # +ve lags
			end
		end

		# apply precon
		for i in eachindex(storage)
			if(iszero(pa.sprecon[i]))
				storage[i]=0.0
			else
				storage[i] = storage[i]*pa.spreconI[i]
			end
		end
		# factor, because s was divided by norm of x
		if(pa.snorm_flag)
			copy!(pa.dsnorm, storage)
			Misfits.derivative_vector_magnitude!(storage,pa.dsnorm,x,pa.snormmat)
		end

	else(pa.attrib_inv == :g)
		Conv.mod!(pa.cal, :g, g=pa.dg, d=dcal)
		copy!(storage, pa.dg) # remove?

		for i in eachindex(storage)
			if(iszero(pa.gprecon[i]))
				storage[i]=0.0
			else
				storage[i]=pa.dg[i]/pa.gprecon[i]
			end
		end

	end
	return storage
end

# core algorithm
function update!(pa::Param, x, f, g!; 
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

function project_s!(pa::Param)
	copy!(pa.ds, pa.cal.s)
	project!(pa.cal.s, pa.ds, pa.sproject)
end

function replace_obss!(pa::Param, s)
	if(length(s)==length(pa.obs.s))
		copy!(pa.obs.s, s)
	elseif(length(s)==size(pa.obs.s,1))
		for j in 1:pa.nr, i in 1:size(pa.obs.s,1)
				pa.obs.s[i,j]=s[i]
		end
	else
		error("invalid input s")
	end
	# observed data changes as well :)
	Conv.mod!(pa.obs,:d)
end

function update_g!(pa::Param, xg)
	pa.attrib_inv=:g    
	resg = update!(pa, xg,  pa.g_func, pa.g_grad!)
	fg = Optim.minimum(resg)
	return fg
end

function update_s!(pa::Param, xs)
	pa.attrib_inv=:s    
	if(pa.mode==:bda)
		for itr in 1:100
			ress = update!(pa, xs, pa.s_func, pa.s_grad!, iterations=1)
			project_s!(pa)
		end
	else
		ress = update!(pa, xs, pa.s_func, pa.s_grad!)
	end
	fs = Optim.minimum(ress)
	return fs
end


"""
Remove preconditioners from pa
"""
function remove_gprecon!(pa; all=false)
	for i in eachindex(pa.gprecon)
		if((pa.gprecon[i]≠0.0) || all)
			pa.gprecon[i]=1.0
			pa.gpreconI[i]=1.0
		end
	end
end

"""
Remove weights from pa
"""
function remove_gweights!(pa; all=false)
	for i in eachindex(pa.gweights)
		if((pa.gweights[i]≠0.0) || all)
			pa.gweights[i]=1.0
		end
	end
end


"""
* re_init_flag :: re-initialize inversions with random input or not?
"""
function update_all!(pa::Param; max_roundtrips=100, max_reroundtrips=10, ParamAM_func=nothing, roundtrip_tol=1e-6,
		     optim_tols=[1e-6, 1e-6], verbose=true, )

	if(ParamAM_func===nothing)
		ParamAM_func=x->Inversion.ParamAM(x, optim_tols=optim_tols,name="Blind Decon",
				    roundtrip_tol=roundtrip_tol, max_roundtrips=max_roundtrips,
				    max_reroundtrips=max_reroundtrips,
				    min_roundtrips=10,
				    verbose=verbose,
				    reinit_func=x->initialize!(pa),
				    after_reroundtrip_func=x->(err!(pa); update_calsave!(pa);),
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


function initialize!(pa)
	if(pa.mode ∈ [:bd, :bda])
		# starting random models
		for i in 1:pa.nt
			x=(pa.sprecon[i]≠0.0) ? randn() : 0.0
			pa.cal.s[i]=x
		end
	elseif(pa.mode==:ibd)
		for i in 1:pa.nt-1
			x=(pa.sprecon[i]≠0.0) ? randn() : 0.0
			pa.cal.s[pa.nt+i]=x*0.0
			pa.cal.s[pa.nt-i]=x*0.0
		end
		pa.cal.s[pa.nt]=1.0
	end
	for i in eachindex(pa.cal.g)
		x=(pa.gprecon[i]≠0.0) ? randn() : 0.0
		pa.cal.g[i]=x
	end
end





"""
Create preconditioners using the observed Green Functions.
* `cflag` : impose causaulity by creating gprecon using gobs
* `max_tfrac_gprecon` : maximum length of precon windows on g
"""
function create_weights(ntg, nt, gobs; αexp=0.0, cflag=true,
		       max_tfrac_gprecon=1.0)
	
	ntgprecon=round(Int,max_tfrac_gprecon*ntg);

	nr=size(gobs,2)
	sprecon=ones(nt)
	gprecon=ones(ntg, nr); 
	gweights=ones(ntg, nr); 
	minindz=ntg
	gweights=ones(ntg, nr)
	for ir in 1:nr
		g=normalize(view(gobs,:,ir))
		indz=findfirst(x->abs(x)>1e-6, g)
	#	if(indz > 1) 
	#		indz -= 1 # window one sample less than actual
	#	end
		if(!cflag && indz≠0)
			indz=1
		end
		if(indz≠0)
			for i in 1:indz-1
				gprecon[i,ir]=0.0
				gweights[i,ir]=0.0
			end
			for i in indz:indz+ntgprecon
				if(i≤ntg)
					gweights[i,ir]=exp(αexp*(i-indz-1)/ntg)  # exponential weights
					gprecon[i,ir]=exp(αexp*(i-indz-1)/ntg)  # exponential weights
				end
			end
			for i in indz+ntgprecon+1:ntg
				gprecon[i,ir]=0.0
				gweights[i,ir]=0.0
			end
		else
			gprecon[:,ir]=0.0
			gweights[:,ir]=0.0
		end
	end
	return gprecon, gweights, sprecon
end

function create_white_weights(ntg, nt, nr)
	nrr=binomial(nr,2)+nr
	sprecon=ones(nt)
	gprecon=ones(2*ntg-1, nrr); # put ones everywhere for precon
	gweights=zeros(2*ntg-1, nrr); 

	irr=1  # auto correlation index
	for ir in 1:nr
		for i in 1:ntg-1    
			gprecon[ntg+i,irr]=0.0    # put zero at +ve lags
			gprecon[ntg-i,irr]=0.0    # put zero at -ve lags
		end
		for i in 1:ntg-1
			gweights[ntg+i,irr]=i*2
			gweights[ntg-i,irr]=i*2
		end
		irr+=nr-(ir-1)
	end
	return gprecon, gweights, nothing
end


"""
Focused Blind Deconvolution
"""
function fbd!(pa; verbose=true)

	(pa.mode≠:ibd) && error("only ibd mode accepted")

	# set α=∞ 
	gprecon, gweights, sprecon=create_white_weights(size(pa.gobs,1), size(pa.sobs), size(pa.gobs,2))
	add_gprecon!(pa, gprecon)
	add_gweights!(pa, gweights)
	add_sprecon!(pa, sprecon)

	update_func_grad!(pa,goptim=[:ls], gαvec=[1.]);
	DeConv.initialize!(pa)
	update_all!(pa, max_reroundtrips=1, max_roundtrips=100000, roundtrip_tol=1e-8, verbose=verbose)

	update_calsave!(pa)
	err!(pa)

	# set α=0
	remove_gprecon!(pa, all=true)
	remove_gweights!(pa, all=true)
	update_all!(pa, max_reroundtrips=1, max_roundtrips=2000, roundtrip_tol=1e-6, verbose=verbose)
	#
	update_calsave!(pa)
	err!(pa)
end

function bd!(pa)

	(pa.mode ∉ [:bd, :bda]) && error("only bd modes accepted")

	update_func_grad!(pa,goptim=[:ls], gαvec=[1.]);
	DeConv.initialize!(pa)
	update_all!(pa, max_reroundtrips=1, max_roundtrips=100000, roundtrip_tol=1e-8)

	update_calsave!(pa)
	err!(pa)
end


include("Plots.jl")

"""
Save Param
"""
function save(pa::Param, folder; tgridg=nothing, tgrid=nothing)
	!(isdir(folder)) && error("invalid directory")


	(tgridg===nothing) && (tgridg = Grid.M1D(0.0, (pa.ntg-1)*1.0, pa.ntg))

	# save original g
	file=joinpath(folder, "gobs.csv")
	CSV.write(file,DataFrame(hcat(tgridg.x, pa.gobs)))
	# save for imagesc
	file=joinpath(folder, "imgobs.csv")
	CSV.write(file,DataFrame(hcat(repeat(tgridg.x,outer=pa.nra),
				      repeat(1:pa.nra,inner=pa.ntg),vec(pa.gobs))),)
	# file=joinpath(folder, "gcal.csv")
	# CSV.write(file,DataFrame(hcat(tgridg.x, pa.calsave.g)))

	# compute cross-correlations of g
	xtgridg=Grid.M1D_xcorr(tgridg); # grid
	if(pa.mode ∈ [:bd, :bda])
		xgobs=hcat(Conv.xcorr(pa.gobs)...)
	elseif(pa.mode==:ibd)
		xgobs=pa.obs.g
	end
	file=joinpath(folder, "xgobs.csv")
	CSV.write(file,DataFrame(hcat(xtgridg.x, xgobs)))

	file=joinpath(folder, "imxgobs.csv")
	CSV.write(file,DataFrame(hcat(repeat(xtgridg.x,outer=pa.nra),
				      repeat(1:pa.nra,inner=xtgridg.nx),vec(xgobs[:,1:pa.nra]))),)
	if(pa.mode ∈ [:bd, :bda])
		xgcal=hcat(Conv.xcorr(pa.cal.g)...)
	elseif(pa.mode==:ibd)
		xgcal=pa.calsave.g
	end
	file=joinpath(folder, "xgcal.csv")
	CSV.write(file,DataFrame(hcat(xtgridg.x,xgcal)))
	file=joinpath(folder, "imxgcal.csv")
	CSV.write(file,DataFrame(hcat(repeat(xtgridg.x,outer=pa.nra),
				      repeat(1:pa.nra,inner=xtgridg.nx),vec(xgcal[:,1:pa.nra]))),)

	# compute cross-correlations without blind Decon
	xg_nodecon=hcat(Conv.xcorr(pa.dobs, lags=[pa.ntg-1, pa.ntg-1])...)
	file=joinpath(folder, "xg_nodecon.csv")
	CSV.write(file,DataFrame(hcat(xtgridg.x,xg_nodecon)))

	file=joinpath(folder, "imxg_nodecon.csv")
	CSV.write(file,DataFrame(hcat(repeat(xtgridg.x,outer=pa.nra),
				      repeat(1:pa.nra,inner=xtgridg.nx),vec(xg_nodecon[:,1:pa.nra]))),)

	# compute cross-correlation of source selet
	xsobs=hcat(Conv.xcorr(pa.sobs, lags=[pa.ntg-1, pa.ntg-1])...)
	file=joinpath(folder, "xsobs.csv")
	CSV.write(file,DataFrame(hcat(xtgridg.x,xsobs)))

	# cross-plot of g
	file=joinpath(folder, "gcross.csv")
	CSV.write(file,DataFrame( hcat(vec(xgobs), vec(xgcal))))
	file=joinpath(folder, "gcross_nodecon.csv")
	CSV.write(file,DataFrame( hcat(vec(xgobs), vec(xg_nodecon))))


	# compute autocorrelations of source
	if(pa.mode ∈ [:bd, :bda])
		xsobs=autocor(pa.obs.s[:,1], 1:pa.nt-1, demean=true)
		xscal=autocor(pa.calsave.s[:,1], 1:pa.nt-1, demean=true)
	elseif(pa.mode==:ibd)
		xsobs=pa.obs.s[:,1]
		xscal=pa.calsave.s[:,1]
	end
	smat=hcat(vec(xsobs), vec(xscal))
	scale!(smat, inv(maximum(abs, smat)))
	
	# resample s, final sample should not exceed 1000
	ns=length(xsobs)
	fact=(ns>1000) ? round(Int,ns/1000) : 1
	xsobs=xsobs[1:fact:ns] # resample
	xscal=xscal[1:fact:ns] # resample

	# x plots of s
	file=joinpath(folder, "scross.csv")
	CSV.write(file,DataFrame(smat))

	# x plots of data
	dcal=pa.calsave.d
	dobs=pa.obs.d
	nd=length(dcal);
	fact=(nd>1000) ? round(Int,nd/1000) : 1
	dcal=dcal[1:fact:nd]
	dobs=dobs[1:fact:nd]
	datmat=hcat(vec(dcal), vec(dobs))
	scale!(datmat, inv(maximum(abs, datmat)))

	file=joinpath(folder, "datcross.csv")
	CSV.write(file,DataFrame( datmat))

	# finally save err
	err!(pa, cal=pa.calsave) # compute err in calsave 
	file=joinpath(folder, "err.csv")
	CSV.write(file, pa.err)
end






"""
Deterministic Decon, where selet is known
"""
mutable struct ParamD{T<:Real,N}
	np2::Int
	ntd::Int
	nts::Int
	d::Array{T,N}
	g::Array{T,N}
	s::Vector{T}
	dpad::Array{T,N}
	gpad::Array{T,N}
	spad::Vector{T}
	dfreq::Array{Complex{T},N}
	greq::Array{Complex{T},N}
	sfreq::Array{Complex{T},1}
	fftplan::Base.DFT.FFTW.rFFTWPlan
	ifftplan::Base.DFT.ScaledPlan
	fftplans::Base.DFT.FFTW.rFFTWPlan
	ϵ::T
end

function ParamD(;ntd=1, nts=1, dims=(), np2=nextfastfft(maximum([2*nts, 2*ntd])), # fft dimension for plan
			d=zeros(ntd, dims...), s=zeros(nts), g=zeros(d), ϵ=1e-2)
	T=eltype(d)

	dims=size(d)[2:end]
	nrfft=div(np2,2)+1
	fftplan=plan_rfft(zeros(T, np2,dims...),[1])
	ifftplan=plan_irfft(complex.(zeros(T, nrfft,dims...)),np2,[1])
	fftplans=plan_rfft(zeros(T, np2,),[1])
	
	dfreq=complex.(zeros(T,nrfft,dims...))
	greq=complex.(zeros(T,nrfft,dims...))
	sfreq=complex.(zeros(T,nrfft))

	# preallocate padded arrays
	dpad=(zeros(T,np2,dims...))
	gpad=(zeros(T,np2,dims...))
	spad=(zeros(T,np2,))

	sv=normalize(s) # make a copy, don't edit s

	return ParamD(np2,ntd,nts,d,g,sv,dpad,gpad,spad,dfreq,greq,sfreq,
		fftplan, ifftplan, fftplans, ϵ)

end

"""
Convolution that allocates `Param` internally.
"""
function mod!{T,N}(
	   d::AbstractArray{T,N}, 
	   s::AbstractVector{T}, attrib::Symbol)
	ntd=size(d,1)
	ntg=size(g,1)
	nts=size(s,1)

	# allocation of freq matrices
	pa=ParamD(ntd=ntd, nts=nts, s=s, d=d)

	# using pa, return d, g, s according to attrib
	mod!(pa)
end

"""
Convolution modelling with no allocations at all.
By default, the fields `g`, `d` and `s` in pa are modified accordingly.
Otherwise use keyword arguments to input them.
"""
function mod!(pa::ParamD; 
	      g=pa.g, d=pa.d, s=pa.s # external arrays to be modified
	     )
	T=eltype(pa.d)
	ntd=size(pa.d,1)
	nts=size(pa.s,1)
	
	# initialize freq vectors
	pa.dfreq[:] = complex(T(0))
	pa.greq[:] = complex(T(0))
	pa.sfreq[:] = complex(T(0))

	pa.gpad[:]=T(0)
	pa.dpad[:]=T(0)
	pa.spad[:]=T(0)

	# necessary zero padding
	Conv.pad_truncate!(g, pa.gpad, ntd-1, 0, pa.np2, 1)
	Conv.pad_truncate!(d, pa.dpad, ntd-1, 0, pa.np2, 1)
	Conv.pad_truncate!(s, pa.spad, nts-1, 0, pa.np2, 1)

	A_mul_B!(pa.sfreq, pa.fftplans, pa.spad)
	A_mul_B!(pa.dfreq, pa.fftplan, pa.dpad)
	for i in eachindex(pa.greq)
		ii=ind2sub(pa.greq,i)[1]
		pa.greq[i]=pa.dfreq[i]*inv(pa.sfreq[ii]*conj(pa.sfreq[ii])+pa.ϵ)
	end
	A_mul_B!(pa.gpad, pa.ifftplan, pa.greq)

	Conv.pad_truncate!(g, pa.gpad, ntd-1, 0, pa.np2, -1)
	

	return g

end


include("Doppler.jl")


end # module
