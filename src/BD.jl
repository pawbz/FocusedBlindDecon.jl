

mutable struct BD
	om::ObsModel
	optm::OptimModel
	calsave::Conv.Param{Float64,2,2,1} # save the best result
	gx::X
	sx::X
	snorm_flag::Bool 	# restrict s along a unit circle during optimization
	snormmat::Matrix{Float64}            # stored outer product of s
	dsnorm::Vector{Float64}		# gradient w.r.t. normalized selet
	attrib_inv::Symbol
	verbose::Bool
	err::DataFrames.DataFrame
	# trying to penalize the energy in the correlations of g (not in practice)
	g_acorr::Conv.Param{Float64,2,2,2}
	dg_acorr::Array{Float64,2}
	sproject::ForceAutoCorr
	saobs::Vector{Float64} # auto correlation of the source (known for mode :bda)
end



"""
`gprecon` : a preconditioner applied to each Greens functions [ntg]
"""
function BD(ntg, nt, nr; 
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
	obs=Conv.Param(ssize=[nt], dsize=[nt,nr], gsize=[ntg,nr], slags=[nt-1, 0], fftwflag=fftwflag)
	cal=deepcopy(obs)
	calsave=deepcopy(cal);

	# initial values are random
	ds=zeros(cal.s)
	dg=zeros(cal.g)
	ddcal=zeros(cal.d)
	
	# inversion variables allocation
	xg = zeros(length(cal.g));
	xs = zeros(nt);
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
		for i in eachindex(pa.om.g)
			pa.om.g[i]=gobs[i]
		end# save gobs, before modifying
	end

	if(!(sobs===nothing))
		for i in eachindex(pa.om.s)
			pa.om.s[i]=sobs[i]
		end# save gobs, before modifying
	end

	if(!(dobs===nothing))
		for i in eachindex(pa.om.d)
			pa.om.d[i]=dobs[i]
		end
	else # otherwise perform modelling
		(iszero(pa.om.g) || iszero(pa.om.s)) && error("need gobs and sobs")
		obstemp=Conv.Param(ssize=[nt], dsize=[nt,nra], gsize=[ntg,nra], 
		     slags=[nt-1, 0])
		copy!(obstemp.g, pa.om.g)
		copy!(obstemp.s, pa.om.s)
		Conv.mod!(obstemp, :d) # model observed data
		copy!(pa.om.d, obstemp.d)
	end

	gobs=pa.om.g
	sobs=pa.om.s
	dobs=pa.om.d

	# obs.g <-- gobs
	copy!(pa.optm.obs.g, gobs)
	# obs.s <-- sobs
	replace_obss!(pa, sobs)
	# obs.d <-- dobs
	copy!(pa.optm.obs.d, dobs) 

	initialize!(pa)
	update_func_grad!(pa,goptim=goptim,soptim=soptim,gαvec=gαvec,sαvec=sαvec)

	return pa
	
end



function model_to_x!(x, pa)
	if(pa.attrib_inv == :s)
		if(pa.mode ∈ [:bd, :bda])
			for i in eachindex(x)
				x[i]=pa.optm.cal.s[i]*pa.sx.precon[i]
			end
		elseif(pa.mode==:ibd)
			for i in eachindex(x)
				x[i]=pa.optm.cal.s[i+pa.nt-1]*pa.sx.precon[i] # just take any one receiver and positive lags
			end
			x[1]=1.0 # zero lag
		end
	else(pa.attrib_inv == :g)
		for i in eachindex(x)
			x[i]=pa.optm.cal.g[i]*pa.gx.precon[i] 		# multiply by gprecon
		end
	end
	return x
end



function x_to_model!(x, pa)
	if(pa.attrib_inv == :s)
		if(pa.mode ∈ [:bd, :bda])
			for i in 1:pa.nt
				# put same in all receivers
				pa.optm.cal.s[i]=x[i]*pa.sx.preconI[i]
			end
			if(pa.snorm_flag)
				xn=vecnorm(x)
				scale!(pa.optm.cal.s, inv(xn))
			end
		elseif(pa.mode==:ibd)
			pa.optm.cal.s[pa.nt]=1.0 # fix zero lag
			for i in 1:pa.nt-1
				# put same in all receivers
				pa.optm.cal.s[pa.nt+i]=x[i+1]*pa.sx.preconI[i+1]
				# put same in negative lags
				pa.optm.cal.s[pa.nt-i]=x[i+1]*pa.sx.preconI[i+1]
			end
		end
	else(pa.attrib_inv == :g)
		for i in eachindex(pa.optm.cal.g)
			pa.optm.cal.g[i]=x[i]*pa.gx.preconI[i]
		end
	end
	return pa
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


function bd!(pa)

	(pa.mode ∉ [:bd, :bda]) && error("only bd modes accepted")

	update_func_grad!(pa,goptim=[:ls], gαvec=[1.]);
	initialize!(pa)
	update_all!(pa, max_reroundtrips=1, max_roundtrips=100000, roundtrip_tol=1e-8)

	update_calsave!(pa)
	err!(pa)
end

function F!(pa::BD,	x::AbstractVector{Float64}  )
	if(pa.attrib_inv==:s)
		compute=(x!=pa.sx.last_x)
	elseif(pa.attrib_inv==:g)
		compute=(x!=pa.gx.last_x)
	else
		compute=false
	end

	if(compute)

		x_to_model!(x, pa) # modify pa.optm.cal.s or pa.optm.cal.g

		#pa.verbose && println("updating buffer")
		if(pa.attrib_inv==:s)
			copy!(pa.sx.last_x, x)
		elseif(pa.attrib_inv==:g)
			copy!(pa.gx.last_x, x)
		end

		Conv.mod!(pa.optm.cal, :d) # modify pa.optm.cal.d
		return pa
	end
end


"""
Apply Fadj to 
x is not used?
"""
function Fadj!(pa::BD, x, storage, dcal)
	storage[:] = 0.
	if(pa.attrib_inv == :s)
		Conv.mod!(pa.optm.cal, :s, d=dcal, s=pa.optm.ds)
		# stack ∇s along receivers
		for j in 1:size(pa.optm.ds,1)
			storage[j] = pa.optm.ds[j]
		end

		# apply precon
		for i in eachindex(storage)
			if(iszero(pa.sx.precon[i]))
				storage[i]=0.0
			else
				storage[i] = storage[i]*pa.sx.preconI[i]
			end
		end
		# factor, because s was divided by norm of x
		if(pa.snorm_flag)
			copy!(pa.optm.dsnorm, storage)
			Misfits.derivative_vector_magnitude!(storage,pa.optm.dsnorm,x,pa.snormmat)
		end

	else(pa.attrib_inv == :g)
		Conv.mod!(pa.optm.cal, :g, g=pa.optm.dg, d=dcal)
		copy!(storage, pa.optm.dg) # remove?

		for i in eachindex(storage)
			if(iszero(pa.gx.precon[i]))
				storage[i]=0.0
			else
				storage[i]=pa.optm.dg[i]/pa.gx.precon[i]
			end
		end

	end
	return storage
end


function initialize!(pa::BD)
	# starting random models
	for i in 1:pa.nt
		x=(pa.sx.precon[i]≠0.0) ? randn() : 0.0
		pa.optm.cal.s[i]=x
	end
	for i in eachindex(pa.optm.cal.g)
		x=(pa.gx.precon[i]≠0.0) ? randn() : 0.0
		pa.optm.cal.g[i]=x
	end
end



function project_s!(pa::Param)
	copy!(pa.optm.ds, pa.optm.cal.s)
	project!(pa.optm.cal.s, pa.optm.ds, pa.sproject)
end



"""
compute errors
update pa.err
print?
give either cal or calsave?
"""
function err!(pa::BD; cal=pa.optm.cal) 
	xg_nodecon=hcat(Conv.xcorr(pa.om.d, lags=[pa.ntg-1, pa.ntg-1])...)
	xgobs=hcat(Conv.xcorr(pa.om.g)...) # compute xcorr with reference g
	fs = Misfits.error_after_normalized_autocor(cal.s, pa.optm.obs.s)
	xgcal=hcat(Conv.xcorr(cal.g)...) # compute xcorr with reference g
	fg = Misfits.error_squared_euclidean!(nothing, xgcal, xgobs, nothing, norm_flag=true)
	fg_nodecon = Misfits.error_squared_euclidean!(nothing, xg_nodecon, xgobs, nothing, norm_flag=true)
	f = Misfits.error_squared_euclidean!(nothing, cal.d, pa.optm.obs.d, nothing, norm_flag=true)

	push!(pa.err[:s],fs)
	push!(pa.err[:d],f)
	push!(pa.err[:g],fg)
	push!(pa.err[:g_nodecon],fg_nodecon)
	println("Blind Decon Errors\t")
	println("==================")
	show(pa.err)
end 

