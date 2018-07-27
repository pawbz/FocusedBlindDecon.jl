

mutable struct BD
	om::ObsModel
	optm::OptimModel
	calsave::Conv.P_conv{Float64,2,2,1} # save the best result
	gx::X
	sx::X
	snorm_flag::Bool 	# restrict s along a unit circle during optimization
	snormmat::Matrix{Float64}            # stored outer product of s
	dsnorm::Vector{Float64}		# gradient w.r.t. normalized selet
	attrib_inv::Symbol
	verbose::Bool
	err::DataFrames.DataFrame
	fourier_constraints_flag::Bool
end



"""
Constructor for the blind deconvolution problem
"""
function BD(ntg, nt, nr, nts; 
	    gprecon_attrib=:none,
	       gweights=nothing,
	       goptim=nothing,
	       gαvec=nothing,
	       soptim=nothing,
	       sαvec=nothing,
	       sprecon=nothing,
	       snorm_flag=false,
	       fft_threads=false,
	       fftwflag=FFTW.PATIENT,
	       fourier_constraints_flag=false,
	       dobs=nothing, gobs=nothing, sobs=nothing, verbose=false, attrib_inv=:g,
	       ) 

	if(ntg+nts-1 ≠ nt)
		error("invalid sizes for convolutional model")
	end

	if(fftwflag==FFTW.PATIENT)
		# use maximum threads for fft
		fft_threads &&  (FFTW.set_num_threads(Sys.CPU_CORES))
	end

	# store observed data
	om=ObsModel(ntg, nt, nr, nts, d=dobs, g=gobs, s=sobs)

	# create models depending on mode
	optm=OptimModel(ntg, nt, nr, nts, fftwflag=fftwflag, 
	slags=[nts-1, 0], 
	dlags=[nt-1, 0], 
	glags=[ntg-1, 0], 
		 )

	# inversion variables allocation
	gx=X(length(optm.cal.g))
	sx=X(length(optm.cal.s))

	snorm_flag ?	(snormmat=zeros(nts, nts)) : (snormmat=zeros(1,1))
	snorm_flag ?	(dsnorm=zeros(nts)) : (dsnorm=zeros(1))

	err=DataFrame(g=[], g_nodecon=[], s=[], d=[], front_load=[], whiteness=[])


	calsave=deepcopy(optm.cal)
	pa=BD(
		om,		optm,		calsave,		gx,		sx,		snorm_flag,
		snormmat,		dsnorm,		attrib_inv,		verbose,
		err,		# trying to penalize the energy in the correlations of g (not in practice),
		fourier_constraints_flag)




	gobs=pa.om.g
	sobs=pa.om.s
	dobs=pa.om.d

	# obs.g <-- gobs
	replace!(pa.optm, gobs, :obs, :g )
	# obs.s <-- sobs
	replace!(pa.optm, sobs, :obs, :s )
	# obs.d <-- dobs
	copy!(pa.optm.obs.d, dobs) #  


	add_precons!(pa, pa.om.g, attrib=gprecon_attrib)

	initialize!(pa)
	#update_func_grad!(pa,goptim=goptim,soptim=soptim,gαvec=gαvec,sαvec=sαvec)

	return pa
	
end



function model_to_x!(x, pa::BD)
	if(pa.attrib_inv == :s)
		for i in eachindex(x)
			x[i]=pa.optm.cal.s[i]*pa.sx.precon[i]
		end
	else(pa.attrib_inv == :g)
		for i in eachindex(x)
			x[i]=pa.optm.cal.g[i]*pa.gx.precon[i] 		# multiply by gprecon
		end
	end
	return x
end



function x_to_model!(x, pa::BD)
	if(pa.attrib_inv == :s)
		for i in eachindex(pa.optm.cal.s)
			# put same in all receivers
			pa.optm.cal.s[i]=x[i]*pa.sx.preconI[i]
		end
		if(pa.snorm_flag)
			xn=vecnorm(x)
			scale!(pa.optm.cal.s, inv(xn))
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
function add_precons!(pa::BD, gobs; αexp=0.0, cflag=true,
		       max_tfrac_gprecon=1.0, attrib=:focus)
	
	ntg=pa.om.ntg
	nts=pa.om.nts

	ntgprecon=round(Int,max_tfrac_gprecon*ntg);

	nr=size(gobs,2)
	sprecon=ones(nts)
	gprecon=ones(ntg, nr); 
	gweights=ones(ntg, nr); 
	if(attrib==:windows)
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
	elseif(attrib==:focus)
		ir0=inear(gobs)
		# first receiver is a spike
		gprecon[2:end,ir0]=0.0

	end

	add_gprecon!(pa, gprecon)
	add_gweights!(pa, gweights)
	add_sprecon!(pa, sprecon)
 
	return pa
end


"return index of closest receiver"
function inear(gobs, threshold=1e-6)
	nr=size(gobs,2)
	ir0=indmin([findfirst(x->abs(x)>threshold, vec(gobs[:,ir])) for ir in 1:nr])
	return ir0
end


function bd!(pa::BD, io=STDOUT)

	if(io===nothing)
		logfilename=joinpath(pwd(),string("XBD",now(),".log"))
		io=open(logfilename, "a+")
	end

	#update_func_grad!(pa,goptim=[:ls], gαvec=[1.]);
	initialize!(pa)
	update_all!(pa, io, max_reroundtrips=1, max_roundtrips=100000, roundtrip_tol=1e-8)

	update_calsave!(pa.optm, pa.calsave)
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

		Conv.mod!(pa.optm.cal, Conv.D()) # modify pa.optm.cal.d
		return pa
	end
end

"""
To define linear operator
"""
function F!(y, x, pa::BD)
	F!(pa,x)
	for i in eachindex(y)
		y[i]=pa.optm.cal.d[i]
	end
end

"""
Apply Fadj to 
x is not used?
"""
function Fadj!(pa::BD, x, storage, dcal)
	storage[:] = 0.
	if(pa.attrib_inv == :s)
		Conv.mod!(pa.optm.cal, Conv.S(), d=dcal, s=pa.optm.ds)
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
		Conv.mod!(pa.optm.cal, Conv.G(), g=pa.optm.dg, d=dcal)

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

"""
To define linear operator
"""
function Fadj!(y, x, pa::BD)
	for i in eachindex(y)
		pa.optm.ddcal[i]=y[i]
	end
	storage=similar(x)
	println(size(storage))
	println(size(x))
	println(size(pa.gx.precon))
	println(size(pa.optm.dg))
	Fadj!(pa, x, storage, pa.optm.ddcal)
	copy!(x, storage)
end



function initialize!(pa::BD)
	# starting random models
	for i in eachindex(pa.optm.cal.s)
		x=(pa.sx.precon[i]≠0.0) ? randn() : 0.0
		pa.optm.cal.s[i]=x
	end
	for i in eachindex(pa.optm.cal.g)
		x=(pa.gx.precon[i]≠0.0) ? randn() : 0.0
		pa.optm.cal.g[i]=x
	end
	if(pa.fourier_constraints_flag)
		#apply_fourier_constraints!(pa)
		phase_retrievel!(pa.optm.cal.g, pa.gpacse, pa.gx.precon)
		remove_gprecon!(pa, including_zeros=true)
		phase_retrievel!(pa.optm.cal.g, pa.gpacse)
		phase_retrievel!(pa.optm.cal.s, pa.spacse)
	end

end






"""
compute errors
update pa.err
print?
give either cal or calsave?
"""
function err!(pa::BD, io=STDOUT; cal=pa.optm.cal) 
	xg_nodecon=hcat(Conv.xcorr(pa.om.d,Conv.P_xcorr(pa.om.nt, pa.om.nr, cglags=[pa.optm.ntg-1, pa.optm.ntg-1]))...)
	xgobs=hcat(Conv.xcorr(pa.om.g)...) # compute xcorr with reference g
	fs = Misfits.error_after_normalized_autocor(cal.s, pa.optm.obs.s)
	xgcal=hcat(Conv.xcorr(cal.g)...) # compute xcorr with reference g
	fg = Misfits.error_squared_euclidean!(nothing, xgcal, xgobs, nothing, norm_flag=true)
	fg_nodecon = Misfits.error_squared_euclidean!(nothing, xg_nodecon, xgobs, nothing, norm_flag=true)
	f = Misfits.error_squared_euclidean!(nothing, cal.d, pa.optm.obs.d, nothing, norm_flag=true)

	whiteness=Conv.func_grad!(nothing, cal.g, Conv.P_misfit_weighted_acorr(pa.om.ntg,pa.om.nr))

	front_load=Misfits.front_load!(nothing, cal.g)

	push!(pa.err[:s],fs)
	push!(pa.err[:d],f)
	push!(pa.err[:g],fg)
	push!(pa.err[:whiteness],whiteness)
	push!(pa.err[:front_load],front_load)
	push!(pa.err[:g_nodecon],fg_nodecon)
	write(io,"Blind Decon Errors\t\n")
	write(io,"==================\n")
	write(io, string(pa.err))
end 

function update_g!(pa::BD, xg)
	pa.attrib_inv=:g    
#	if(pa.fourier_constraints_flag)
#		phase_retrievel!(pa.optm.cal.g, pa.gpacse)
#	end
	resg = update!(pa, xg)
	fg = Optim.minimum(resg)
	return fg
end

function update_s!(pa::BD, xs)
	pa.attrib_inv=:s    
	ress = update!(pa, xs)
	fs = Optim.minimum(ress)
	return fs
end


struct BandLimit <: Manifold
end

function retract!(::BandLimit,x)
	#=
	x_to_model!(x, pa) # modify pa.optm.cal.s or pa.optm.cal.g

	pac=pa.optm.cal
	if(pa.attrib_inv==:g)
	Conv.pad_truncate!(pac.g, pac.gpad, pac.glags[1], pac.glags[2], pac.np2, 1)
	A_mul_B!(pac.gfreq, pac.gfftp, pac.gpad)

	nr=pa.om.nr
	for ir in 1:nr
		for i in 1:10
			pac.gfreq[i,ir]=complex(0.0,0.0)
		end
	end
	A_mul_B!(pac.gpad, pac.gifftp, pac.gfreq)
	Conv.pad_truncate!(pac.g, pac.gpad, pac.glags[1], pac.glags[2], pac.np2, -1)
	=#
end
export retract!

function project_tangent!(::BandLimit,g,x)

end
