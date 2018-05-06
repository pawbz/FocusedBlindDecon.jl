

mutable struct IBD
	om::ObsModel
	optm::OptimModel
	gx::X
	sx::X
	attrib_inv::Symbol
	verbose::Bool
	err::DataFrames.DataFrame
end



"""
`gprecon` : a preconditioner applied to each Greens functions [ntg]
"""
function IBD(ntg, nt, nr; 
	       fft_threads=true,
	       fftwflag=FFTW.PATIENT,
	       dobs=nothing, 
	       gobs=nothing, 
	       sobs=nothing, 
	       verbose=false,
	       ) 

	# use maximum threads for fft
	fft_threads &&  (FFTW.set_num_threads(Sys.CPU_CORES))

	# store observed data
	om=ObsModel(ntg, nt, nr, d=dobs, g=gobs, s=sobs)

	# create models depending on mode
	optm=OptimModel(2*ntg-1, 2*nt-1, binomial(nr, 2)+nr, fftwflag=fftwflag, 
	slags=[nt-1, nt-1], 
	dlags=[nt-1, nt-1], 
	glags=[ntg-1, ntg-1], 
		 )
		
	# inversion variables allocation
	gx=X(length(optm.cal.g))

	sx=X(nt)
	sx.x[1]=1.0

	err=DataFrame(g=[], g_nodecon=[], s=[],d=[])

	pa=IBD(om, optm, gx, sx, :g, verbose, err)

	# adjust sprecon
	pa.sx.precon[1]=0.0 # do not update zero lag
 
	gobs=hcat(Conv.xcorr(pa.om.g)...)
	sobs=hcat(Conv.xcorr(pa.om.s)...)

	# obs.g <-- gobs
	replace!(pa.optm, gobs, :obs, :g )
	# obs.s <-- sobs
	replace!(pa.optm, sobs, :obs, :s )
	# obs.d <-- dobs
	dobs=hcat(Conv.xcorr(pa.om.d)...) # do a cross-correlation 
	copy!(pa.optm.obs.d, dobs) # overwrites the forward modelling done in previous steps  

	initialize!(pa)
	update_func_grad!(pa)

	return pa
	
end


function model_to_x!(x, pa::IBD)
	if(pa.attrib_inv == :s)
		for i in eachindex(x)
			x[i]=pa.optm.cal.s[i+pa.om.nt-1]*pa.sx.precon[i] # just take positive lags
		end
		x[1]=1.0 # zero lag will be fixed
	else(pa.attrib_inv == :g)
		for i in eachindex(x)
			x[i]=pa.optm.cal.g[i]*pa.gx.precon[i] 		# multiply by gprecon
		end
	end
	return nothing
end


function x_to_model!(x, pa::IBD)
	if(pa.attrib_inv == :s)
		pa.optm.cal.s[pa.om.nt]=1.0 # fix zero lag
		for i in 1:pa.om.nt-1
			# put same in all receivers
			pa.optm.cal.s[pa.om.nt+i]=x[i+1]*pa.sx.preconI[i+1]
			# put same in negative lags
			pa.optm.cal.s[pa.om.nt-i]=x[i+1]*pa.sx.preconI[i+1]
		end
	else(pa.attrib_inv == :g)
		for i in eachindex(pa.optm.cal.g)
			pa.optm.cal.g[i]=x[i]*pa.gx.preconI[i]
		end
	end
	return pa
end

function add_focusing!(pa::IBD, α=Inf)
	(α≠Inf) && error("focusing only enabled for infinte alpha")

	nr=pa.om.nr
	ntg=pa.om.ntg
	gprecon=ones(pa.optm.ntg, pa.optm.nr); 
	gweights=ones(pa.optm.ntg, pa.optm.nr); 

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
	add_gprecon!(pa, gprecon)
	add_gweights!(pa, gweights)

	update_func_grad!(pa,goptim=[:ls], gαvec=[1.]);  # add alpha here later

	return pa
end


function ibd!(pa::IBD)

	initialize!(pa)
	update_all!(pa, max_reroundtrips=1, max_roundtrips=100000, roundtrip_tol=1e-8)

	err!(pa)
end



"""
Focused Blind Deconvolution
"""
function fbd!(pa::IBD; verbose=true)

	# set α=∞ 
	add_focusing!(pa)

	initialize!(pa)
	update_all!(pa, max_reroundtrips=1, max_roundtrips=100000, roundtrip_tol=1e-8, verbose=verbose)

	err!(pa)

	# set α=0
	remove_gprecon!(pa, including_zeros=true)
	remove_gweights!(pa, including_zeros=true)
	update_all!(pa, max_reroundtrips=1, max_roundtrips=2000, roundtrip_tol=1e-6, verbose=verbose)
	#
	err!(pa)
end


function initialize!(pa::IBD)
	for i in eachindex(pa.optm.cal.s)
		pa.optm.cal.s[i]=0.0 # +ve lags and -ve lags
	end
	pa.optm.cal.s[pa.om.nt]=1.0 # fix zero lag to one
	for i in eachindex(pa.optm.cal.g)
		x=(pa.gx.precon[i]≠0.0) ? randn() : 0.0
		pa.optm.cal.g[i]=x
	end
end


function F!(pa::IBD,	x::AbstractVector{Float64}  )
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
function Fadj!(pa, x, storage, dcal)
	storage[:] = 0.
	if(pa.attrib_inv == :s)
		Conv.mod!(pa.optm.cal, :s, d=dcal, s=pa.optm.ds)
		# stacking over +ve and -ve lags
		for j in 2:pa.om.nt
			storage[j] += pa.optm.ds[pa.om.nt-j+1] # -ve lags
			storage[j] += pa.optm.ds[pa.om.nt+j-1] # +ve lags
		end

		# apply precon
		for i in eachindex(storage)
			if(iszero(pa.sx.precon[i]))
				storage[i]=0.0
			else
				storage[i] = storage[i]*pa.sx.preconI[i]
			end
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


function update_g!(pa::IBD, xg)
	pa.attrib_inv=:g    
	resg = update!(pa, xg,  pa.gx.func, pa.gx.grad!)
	fg = Optim.minimum(resg)
	return fg
end

function update_s!(pa::IBD, xs)
	pa.attrib_inv=:s    
	ress = update!(pa, xs, pa.sx.func, pa.sx.grad!)
	fs = Optim.minimum(ress)
	return fs
end



"""
The cross-correlated Green's functions and the auto-correlated source signature have to be reconstructed exactly, except for a scaling factor.
"""
function err!(pa::IBD; cal=pa.optm.cal) 
	fg = Misfits.error_squared_euclidean!(nothing, cal.g, pa.optm.obs.g, nothing, norm_flag=true)
	fs = Misfits.error_squared_euclidean!(nothing, cal.s, pa.optm.obs.s, nothing, norm_flag=true)
	f = Misfits.error_squared_euclidean!(nothing, cal.d, pa.optm.obs.d, nothing, norm_flag=true)

	xg_nodecon=hcat(Conv.xcorr(pa.om.d,Conv.P_xcorr(pa.om.nt, pa.om.nr, cglags=[pa.om.ntg-1, pa.om.ntg-1]))...)
	xgobs=hcat(Conv.xcorr(pa.om.g)...) # compute xcorr with reference g
	fg_nodecon = Misfits.error_squared_euclidean!(nothing, xg_nodecon, xgobs, nothing, norm_flag=true)

	push!(pa.err[:s],fs)
	push!(pa.err[:d],f)
	push!(pa.err[:g],fg)
	push!(pa.err[:g_nodecon],fg_nodecon)
	println("Interferometric Blind Decon Errors\t")
	println("==================")
	show(pa.err)
end 

