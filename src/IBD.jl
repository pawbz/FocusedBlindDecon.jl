
"""
* `sx_attrib` : how to create the inversion vector for sa?
   * =:positive  only positive lags 
   * =:all all lags  
"""
mutable struct IBD
	om::ObsModel
	optm::OptimModel
	gx::X
	sx::X
	sx_attrib::Symbol
	sx_fix_zero_lag_flag::Bool
	sx_fourier_constraint_flag::Bool
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
	       sx_attrib=:positive,
	       sx_fix_zero_lag_flag=true,
	       sx_fourier_constraint_flag=false,
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

	if(sx_attrib==:positive)
		sx=X(nt)
		sx.x[1]=1.0
	elseif(sx_attrib==:all)
		sx=X(length(optm.cal.s))
	else
		error("invalid sx_attrib")
	end

	err=DataFrame(g=[], g_nodecon=[], s=[], d=[], whiteness_obs=[], whiteness_cal=[])

	pa=IBD(om, optm, gx, sx, sx_attrib, sx_fix_zero_lag_flag, 
	sx_fourier_constraint_flag,
	:g, verbose, err)

	if(sx_fix_zero_lag_flag)
		if(sx_attrib==:positive)
			# adjust sprecon
			pa.sx.precon[1]=0.0 # do not update zero lag
			pa.sx.preconI[1]=0.0 # do not update zero lag
		elseif(sx_attrib==:all)
			pa.sx.precon[nt]=0.0 # do not update zero lag
			pa.sx.preconI[nt]=0.0 # do not update zero lag
		else
			error("invalid sx_attrib")
		end
	end

 
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
		if(pa.sx_attrib==:all)
			for i in eachindex(x)
				x[i]=pa.optm.cal.s[i]*pa.sx.precon[i] 
			end
			pa.sx_fix_zero_lag_flag && (x[pa.om.nt]=1.0) # zero lag will be fixed
		elseif(pa.sx_attrib==:positive)
			for i in eachindex(x)
				x[i]=pa.optm.cal.s[i+pa.om.nt-1]*pa.sx.precon[i] # just take positive lags
			end
			pa.sx_fix_zero_lag_flag && (x[1]=1.0) # zero lag will be fixed
		else	
			error("invalid sx_attrib")
		end
	else(pa.attrib_inv == :g)
		for i in eachindex(x)
			x[i]=pa.optm.cal.g[i]*pa.gx.precon[i] 		# multiply by gprecon
		end
	end
	return nothing
end


function x_to_model!(x, pa::IBD)
	if(pa.attrib_inv == :s)
		if(pa.sx_attrib==:all)
			for i in eachindex(pa.optm.cal.s)
				pa.optm.cal.s[i]=x[i]*pa.sx.preconI[i]
			end
			pa.sx_fix_zero_lag_flag && (pa.optm.cal.s[pa.om.nt]=1.0) # fix zero lag
		elseif(pa.sx_attrib==:positive)
			for i in 1:pa.om.nt-1
				# put same in positive lags
				pa.optm.cal.s[pa.om.nt+i]=x[i+1]*pa.sx.preconI[i+1]
				# put same in negative lags
				pa.optm.cal.s[pa.om.nt-i]=x[i+1]*pa.sx.preconI[i+1]
			end
			pa.optm.cal.s[pa.om.nt]=x[1]*pa.sx.preconI[1]
			pa.sx_fix_zero_lag_flag && (pa.optm.cal.s[pa.om.nt]=1.0) # fix zero lag
		else
			error("invalid sx_attrib")
		end
	else(pa.attrib_inv == :g)
		for i in eachindex(pa.optm.cal.g)
			pa.optm.cal.g[i]=x[i]*pa.gx.preconI[i]
		end
	end
	return pa
end

"""
Zero out elements of g_ii, depending on nlags
"""
function focus!(pa::IBD, nlags=1)
	nr=pa.om.nr
	ntg=pa.om.ntg
	irr=1  # auto correlation index
	for ir in 1:nr
		for i in nlags:ntg-1    
			# zero out Greens functions at non zero lags
			pa.optm.cal.g[ntg+i,irr]=0.0
			pa.optm.cal.g[ntg-i,irr]=0.0
		end
		irr+=nr-(ir-1)
	end

end


"""
Add a preconditioner to update only a few lags of the g_ii
Total lags that the precon is non-zero is given by 2*nlags-1
"""
function add_focusing_gprecon!(pa::IBD, nlags=1)

	nr=pa.om.nr
	ntg=pa.om.ntg
	gprecon=ones(pa.optm.ntg, pa.optm.nr); 

	irr=1  # auto correlation index
	for ir in 1:nr
		for i in nlags:ntg-1    
			gprecon[ntg+i,irr]=0.0    # put zero at +ve lags
			gprecon[ntg-i,irr]=0.0    # put zero at -ve lags
		end
		irr+=nr-(ir-1)
	end
	add_gprecon!(pa, gprecon)
	return pa
end

"""
Add focusing in pa.
Zero out pa.optm.cal.g for autocorrelations and non zero lags
"""
function add_focusing_gweights!(pa::IBD)

	nr=pa.om.nr
	ntg=pa.om.ntg
	gweights=zeros(pa.optm.ntg, pa.optm.nr); 

	irr=1  # auto correlation index
	for ir in 1:nr
		for i in 1:ntg-1
			gweights[ntg+i,irr]=abs((i)/(ntg-1))
			gweights[ntg-i,irr]=abs((i)/(ntg-1))
		end
		irr+=nr-(ir-1)
	end
	add_gweights!(pa, gweights)
	return pa
end



"""
compute the amount of focusing in cross-correlated g
"""
function whiteness_focusing(pa::IBD, attrib=:obs)
	g=getfield(pa.optm, attrib).g
	nr=pa.om.nr
	ntg=pa.om.ntg

	irr=1  # auto correlation index
	J = 0.0
	for ir in 1:nr
		gg=view(g, :, irr)
		JJ = 0.0
		fact=inv(gg[ntg]*gg[ntg])
		for i in eachindex(gg)
			JJ += gg[i] * gg[i] * fact  * abs((ntg-i)/(ntg    -1))

		end
		J += JJ
		irr+=nr-(ir-1)
	end
	J = J * inv(nr) # scale with number of receivers
	return J
end

function ibd!(pa::IBD)

	initialize!(pa)
	update_all!(pa, max_reroundtrips=1, max_roundtrips=100000, roundtrip_tol=1e-8)

	err!(pa)
end



"""
Focused Blind Deconvolution
"""
function fibd!(pa::IBD, io=STDOUT; verbose=true, α=[Inf, 0.0])

	if(io===nothing)
		logfilename=joinpath(pwd(),string("XFIBD",now(),".log"))
		io=open(logfilename, "a+")
	end

	updates = x -> update_all!(pa, io,                   
				    max_reroundtrips=1, max_roundtrips=100000, 
				    roundtrip_tol=x, verbose=true)

	add_focusing_gweights!(pa) # adds weights
	for (iα, αv) in enumerate(α)
		write(io, string("## (",iα,"/",length(α),") homotopy for alpha=", αv,"\n"))
		# if set α=∞ 
		if(αv==Inf)
			focus!(pa)
			add_focusing_gprecon!(pa) # adds precon
			update_func_grad!(pa,goptim=[:ls,], gαvec=[1.]); 
			updates(1e-10)
			remove_gprecon!(pa, including_zeros=true)  # remove precon
		elseif(αv==0.0)
			remove_gprecon!(pa, including_zeros=true)  # remove precon
			remove_gweights!(pa, including_zeros=true)
			update_func_grad!(pa,goptim=[:ls,], gαvec=[1.]); 
			updates(1e-6)
		else
			remove_gprecon!(pa, including_zeros=true)  # remove precon
			add_focusing_gweights!(pa) # adds weights
			update_func_grad!(pa,goptim=[:ls, :weights], gαvec=[1., αv]); 
			updates(1e-6)
		end
	end
	remove_gweights!(pa, including_zeros=true)
	err!(pa)
end


function initialize!(pa::IBD)
	for i in eachindex(pa.optm.cal.s)
		pa.optm.cal.s[i]=0.0 # +ve lags and -ve lags
	end
	pa.optm.cal.s[pa.om.nt]=1.0 # initialize zero lag to one
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
		if(pa.sx_attrib == :positive)
			# stacking over +ve and -ve lags
			for j in 2:pa.om.nt
				storage[j] += pa.optm.ds[pa.om.nt-j+1] # -ve lags
				storage[j] += pa.optm.ds[pa.om.nt+j-1] # +ve lags
			end
			if(pa.sx_fix_zero_lag_flag)
				storage[1]=0.0
			else
				storage[1]=pa.optm.ds[pa.om.nt]
			end
		elseif(pa.sx_attrib == :all)
			for i in eachindex(storage)
				storage[i] = pa.optm.ds[i] #
			end
			if(pa.sx_fix_zero_lag_flag)
				storage[pa.om.nt]=0.0
			end
		else
			error("invalid attrib" )
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
	if(pa.sx_fourier_constraint_flag)
		pac=pa.optm.cal
		Conv.pad_truncate!(pac.s, pac.spad, pac.slags[1], pac.slags[2], pac.np2, 1)
		A_mul_B!(pac.sfreq, pac.sfftp, pac.spad)

		for i in eachindex(pac.sfreq)
			pac.sfreq[i]=complex(real(pac.sfreq[i]),0.0)
			if(real(pac.sfreq[i]) < 0.0)
				pac.sfreq[i]=complex(0.0,0.0)
			end
		end


		A_mul_B!(pac.spad, pac.sifftp, pac.sfreq)
		Conv.pad_truncate!(pac.s, pac.spad, pac.slags[1], pac.slags[2], pac.np2, -1)
	end
	ress = update!(pa, xs, pa.sx.func, pa.sx.grad!)
	fs = Optim.minimum(ress)
	return fs
end



"""
The cross-correlated Green's functions and the auto-correlated source signature have to be reconstructed exactly, except for a scaling factor.
"""
function err!(pa::IBD, io=STDOUT; cal=pa.optm.cal) 
	(fg, b)	= Misfits.error_after_scaling(cal.g, pa.optm.obs.g) # error upto a scalar b
	(fs, b)	= Misfits.error_after_scaling(cal.s, pa.optm.obs.s) # error upto a scalar b
	f = Misfits.error_squared_euclidean!(nothing, cal.d, pa.optm.obs.d, nothing, norm_flag=true)

	xg_nodecon=hcat(Conv.xcorr(pa.om.d,Conv.P_xcorr(pa.om.nt, pa.om.nr, cglags=[pa.om.ntg-1, pa.om.ntg-1]))...)
	xgobs=hcat(Conv.xcorr(pa.om.g)...) # compute xcorr with reference g
	fg_nodecon = Misfits.error_squared_euclidean!(nothing, xg_nodecon, xgobs, nothing, norm_flag=true)

	push!(pa.err[:whiteness_obs], whiteness_focusing(pa, :obs))
	push!(pa.err[:whiteness_cal], whiteness_focusing(pa, :cal))

	push!(pa.err[:s],fs)
	push!(pa.err[:d],f)
	push!(pa.err[:g],fg)
	push!(pa.err[:g_nodecon],fg_nodecon)
	write(io,"Interferometric Blind Decon Errors\t\n")
	write(io,"==================\n")
	if(io==STDOUT)
		display(pa.err)
	else
		write(io, string(pa.err))
	end
	write(io, "\n")
end 

