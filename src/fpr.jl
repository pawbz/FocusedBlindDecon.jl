
#=
Focused Phase Retrieval

=#

mutable struct FPR{T}
	p_misfit_xcorr::Conv.P_misfit_xcorr{T}
	p_misfit_xcorr_focus::Conv.P_misfit_xcorr{T}
	g::Matrix{T}
	dg::Matrix{T}
	gobs::Matrix{T}
	f_index_loaded::Vector{Float64} # to store the functional during focusing at different
	index_loaded::Int32
	err::DataFrames.DataFrame
end

function update_cymat!(pa::FPR; cymat=nothing, cy=nothing, gobs=nothing, nearest_receiver=nothing)
	(cymat===nothing) && (cy===nothing) && error("need cy or cymat")
	if(nearest_receiver===nothing)
		if(gobs===nothing)
			@info "assuming ir=1 is the nearest recevier"
			ine=1
		else
			copyto!(pa.gobs, gobs)
			ine=inear(gobs)
		end
	else
		ine=nearest_receiver
	end
	nr=size(pa.g,2)
	if(cy===nothing)
		cyshifted=Conv.cgmat(cymat,nr,cg_indices=circshift(1:nr, -ine+1))
	else
		cymattemp=Conv.cgmat(cy,nr)
		cyshifted=Conv.cgmat(cymattemp,nr,cg_indices=circshift(1:nr, -ine+1))
	end

	# put the shifted cy in pa
	for i in eachindex(pa.p_misfit_xcorr.cy)
		copyto!(pa.p_misfit_xcorr.cy[i], cyshifted[i])
	end
	copyto!(pa.p_misfit_xcorr_focus.cy[1], cyshifted[1])

	return pa
end

function FPR(nt::Int, nr::Int; 
	     cymat=nothing, cy=nothing, 
	       index_loaded=1,
	     gobs=nothing, nearest_receiver=nothing)

	cyshifted=Conv.cgmat(randn(2*nt-1,binomial(nr, 2)+nr),nr)

	p_misfit_xcorr=Conv.P_misfit_xcorr(nt,nr,Conv.P_xcorr(nt,nr, 
						    norm_flag=false, 
						    ),cy=cyshifted)
	p_misfit_xcorr_focus=Conv.P_misfit_xcorr(nt, nr, Conv.P_xcorr(nt, nr, 
						    norm_flag=false, 
						    cg_indices=[1]), cy=[cyshifted[1]])


	# if either of cymat or cy are present, update... otherwise, do that later
	if(!((cymat===nothing) && (cy===nothing)))
		update_cymat!(pa::FPR; cymat=cymat, cy=cy, 
			gobs=gobs, nearest_receiver=nearest_receiver)
	end

	err=DataFrame(g=[])

	if((gobs===nothing))
		gobs=zeros(nt, nr)
	end

	return FPR(p_misfit_xcorr, 
	    p_misfit_xcorr_focus,
	    zeros(nt,nr), zeros(nt, nr), gobs, zeros(nt), Int32(index_loaded), err)
end

"""
Update `g`, whereever w is non zero
"""
function update!(g::AbstractMatrix, w::AbstractMatrix, pa::FPR;
	 focus_flag=false, log_file=false, show_trace=false, g_tol=1e-4, eps=0.5)

	if(focus_flag)
		pax=pa.p_misfit_xcorr_focus
	else
		pax=pa.p_misfit_xcorr
	end

	f=function fg!(storage, x, pa)
		for i in eachindex(x)
			if(w[i]≠0.0)
				pa.g[i]=x[i]/w[i]
			end
		end

		if(storage===nothing)
			J=Conv.func_grad!(storage, pa.g, pax)
			J2=Misfits.front_load!(storage,pa.g)
		else
			J=Conv.func_grad!(pa.dg, pa.g, pax)
			for i in eachindex(storage)
				storage[i]=(1.0-eps)*pa.dg[i]
			end

			J2=Misfits.front_load!(pa.dg, pa.g)
			for i in eachindex(storage)
				storage[i]+=eps*pa.dg[i]
			end

			for i in eachindex(x)
				if(w[i]≠0.0)
					storage[i] = storage[i]/w[i]
				else
					storage[i]=0.0
				end
			end


		end

		return (1.0-eps)*J+eps*J2
	end

	nt=size(g,1)
	nr=size(g,2)

	# initial x
	x=randn(nt*nr)
	for i in eachindex(x)
		x[i]=g[i]
	end

	"""
	Unbounded LBFGS inversion, only for testing
	"""
	res = optimize(xx->fg!(nothing, xx, pa), (storage, xx)->fg!(storage, xx, pa), 
		x, 
		ConjugateGradient(),
		Optim.Options(iterations = 10000, g_tol=g_tol,
			show_every=250,
			store_trace=true,
			callback=x->err!(pa,pa.g),
		       show_trace=show_trace))
	if(show_trace)
		println(res)
	end
	flush(stdout)

	for i in eachindex(g)
		g[i]=Optim.minimizer(res)[i]
		pa.g[i]=Optim.minimizer(res)[i]
	end

	if(log_file)
		# save log file
		PRlogfile=joinpath(pwd(),string("PR",Dates.now(),".log"))
		CSV.write(PRlogfile, 
		   DataFrame(hcat(0:Optim.iterations(res), Optim.f_trace(res))))
	end

	return Optim.minimum(res)
	#return maximum(abs, g[:,1])
end

"""
"""
function fpr!(g::AbstractMatrix, pa::FPR, io=stdout; precon=[:focus,:pr], 
	      show_trace=false, index_loaded, g_tol=1e-4, eps=0.9:-0.1:0.1,
	      w=nothing)


	if(io===nothing)
		logfilename=joinpath(pwd(),string("XFPR",Dates.now(),".log"))
		io=open(logfilename, "a+")
	end

	ine=1;
	if(:focus ∈ precon)
		if(show_trace)
			write(io, "Focused Phase Retrieval\n")  
			write(io, "========================================\n")  
		end
		w1=ones(size(g))
		w1[:,ine] .= 0.0
		w1[index_loaded,ine]=1.0
		# initialize g such that g_near is focused
		for i in eachindex(g)
			if(w1[i]==0.0)
				g[i]=0.0
			else
				g[i]=randn()
			end
		end
		f=update!(g,w1,pa, focus_flag=true, show_trace=show_trace, 
	    		g_tol=1e-8, 
		    eps=0.);
	end

	if(:pr ∈ precon)
		if(show_trace)
			write(io, "Regular Phase Retrieval\n")  
			write(io, "=======================\n")  
		end
		w2=ones(size(g))
		f=update!(g,w2,pa, focus_flag=false, show_trace=show_trace, 
			    g_tol=g_tol, eps=0.0);
	end

	"""
	Focus all the channels simultaneously 
	"""
	if(:focus_all ∈ precon)
		if(w===nothing)
			w2=ones(size(g))
		else
			w2=w
		end
		for eps1 in eps
			if(show_trace)
				write(io, "Focused Phase Retrieval with eps\t",
	  						string(eps1),"\n")  
				write(io, "=======================\n")  
			end
			f=update!(g, w2, pa, focus_flag=false, show_trace=show_trace, 
			    g_tol=g_tol, eps=eps1);
		end
	end

	return f
end

function update_f_index_loaded!(pa::FPR)
	fill!(pa.f_index_loaded, 0.0)
	for it in 1:size(pa.g,1)
		pa.f_index_loaded[it]=fpr!(pa.g,  pa, 
		  precon=[:focus], 
		index_loaded=it, 
		       g_tol=1e-8, show_trace=false)
	end
	pa.index_loaded=argmax(pa.f_index_loaded)
end


function err!(pa::FPR, g) 
	fg = Misfits.error_after_translation(g, pa.gobs)[1]
	push!(pa.err[!,:g],fg)
	return false
end
