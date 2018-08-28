
#=
Focused Phase Retrieval

=#

mutable struct FPR
	p_misfit_xcorr::Conv.P_misfit_xcorr
	p_misfit_xcorr_focus::Conv.P_misfit_xcorr
	g::Matrix{Float64}
	dg::Matrix{Float64}
end

function update_cymat!(pa::FPR; cymat=nothing, cy=nothing, gobs=nothing, nearest_receiver=nothing)
	(cymat===nothing) && (cy===nothing) && error("need cy or cymat")
	if(nearest_receiver===nothing)
		if(gobs===nothing)
			info("assuming ir=1 is the nearest recevier")
			ine=1
		else
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
	     gobs=nothing, nearest_receiver=nothing)

	cyshifted=Conv.cgmat(randn(2*nt-1,binomial(nr, 2)+nr),nr)

	p_misfit_xcorr=Conv.P_misfit_xcorr(nt, nr, Conv.P_xcorr(nt, nr, 
						    norm_flag=false, 
						    ), cy=cyshifted)
	p_misfit_xcorr_focus=Conv.P_misfit_xcorr(nt, nr, Conv.P_xcorr(nt, nr, 
						    norm_flag=false, 
						    cg_indices=[1]), cy=[cyshifted[1]])


	# if either of cymat or cy are present, update... otherwise, do that later
	if(!((cymat===nothing) && (cy===nothing)))
		update_cymat!(pa::FPR; cymat=cymat, cy=cy, gobs=gobs, nearest_receiver=nearest_receiver)
	end

	return FPR(p_misfit_xcorr, 
	    p_misfit_xcorr_focus,
	    zeros(nt,nr), zeros(nt, nr))
end

"""
Update `g`, whereever w is non zero
"""
function update!(g::AbstractMatrix{Float64}, w::AbstractMatrix{Float64}, pa::FPR;
		 focus_flag=false)

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
		else
			J=Conv.func_grad!(pa.dg, pa.g, pax)
			for i in eachindex(storage)
				storage[i]=pa.dg[i]
			end
			for i in eachindex(x)
				if(w[i]≠0.0)
					storage[i] = storage[i]/w[i]
				else
					storage[i]=0.0
				end
			end
		end

		return J
	end

	nt=size(g,1)
	nr=size(g,2)

	# initial w to x
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
		Optim.Options(iterations = 10000, g_tol=1e-4,
			show_every=250,
		       show_trace=true))
	println(res)
	flush(stdout)

	for i in eachindex(g)
		g[i]=Optim.minimizer(res)[i]
	end
	return g
end

"""
"""
function fpr!(g::AbstractMatrix{Float64}, pa::FPR; precon=:focus)
	ine=1;
	if(precon==:focus)
		println("Phase Retrieval with Focusing Constraint")  
		println("========================================")  
		w1=ones(size(g))
		w1[:,ine]=0.0
		w1[1,ine]=1.0
		# initialize g such that g_near is focused
		for i in eachindex(g)
			if(w1[i]==0.0)
				g[i]=0.0
			else
				g[i]=randn()
			end
		end
		update!(g,w1,pa, focus_flag=true);
	end

	println("Regular Phase Retrieval")  
	println("=======================")  
	w2=ones(size(g))
	update!(g,w2,pa, focus_flag=false);
end

