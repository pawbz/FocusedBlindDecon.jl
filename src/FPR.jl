
#=
Focused Phase Retrieval

=#

mutable struct FPR
	p_misfit_xcorr::Conv.P_misfit_xcorr
	p_misfit_xcorr_focus::Conv.P_misfit_xcorr
	g::Matrix{Float64}
	dg::Matrix{Float64}
end

function FPR(nt::Int, nr::Int; cymat=nothing, cy=nothing)
	(cymat===nothing) && (cy===nothing) && error("need cy or cymat")
	if(cy===nothing)
		cy=Conv.cgmat(cymat,nr)
	end
	p_misfit_xcorr=Conv.P_misfit_xcorr(nt, nr, Conv.P_xcorr(nt, nr, 
						    norm_flag=false, 
						    ), cy=cy)
	p_misfit_xcorr_focus=Conv.P_misfit_xcorr(nt, nr, Conv.P_xcorr(nt, nr, 
						    norm_flag=false, 
						    cg_indices=[1]), cy=[cy[1]])
	return FPR(p_misfit_xcorr, 
	    p_misfit_xcorr_focus,
	    zeros(nt,nr), zeros(nt, nr))
end

"""
Update `g`, whereever w is non zero
"""
function update!(g::AbstractMatrix{Float64}, w::AbstractMatrix{Float64}, pa::FPR;
		 focus_flag=false,
		 store_trace::Bool=false, 
		 extended_trace::Bool=false, 
	         f_tol::Float64=1e-10, g_tol::Float64=1e-30, x_tol::Float64=1e-30)

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
	res = optimize(x->fg!(nothing, x, pa), (storage, x)->fg!(storage, x, pa), 
		x, 
		ConjugateGradient(),
		#BFGS(),
		       Optim.Options(g_tol = g_tol, f_tol=f_tol, x_tol=x_tol,
		       iterations = 20000, store_trace = store_trace,
		       show_every=250,
		       extended_trace=extended_trace, show_trace=true))
	println(res)
	flush(STDOUT)

	for i in eachindex(g)
		g[i]=Optim.minimizer(res)[i]
	end
	return g
end

"""
* `inear` : nearest receiver index
"""
function fpr!(g::AbstractMatrix{Float64}, inear::Int, pa::FPR; precon=:focus)
	if(precon==:focus)
		println("Phase Retrieval with Focusing Constraint")  
		println("========================================")  
		w1=ones(size(g))
		w1[:,inear]=0.0
		w1[1,inear]=1.0
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

#=

function phase_retrievel!(y, pacse::Misfits.Param_CSE, 
			  w=ones(y),
		 store_trace::Bool=false, 
		 extended_trace::Bool=false, 
	     f_tol::Float64=1e-10, g_tol::Float64=1e-30, x_tol::Float64=1e-30)


	f=function f(x)
		x=reshape(x,nt,nr)
		for i in eachindex(x)
			if(w[i]≠0.0)
				x[i] = x[i]/w[i]
			end
		end
		J=Misfits.error_corr_squared_euclidean!(nothing,x,pacse)
		return J
	end
	g! =function g!(storage, x) 
		x=reshape(x,nt,nr)
		for i in eachindex(x)
			if(w[i]≠0.0)
				x[i] = x[i]/w[i]
			end
		end
		gg=zeros(nt,nr)
		Misfits.error_corr_squared_euclidean!(gg, x, pacse)
		copy!(storage,gg)
		for i in eachindex(x)
			if(w[i]≠0.0)
				storage[i] = storage[i]/w[i]
			else
				storage[i]=0.0
			end
		end
	end


end



function phase_retrievel(Ay, nt, nr,



#	Ayy=[Ay[:,1+(ir-1)*nr:ir*nr]for ir in 1:nr]
	pacse=Misfits.Param_CSE(nt,nr,Ay=Ay)

	# initial w to x
	x=randn(nt*nr)

	return phase_retrievel!(y,pacse)

	"""
	Unbounded LBFGS inversion, only for testing
	"""
	res = optimize(f, g!, x, 
		ConjugateGradient(),
		       Optim.Options(g_tol = g_tol, f_tol=f_tol, x_tol=x_tol,
		       iterations = 1000, store_trace = store_trace,
		       extended_trace=extended_trace, show_trace = true))
	println(res)

	for i in eachindex(y)
		y[i]=Optim.minimizer(res)[i]
	end
	return y

end

=#
