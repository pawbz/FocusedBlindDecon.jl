







type Intensity
	xaabs::Vector{Float64}
	yfreq::Vector{Complex{Float64}}
	ypad::Vector{Float64}
	fftp::FFTW.rFFTWPlan
	ifftp::FFTW.Plan
	np2::Int64
end


"""
xa is the autocorrelation sequence
"""
function Intensity(xa::Vector{Float64})
	!isodd(length(xa)) && error("length of x should be odd")
	np2=nextfastfft(2*length(xa))
	T=Float64
	nrfft=div(np2,2)+1

	fftp=plan_rfft(zeros(T, np2),[1])
	ifftp=plan_irfft(complex.(zeros(T, nrfft)),np2,[1])

	xapad=zeros(np2)
	ypad=zeros(np2)
	nlag=div(length(xa)-1,2)
	Conv.pad_truncate!(xa, xapad, nlag, nlag, np2, 1)

	xafreq=complex.(zeros(T,nrfft))
	yfreq=complex.(zeros(T,nrfft))
	A_mul_B!(xafreq, fftp, xapad)
	xaabs=sqrt.(abs.(xafreq))

	return Intensity(xaabs, yfreq, ypad, fftp, ifftp, np2)

end


"""
Make minimum changes to the input `y`, such that the constraints in the Fourier domain are satisfied.
Returns the distance between the input and output.
"""
function fourier_constraints!(y, pa=Intensity(y,nextfastfft(length(y))))
	T=Float64

	#println("projecting.....")

	pa.yfreq[:] = complex(T(0))
	pa.ypad[:]=T(0)
	Conv.pad_truncate!(y, pa.ypad, length(y)-1, 0, pa.np2, 1)
	A_mul_B!(pa.yfreq, pa.fftp, pa.ypad)
	dist=0.0
	for i in eachindex(pa.xaabs)
		dist += (abs(pa.yfreq[i]) - pa.xaabs[i])^2
	end
	# the amplitude spectrum in yfreq will be same as that of xfreq
	for i in eachindex(pa.xaabs)
		pa.yfreq[i]=pa.yfreq[i]*inv(abs(pa.yfreq[i]))*(pa.xaabs[i])
	end

	A_mul_B!(pa.ypad, pa.ifftp, pa.yfreq)

	return dist
end

function support_constraints!(yout, pa::Intensity)
	Conv.pad_truncate!(yout, pa.ypad, length(yout)-1, 0, pa.np2, -1)
	dist=0.0
	for i in length(yout)+1:length(pa.ypad)
		dist += pa.ypad[i]*pa.ypad[i]
	end
	return dist
end



function phase_retrievel(Ay, nt, nr,
		 store_trace::Bool=false, 
		 extended_trace::Bool=false, 
	     f_tol::Float64=1e-8, g_tol::Float64=1e-30, x_tol::Float64=1e-30)



#	Ayy=[Ay[:,1+(ir-1)*nr:ir*nr]for ir in 1:nr]
	pacse=Misfits.Param_CSE(nt,nr,Ay=Ay)
	f=function f(x)
		x=reshape(x,nt,nr)
		J=Misfits.error_corr_squared_euclidean!(nothing,x,pacse)
		return J
	end
	g! =function g!(storage, x) 
		x=reshape(x,nt,nr)
		gg=zeros(nt,nr)
		Misfits.error_corr_squared_euclidean!(gg, x, pacse)
		copy!(storage,gg)
	end

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

	nt=size(y,1)
	nr=size(y,2)
	# initial w to x
	x=randn(nt*nr)
	for i in eachindex(x)
		x[i]=y[i]
	end

	"""
	Unbounded LBFGS inversion, only for testing
	"""
	res = optimize(f, g!, x, 
		#ConjugateGradient(),
		BFGS(),
		       Optim.Options(g_tol = g_tol, f_tol=f_tol, x_tol=x_tol,
		       iterations = 5000, store_trace = store_trace,
		       extended_trace=extended_trace, show_trace = true))
	#println(res)

	for i in eachindex(y)
		y[i]=Optim.minimizer(res)[i]
	end
	return y

end



