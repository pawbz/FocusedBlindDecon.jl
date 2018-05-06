







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





