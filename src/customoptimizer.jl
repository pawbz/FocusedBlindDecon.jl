using LinearAlgebra: norm, dot

function gdoptimize(f, g!, fg!, x0::AbstractArray{T}, nmf_flag,
                    maxiter::Int = 1,
                    g_rtol::T = sqrt(eps(T)), g_atol::T = eps(T)) where T <: Number

	linesearch = BackTracking(order=3)	
#	linesearch = StrongWolfe()

	x = copy(x0)
	xx = copy(x0)
	gvec = similar(x)
	g!(gvec, x)
	fx = f(x)

	gnorm = norm(gvec)
	gtol = max(g_rtol*gnorm, g_atol)

	# Univariate line search functions
	function ϕ(α)
		for i in eachindex(xx)
			xx[i] = x[i] + α * s[i]
		end
		nmf_flag && prox!(xx)
		f(xx)
	end
	function dϕ(α)
		g!(gvec, x)
		return dot(gvec, s)
	end
	function ϕdϕ(α)
		for i in eachindex(xx)
			xx[i] = x[i] + α * s[i]
		end
		nmf_flag && prox!(xx)
		phi = fg!(nothing, xx)
		fg!(gvec, x)
		dphi = dot(gvec, s)
		return (phi, dphi)
	end

	s = similar(gvec) # Step direction

	iter = 0
	while iter < maxiter && gnorm > gtol
		iter += 1
		s .= -gvec

		dϕ_0 = dot(s, gvec)
		α, fx = linesearch(ϕ, dϕ, ϕdϕ, 1.0, fx, dϕ_0)


		for i in eachindex(x)
			x[i] = x[i] + α * s[i]
		end
		nmf_flag && prox!(x)
		g!(gvec, x)
		gnorm = norm(gvec)
	end

	copyto!(x0, x)

	return fx
end


