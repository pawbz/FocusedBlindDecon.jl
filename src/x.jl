


"""
Inversion variables
"""
mutable struct X{T}
	x::Vector{T}
	last_x::Vector{T}
	lower_x::Vector{T}
	upper_x::Vector{T}
	precon::Vector{T}
	preconI::Vector{T}
	weights::Vector{T}
end

function X(n,T)
	return X{T}(zeros(n), randn(n), fill(-Inf,n), fill(Inf,n), ones(n), ones(n), zeros(n))
end


