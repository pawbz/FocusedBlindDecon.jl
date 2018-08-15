struct S end
struct G end

mutable struct FourierConstraints
	window::Vector{Float64}
	autocorr_flag::Bool
end



"""
Optimization is performed on this model
"""
mutable struct OptimModel
	ntg::Int64
	nt::Int64
	nts::Int64
	nr::Int64
	obs::Conv.P_conv{Float64,2,2,1} # observed convolutional model
	cal::Conv.P_conv{Float64,2,2,1} # calculated convolutional model
	dg::Array{Float64,2} # store gradients
	ds::Vector{Float64} # store gradients
	ddcal::Array{Float64,2}
end
	

function OptimModel(ntg, nt, nr, nts; fftwflag=FFTW.MEASURE, slags=nothing, dlags=nothing, glags=nothing)

	obs=Conv.P_conv(ssize=[nts], dsize=[nt,nr], gsize=[ntg,nr], 
	slags=slags,
	glags=glags,
	dlags=dlags,
	fftwflag=fftwflag)
	cal=deepcopy(obs)

	# initial values are random
	ds=zeros(cal.s)
	dg=zeros(cal.g)
	ddcal=zeros(cal.d)

	return OptimModel(ntg, nt, nts, nr, obs, cal, dg, ds, ddcal)
end

function replace!(pa::OptimModel, x, fieldmod::Symbol, field::Symbol)
	pao=getfield(pa, fieldmod)
	paoo=getfield(pao, field)
	if(length(paoo)==length(x))
		for i in eachindex(paoo)
			paoo[i]=x[i]
		end
	else
		error("error dimension replacing")
	end
	# perform modelling :<)
	Conv.mod!(pao, Conv.D())
end



"""
Store actual 
"""
mutable struct ObsModel
	ntg::Int64
	nt::Int64
	nts::Int64
	nr::Int64
	g::Array{Float64,2} 
	s::Vector{Float64} 
	d::Array{Float64,2} 
end


function ObsModel(ntg, nt, nr, nts;
	       d=nothing, 
	       g=nothing, 
	       s=nothing)
	(s===nothing) && (s=zeros(nts))
	(g===nothing) && (g=zeros(ntg,nr))
	if(d===nothing)
		(iszero(g) || iszero(s)) && error("need gobs and sobs")
		obstemp=Conv.P_conv(ssize=[nts], dsize=[nt,nr], gsize=[ntg,nr], slags=[nts-1, 0])
		copyto!(obstemp.g, g)
		copyto!(obstemp.s, s)
		Conv.mod!(obstemp, Conv.D()) # do a convolution to model data
		d=obstemp.d
	end

	return ObsModel(ntg, nt, nts, nr, g, s, d)
end

"""
Inversion variables
"""
mutable struct X
	x::Vector{Float64}
	last_x::Vector{Float64}
	precon::Vector{Float64}
	preconI::Vector{Float64}
	weights::Vector{Float64}
end

function X(n)
	return X(zeros(n), randn(n), ones(n), ones(n), zeros(n))
end
