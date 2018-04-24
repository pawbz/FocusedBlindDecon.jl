

"""
Optimization is performed on this model
"""
mutable struct OptimModel
	ntg::Int64
	nt::Int64
	nr::Int64
	obs::Conv.Param{Float64,2,2,1} # observed convolutional model
	cal::Conv.Param{Float64,2,2,1} # calculated convolutional model
	dg::Array{Float64,2} # store gradients
	ds::Vector{Float64} # store gradients
	ddcal::Array{Float64,2}
end
	

function OptimModel(ntg, nt, nr; fftwflag=FFTW.MEASURE, slags=nothing, dlags=nothing, glags=nothing)

	obs=Conv.Param(ssize=[nt], dsize=[nt,nr], gsize=[ntg,nr], 
	slags=slags,
	glags=glags,
	dlags=dlags,
	fftwflag=fftwflag)
	cal=deepcopy(obs)

	# initial values are random
	ds=zeros(cal.s)
	dg=zeros(cal.g)
	ddcal=zeros(cal.d)

	return OptimModel(ntg, nt, nr, obs, cal, dg, ds, ddcal)
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
	Conv.mod!(pao, :d)
end



"""
Store actual 
"""
mutable struct ObsModel
	ntg::Int64
	nt::Int64
	nr::Int64
	g::Array{Float64,2} 
	s::Vector{Float64} 
	d::Array{Float64,2} 
end


function ObsModel(ntg, nt, nr;
	       d=nothing, 
	       g=nothing, 
	       s=nothing)
	(s===nothing) && (s=zeros(nt))
	(g===nothing) && (s=zeros(ntg,nr))
	if(d===nothing)
		(iszero(g) || iszero(s)) && error("need gobs and sobs")
		obstemp=Conv.Param(ssize=[nt], dsize=[nt,nr], gsize=[ntg,nr], 
		     slags=[nt-1, 0])
		copy!(obstemp.g, g)
		copy!(obstemp.s, s)
		Conv.mod!(obstemp, :d) # do a convolution to model data
		d=obstemp.d
	end

	return ObsModel(ntg, nt, nr, g, s, d)
end

"""
Inversion variables
"""
mutable struct X
	x::Vector{Float64}
	last_x::Vector{Float64}
	func::Function
	grad!::Function
	precon::Vector{Float64}
	preconI::Vector{Float64}
	weights::Vector{Float64}
end

function X(n)
	return X(zeros(n), randn(n), x->randn(), x->randn(), ones(n), ones(n), zeros(n))
end
