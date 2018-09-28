struct S end
struct G end


"""
Optimization is performed on this model
"""
mutable struct OptimModel{T}
	ntg::Int64
	nt::Int64
	nts::Int64
	nr::Int64
	obs::Conv.P_conv{T,2,2,1} # observed convolutional model
	cal::Conv.P_conv{T,2,2,1} # calculated convolutional model
	dg::Array{T,2} # store gradients
	ds::Vector{T} # store gradients
	ddcal::Array{T,2}
	dvec::Vector{T}
end
	

function OptimModel(ntg, nt, nr, nts, T; fftwflag=FFTW.MEASURE, slags=nothing, dlags=nothing, glags=nothing)

	obs=Conv.P_conv(T,ssize=[nts], dsize=[nt,nr], gsize=[ntg,nr], 
	slags=slags,
	glags=glags,
	dlags=dlags,
	fftwflag=fftwflag,
	)
	cal=deepcopy(obs)

	# initial values are random
	ds=zero(cal.s)
	dg=zero(cal.g)
	ddcal=zero(cal.d)
	dvec=zeros(nt*nr) # if data needs to be stored as vector 

	return OptimModel{T}(ntg, nt, nts, nr, obs, cal, dg, ds, ddcal, dvec)
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
mutable struct ObsModel{T}
	ntg::Int64
	nt::Int64
	nts::Int64
	nr::Int64
	g::Array{T,2} 
	s::Vector{T} 
	d::Array{T,2} 
end


function ObsModel(ntg, nt, nr, nts, T::DataType;
	       d=nothing, 
	       g=nothing, 
	       s=nothing)
	(s===nothing) && (s=zeros(nts))
	(g===nothing) && (g=zeros(ntg,nr))
	if(d===nothing)
		(iszero(g) || iszero(s)) && error("need gobs and sobs")
		obstemp=Conv.P_conv(T,ssize=[nts], dsize=[nt,nr], gsize=[ntg,nr], slags=[nts-1, 0])
		copyto!(obstemp.g, g)
		copyto!(obstemp.s, s)
		Conv.mod!(obstemp, Conv.D()) # do a convolution to model data
		d=obstemp.d
	end

	return ObsModel{T}(ntg, nt, nts, nr, g, s, d)
end

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
