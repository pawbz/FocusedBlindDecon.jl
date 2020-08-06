
mutable struct LinearModel{T}
	d::Array{T,2}
	g::Array{T,2}
	s::Array{T,2}
end

mutable struct LRAOpt{T}
	nt::Int64
	nr::Int64
	nra::Int64 # reduced dimension
	dg::Array{T,2} # store gradients
	ds::Array{T,2} # store gradients
	ddcal::Array{T,2} # store error
	dvec::Vector{T}
	obs::LinearModel{T}
	cal::LinearModel{T}
end


function LinearModel(nt,nr,nra,T;
	       d=nothing, 
	       g=nothing, 
	       s=nothing)
	(s===nothing) && (s=zeros(T,nt,nra))
	(g===nothing) && (g=zeros(T,nra,nr))
	d=s*g
	return LinearModel(d,g,s)
end

function LRAOpt(nt,nr,nra,T)
	obs=LinearModel(nt,nr,nra,T)
	cal=LinearModel(nt,nr,nra,T)
	dg=zero(obs.g)
	ds=zero(obs.s)
	ddcal=zero(obs.d)
	dvec=zeros(nt*nr) # if data needs to be stored as vector 
	return LRAOpt(nt,nr,nra,dg,ds,ddcal,dvec,obs,cal)
end
