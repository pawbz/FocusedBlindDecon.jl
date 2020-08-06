# Low Rank Approximation


mutable struct LRA{T}
	om::LinearModel{T}
	optm::LRAOpt{T}
	gx::X{T}
	sx::X{T}
	opG::LinearMaps.LinearMap{T}
	opS::LinearMaps.LinearMap{T}
	err::DataFrames.DataFrame
end




"""
Constructor for the blind deconvolution problem
"""
function LRA(nt, nr, nra; 
	       dobs=nothing, gobs=nothing, sobs=nothing, verbose=false, attrib_inv=:g,
	       ) 
	# determine type of IBD
	if(!(dobs===nothing))
		(size(dobs)≠(nt,nr)) && error("dobs size error")
		T=eltype(dobs)
	else
		T1=eltype(sobs)
		T2=eltype(gobs)
		(T1≠T2) ? error("type difference") : (T=T1)
		(size(sobs)≠(nt,nra)) && error("sobs size error")
		(size(gobs)≠(nra,nr)) && error("gobs size error")
	end

	# store observed data
	om=LinearModel(nt,nr,nra,T, d=dobs, g=gobs, s=sobs)

	# create models depending on mode
	optm=LRAOpt(nt,nr,nra,T)

	# inversion variables allocation
	gx=X(length(optm.cal.g), T)
	sx=X(length(optm.cal.s), T)

	err=DataFrame(g=[], s=[], d=[])

	# dummy
	opG=LinearMap{T}(x->0.0, y->0.0,1,1, ismutating=true)
	opS=LinearMap{T}(x->0.0, y->0.0,1,1, ismutating=true)

	pa=LRA(om,optm,sx,gx,opG,opS,err)


	# update operators
	pa.opG=create_operator(pa, G())
	pa.opS=create_operator(pa, S())

	!(gobs===nothing) && copyto!(pa.optm.obs.g, gobs)
	!(sobs===nothing) && copyto!(pa.optm.obs.s, sobs)
	if(dobs===nothing)
		mul!(pa.optm.obs.d,pa.optm.obs.s,pa.optm.obs.g)
	else
		copyto!(pa.optm.obs.d, dobs)
	end

	initialize!(pa)


	return pa
	
end



function update_prepare!(pa::LRA, ::S)
end
function update_prepare!(pa::LRA, ::G)
end
function update_finalize!(pa::LRA, ::S)
end
function update_finalize!(pa::LRA, ::G)
end

function bd!(pa::LRA, io=stdout; tol=1e-6)

	if(io===nothing)
		logfilename=joinpath(pwd(),string("XBD",Dates.now(),".log"))
		io=open(logfilename, "a+")
	end

	update_all!(pa, io, max_reroundtrips=1, max_roundtrips=100000, roundtrip_tol=tol)

	err!(pa)
end


function F!(y, x, pa::LRA, ::S)
	for i in eachindex(x)
		pa.optm.cal.g[i]=x[i]
	end
	mul!(pa.optm.cal.d, pa.optm.cal.s, pa.optm.cal.g)
	copyto!(y,pa.optm.cal.d)
	return nothing
end

function F!(y, x, pa::LRA, ::G)
	for i in eachindex(x)
		pa.optm.cal.s[i]=x[i]
	end
	mul!(pa.optm.cal.d, pa.optm.cal.s, pa.optm.cal.g)
	copyto!(y,pa.optm.cal.d)
	return nothing
end

function F!(pa::LRA,x::AbstractVector, attrib::S)
	compute=(x!=pa.sx.last_x)
	if(compute)
		x_to_model!(x, pa, attrib) #
		copyto!(pa.sx.last_x, x)
		mul!(pa.optm.cal.d, pa.optm.cal.s, pa.optm.cal.g)
		return pa
	end
end

function F!(pa::LRA,x::AbstractVector, attrib::G)
	compute=(x!=pa.gx.last_x)
	if(compute)
		x_to_model!(x, pa, G())
		copyto!(pa.gx.last_x, x)
		mul!(pa.optm.cal.d, pa.optm.cal.s, pa.optm.cal.g)
		return pa
	end
end



function Fadj!(y, x, pa::LRA, ::S)
	copyto!(pa.optm.ddcal,x)
	mul!(pa.optm.cal.g,transpose(pa.optm.cal.s),pa.optm.ddcal)
	copyto!(y,pa.optm.cal.g)
	return nothing
end


function Fadj!(y, x, pa::LRA, ::G)
	copyto!(pa.optm.ddcal,x)
	mul!(pa.optm.cal.s,pa.optm.ddcal,transpose(pa.optm.cal.g))
	copyto!(y,pa.optm.cal.s)
	return nothing
end

function x_to_model!(x, pa::LRA, ::G)
	for i in eachindex(pa.optm.cal.g)
		pa.optm.cal.g[i]=x[i]*pa.gx.preconI[i]
	end
	return nothing
end
function x_to_model!(x, pa::LRA, ::S)
	for i in eachindex(pa.optm.cal.s)
		pa.optm.cal.s[i]=x[i]*pa.gx.preconI[i]
	end
	return nothing
end

function model_to_x!(x, pa::LRA, ::S)
	for i in eachindex(x)
		x[i]=pa.optm.cal.s[i]*pa.sx.precon[i]
	end
	return nothing
end
function model_to_x!(x, pa::LRA, ::G)
	for i in eachindex(x)
		x[i]=pa.optm.cal.g[i]*pa.gx.precon[i] 
	end
	return nothing
end


function initialize!(pa::LRA)
	for i in eachindex(pa.optm.cal.g)
		x=(pa.gx.precon[i]≠0.0) ? randn() : 0.0
		pa.optm.cal.g[i]=x
	end

	for i in eachindex(pa.optm.cal.s)
		x=(pa.sx.precon[i]≠0.0) ? randn() : 0.0
		pa.optm.cal.s[i]=x
	end
end


