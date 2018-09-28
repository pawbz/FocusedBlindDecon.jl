

operator(pa, ::S)=pa.opS
operator(pa, ::G)=pa.opG

"""
A linear operator corresponding to convolution with either s or g
"""
function create_operator(pa, attrib)
	global to

	fw=(y,x)->@timeit to "F" F!(y, x, pa, attrib)
	bk=(y,x)->@timeit to "Fadj" Fadj!(y, x, pa, attrib)

	return LinearMap{collect(typeof(pa).parameters)...}(fw, bk, 
		  length(pa.optm.cal.d),  # length of output
		  ninv(pa, attrib),
		  ismutating=true, isposdef=true)
end

function F!(y, x, pa, attrib)
	x_to_model!(x, pa, attrib) # modify pa.optm.cal.s 
	Conv.mod!(pa.optm.cal, Conv.D()) # modify pa.optm.cal.d
	copyto!(y,pa.optm.cal.d)
	return nothing
end

function Fadj!(y, x, pa, attrib)
	copyto!(pa.optm.ddcal,x)
	Fadj!(pa, y, pa.optm.ddcal, attrib)
	return nothing
end




