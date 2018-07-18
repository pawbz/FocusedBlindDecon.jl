#=
This file contains routines that are common to both BD and IBD
=#

function func_grad!(storage, x::AbstractVector{Float64},pa)

	# x to pa.optm.cal.s or pa.optm.cal.g 
	x_to_model!(x, pa)

	F!(pa, x) # forward

	if(storage === nothing)
		# compute misfit and δdcal
		f = Misfits.error_squared_euclidean!(nothing, pa.optm.cal.d, pa.optm.obs.d, 
				       nothing, norm_flag=false)
	else
		f = Misfits.error_squared_euclidean!(pa.optm.ddcal, pa.optm.cal.d, 
				       pa.optm.obs.d, nothing, norm_flag=false)
		Fadj!(pa, x, storage, pa.optm.ddcal)
	end
	return f

end


# exponential-weighted norm for the green functions
function func_grad_g_weights!(storage, x, pa)
	x_to_model!(x, pa)
	!(pa.attrib_inv == :g) && error("only for g inversion")
	if(!(storage === nothing)) #
		f = Misfits.error_weighted_norm!(pa.optm.dg,pa.optm.cal.g, pa.gx.weights) #
		for i in eachindex(storage)
			storage[i]=pa.optm.dg[i]
		end
	else	
		f = Misfits.error_weighted_norm!(nothing,pa.optm.cal.g, pa.gx.weights)
	end
	return f
end

# exponential-weighted norm for the green functions
function func_grad_g_acorr_weights!(storage, x, pa)
	x_to_model!(x, pa)
	!(pa.attrib_inv == :g) && error("only for g inversion")

	if(!(storage === nothing)) #
		f = Misfits.error_acorr_weighted_norm!(pa.optm.dg,pa.optm.cal.g, 
					 paconv=pa.g_acorr,dfds=pa.optm.dg_acorr) #
		for i in eachindex(storage)
			storage[i]=pa.optm.dg[i]
		end
	else	
		f = Misfits.error_acorr_weighted_norm!(nothing,pa.optm.cal.g, 
					 paconv=pa.g_acorr,dfds=pa.optm.dg_acorr)
	end

	return f
end
#  


