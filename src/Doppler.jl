

"""
xx = x cos(δω t)
"""
function doppler!(xx, x, δω, )
	for i in eachindex(x)
		xx[i] = x[i] * cos(δω*i)
	end
end 




"""

"""
