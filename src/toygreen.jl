
"""
Construct Toy Green's functions
Decaying peaks, control number of events, and their positions, depending on bfrac and efrac.
* `afrac` : control amplitude of peaks
"""
function toy_reflec_green!(x; bfrac=0.0, c=1.0, afrac=1.0)
	nt=size(x,1)
	nr=size(x,2)
	c=c*sqrt((nr^2)/(nt-1))

	it0=1+round(Int,bfrac*nt)
	for ir in 1:nr
		it=round(Int, it0+(ir*ir)*inv(c)*inv(c))
		if(it<=nt)
			x[it, ir]=1.0*afrac
		end
	end
	return x
end

function toy_direct_green!(x; bfrac=0.0, c=1.0, afrac=1.0)
	nt=size(x,1)
	nr=size(x,2)
	c=c*((nr)/(nt-1))

	it0=1+round(Int,bfrac*nt)
	for ir in 1:nr
		it=round(Int, it0+(ir)*inv(c))
		if(it<=nt)
			x[it, ir]=1.0*afrac
		end
	end
	return x

end


