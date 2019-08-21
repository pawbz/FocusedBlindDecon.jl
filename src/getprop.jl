
"""
Let `pa` be in instance of the FBD model.
Its properties can be easily accessed as using the following `getindex` calls.

**`pa[:s]`** returns the source, either estimated after `fbd!` or `lsbd!`, from the model.

**`pa[:sa]`** returns the auto-correlated source, estimated after `fibd!`.

**`pa[:g]`** returns the channel impulse responses, estimated after `fbd!` or `lsbd!`.

**`pa[:xg]`** returns the interferometric channel impulse responses estimated after `fibd!`.

**`pa[:d]`** returns the convolution of `pa[:s]` and `pa[:g]`.

**`pa[:xd]`** returns the convolution of `pa[:sa]` and `pa[:xg]`.

**`pa[:sobs]`** returns the true source, when stored.

**`pa[:gobs]`** returns the true impulse responses, when stored.

**`pa[:xgobs]`** returns the true interferometric channel impulse responses, if stored.

**`pa[:dobs]`** returns the measured channel outputs that `fbd!` or `lsbd!` aims to factorize.

**`pa[:xdobs]`** returns the measured interferometric channel outputs that `fibd!` aims to factorize.
"""
function Base.getindex(pa::P_fbd, s::Symbol)
	@assert s in [:s, :sobs, :g, :gobs, :d, :dobs, :xg, :xgobs, :sa, :saobs, :xd, :xdobs]
	if(s==:s)
		return pa.plsbd.optm.cal.s
	elseif(s==:sobs)
		return pa.plsbd.optm.obs.s
	elseif(s==:sa)
		sa=copy(pa.pfibd.optm.cal.s)
		sa[div(length(sa),2)+1]=1.0
		return sa
	elseif(s==:saobs)
		return pa.pfibd.optm.obs.s
	elseif(s==:g)
		return pa.plsbd.optm.cal.g
	elseif(s==:xg)
		return pa.pfibd.optm.cal.g
	elseif(s==:xgobs)
		return pa.pfibd.optm.obs.g
	elseif(s==:gobs)
		return pa.plsbd.optm.obs.g
	elseif(s==:d)
		return pa.plsbd.optm.cal.d
	elseif(s==:dobs)
		return pa.plsbd.optm.obs.d
	elseif(s==:xd)
		return pa.pfibd.optm.cal.d
	elseif(s==:xdobs)
		return pa.pfibd.optm.obs.d
	end
end

