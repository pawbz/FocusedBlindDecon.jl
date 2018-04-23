

@userplot Plot

@recipe function f(p::Plot; rvec=nothing, δt=1.0, attrib=:0)
	pa=p.args[1]
	(rvec===nothing) && (rvec=1:pa.nr)


	# observed G
	if(isdefined(pa, :gobs))
		gobs=pa.gobs
	elseif(isdefined(pa, :obs1))
		gobs=pa.obs1.g
	end

	# observed S
	if(isdefined(pa, :gobs))
		gobs=pa.gobs
	elseif(isdefined(pa, :obs1))
		gobs=pa.obs1.g
	end





	gxobs=zeros(2*size(pa.obs.g,1)-1, size(pa.obs.g,2))
	gxcal=similar(gxobs)
	scobs=vecnorm(pa.obs.g[:,1])^2
	sccal=vecnorm(pa.cal.g[:,1])^2
	for ir in 1:pa.nr
		gxobs[:,ir] = xcorr(pa.obs.g[:,1], pa.obs.g[:, ir])/scobs
		gxcal[:,ir] = xcorr(pa.cal.g[:,1], pa.cal.g[:, ir])/sccal
	end
	#
	g=collect(1:pa.ntg)*δt
	gx=collect(-pa.ntg+1:1:pa.ntg-1)*δt

	# time vectors
	# autocorr s
	asobs=autocor(pa.obs.s[:,1], 1:pa.nt-1, demean=true)
	as=autocor(pa.cal.s[:,1], 1:pa.nt-1, demean=true)
	sli=max(maximum(abs,asobs), maximum(abs,as))
	# autocorr g 
	agobs=autocor(pa.obs.g,1:pa.ntg-1, demean=true)
	ag=autocor(pa.cal.g,1:pa.ntg-1, demean=true)
	gli=max(maximum(abs,agobs), maximum(abs,ag))

#	asobs=asobs[1:fact:ns] # resample
#	as=as[1:fact:ns] # resample

	#fact=(pa.nt*pa.nr>1000) ? round(Int,pa.nt*pa.nr/1000) : 1
	# cut receivers
#	dcal=pa.cal.d[1:fact:pa.nt*pa.nr]
#	dobs=pa.obs.d[1:fact:pa.nt*pa.nr]


	if(attrib==:0)
		layout := (3,1)

		@series begin        
			subplot := 1
	#		aspect_ratio := :auto
			legend := false
	#		l := :plot
			title := "\$g_0\$"
			w := 1
			pa.gobs
		end
		@series begin        
			subplot := 2
	#		aspect_ratio := :auto
			legend := false
			title := "\$s_0\$"
			w := 1
			pa.sobs
		end
		@series begin        
			subplot := 3
	#		aspect_ratio := :auto
			legend := false
			title := "\$d_i\$"
			w := 1
			pa.dobs
		end
	end

	if((attrib==:obs) || (attrib==:cal))
		ns=length(pa.sobs)
		fact=(ns>1000) ? round(Int,ns/1000) : 1
		fnrp=(pa.nr>10) ? round(Int,pa.nr/10) : 1
		layout := (3,1)

		@series begin        
			subplot := 1
	#		aspect_ratio := :auto
			legend := false
	#		l := :plot
			title := "\$g_0\$"
			w := 1
			getfield(pa,attrib).g[:,1:fnrp:end]
		end
		@series begin        
			subplot := 2
	#		aspect_ratio := :auto
			legend := false
			title := "\$s_0\$"
			w := 1
			getfield(pa,attrib).s[1:fact:end,1]
		end
		@series begin        
			subplot := 3
	#		aspect_ratio := :auto
			legend := false
			title := "\$d_i\$"
			w := 1
			getfield(getfield(pa,attrib),fieldnames(getfield(pa,attrib))[5])[1:fact:end,1:fnrp:end]
		end
	end


	if(attrib==:x)
		ns=length(pa.sobs)
		fact=(ns>1000) ? round(Int,ns/1000) : 1

		fnrp=(pa.nr>10) ? round(Int,pa.nr/10) : 1

		xsobs=pa.obs.s[1:fact:end,1]
		xscal=pa.cal.s[1:fact:end,1]
		xdcal=getfield(pa.cal,fieldnames(pa.cal)[5])[1:fact*pa.nr:end]
	        xdobs=getfield(pa.obs,fieldnames(pa.obs)[5])[1:fact*pa.nr:end]
		xgcal=pa.cal.g
		xgobs=pa.obs.g
		layout := (2,2)
		@series begin        
			subplot := 1
#			aspect_ratio := :equal
			seriestype := :scatter
			title := "scatter s"
			legend := false
			xsobs, xscal
		end
		@series begin        
			subplot := 2
#			aspect_ratio := :equal
			seriestype := :scatter
			title := "scatter g"
			legend := false
			xgobs[:,1:fnrp:end], xgcal[:,1:fnrp:end]
		end

		@series begin        
			subplot := 3
#			aspect_ratio := :equal
			seriestype := :scatter
			title := "scatter d"
			legend := false
			xdobs, xdcal
		end
	end

	if(attrib==:precon)
		layout := (3,1)
		@series begin        
			subplot := 1
	#		aspect_ratio := :auto
			legend := false
			title := "gprecon"
			w := 1
			pa.gprecon
		end
		@series begin        
			subplot := 2
	#		aspect_ratio := :auto
			legend := false
			title := "sprecon"
			w := 1
			pa.sprecon
		end
		@series begin        
			subplot := 3
	#		aspect_ratio := :auto
			legend := false
			title := "gweights"
			w := 1
			pa.gweights
		end




	end
end


