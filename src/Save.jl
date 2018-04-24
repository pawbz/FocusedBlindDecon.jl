
"""
Save Param
"""
function save(pa, folder; tgridg=nothing, tgrid=nothing)
	!(isdir(folder)) && error("invalid directory")


	(tgridg===nothing) && (tgridg = Grid.M1D(0.0, (pa.ntg-1)*1.0, pa.ntg))

	# save original g
	file=joinpath(folder, "gobs.csv")
	CSV.write(file,DataFrame(hcat(tgridg.x, pa.om.g)))
	# save for imagesc
	file=joinpath(folder, "imgobs.csv")
	CSV.write(file,DataFrame(hcat(repeat(tgridg.x,outer=pa.nra),
				      repeat(1:pa.nra,inner=pa.ntg),vec(pa.om.g))),)
	# file=joinpath(folder, "gcal.csv")
	# CSV.write(file,DataFrame(hcat(tgridg.x, pa.optm.calsave.g)))

	# compute cross-correlations of g
	xtgridg=Grid.M1D_xcorr(tgridg); # grid
	if(pa.mode ∈ [:bd, :bda])
		xgobs=hcat(Conv.xcorr(pa.om.g)...)
	elseif(pa.mode==:ibd)
		xgobs=pa.optm.obs.g
	end
	file=joinpath(folder, "xgobs.csv")
	CSV.write(file,DataFrame(hcat(xtgridg.x, xgobs)))

	file=joinpath(folder, "imxgobs.csv")
	CSV.write(file,DataFrame(hcat(repeat(xtgridg.x,outer=pa.nra),
				      repeat(1:pa.nra,inner=xtgridg.nx),vec(xgobs[:,1:pa.nra]))),)
	if(pa.mode ∈ [:bd, :bda])
		xgcal=hcat(Conv.xcorr(pa.optm.cal.g)...)
	elseif(pa.mode==:ibd)
		xgcal=pa.optm.calsave.g
	end
	file=joinpath(folder, "xgcal.csv")
	CSV.write(file,DataFrame(hcat(xtgridg.x,xgcal)))
	file=joinpath(folder, "imxgcal.csv")
	CSV.write(file,DataFrame(hcat(repeat(xtgridg.x,outer=pa.nra),
				      repeat(1:pa.nra,inner=xtgridg.nx),vec(xgcal[:,1:pa.nra]))),)

	# compute cross-correlations without blind Decon
	xg_nodecon=hcat(Conv.xcorr(pa.om.d, lags=[pa.ntg-1, pa.ntg-1])...)
	file=joinpath(folder, "xg_nodecon.csv")
	CSV.write(file,DataFrame(hcat(xtgridg.x,xg_nodecon)))

	file=joinpath(folder, "imxg_nodecon.csv")
	CSV.write(file,DataFrame(hcat(repeat(xtgridg.x,outer=pa.nra),
				      repeat(1:pa.nra,inner=xtgridg.nx),vec(xg_nodecon[:,1:pa.nra]))),)

	# compute cross-correlation of source selet
	xsobs=hcat(Conv.xcorr(pa.om.s, lags=[pa.ntg-1, pa.ntg-1])...)
	file=joinpath(folder, "xsobs.csv")
	CSV.write(file,DataFrame(hcat(xtgridg.x,xsobs)))

	# cross-plot of g
	file=joinpath(folder, "gcross.csv")
	CSV.write(file,DataFrame( hcat(vec(xgobs), vec(xgcal))))
	file=joinpath(folder, "gcross_nodecon.csv")
	CSV.write(file,DataFrame( hcat(vec(xgobs), vec(xg_nodecon))))


	# compute autocorrelations of source
	if(pa.mode ∈ [:bd, :bda])
		xsobs=autocor(pa.optm.obs.s[:,1], 1:pa.nt-1, demean=true)
		xscal=autocor(pa.optm.calsave.s[:,1], 1:pa.nt-1, demean=true)
	elseif(pa.mode==:ibd)
		xsobs=pa.optm.obs.s[:,1]
		xscal=pa.optm.calsave.s[:,1]
	end
	smat=hcat(vec(xsobs), vec(xscal))
	scale!(smat, inv(maximum(abs, smat)))
	
	# resample s, final sample should not exceed 1000
	ns=length(xsobs)
	fact=(ns>1000) ? round(Int,ns/1000) : 1
	xsobs=xsobs[1:fact:ns] # resample
	xscal=xscal[1:fact:ns] # resample

	# x plots of s
	file=joinpath(folder, "scross.csv")
	CSV.write(file,DataFrame(smat))

	# x plots of data
	dcal=pa.optm.calsave.d
	dobs=pa.optm.obs.d
	nd=length(dcal);
	fact=(nd>1000) ? round(Int,nd/1000) : 1
	dcal=dcal[1:fact:nd]
	dobs=dobs[1:fact:nd]
	datmat=hcat(vec(dcal), vec(dobs))
	scale!(datmat, inv(maximum(abs, datmat)))

	file=joinpath(folder, "datcross.csv")
	CSV.write(file,DataFrame( datmat))

	# finally save err
	err!(pa, cal=pa.optm.calsave) # compute err in calsave 
	file=joinpath(folder, "err.csv")
	CSV.write(file, pa.err)
end




