
"""
Save Param
"""
function save(pa::BD, folder; tgridg=nothing, tgrid=nothing)
	!(isdir(folder)) && error("invalid directory")


	save(pa.om, folder, tgridg=tgridg, tgrid=tgrid)
	save(pa.optm, folder, tgridg=tgridg, tgrid=tgrid)
	savex(pa.optm, folder, tgridg=tgridg, tgrid=tgrid)

	# finally save err
	err!(pa) # compute err in cal 
	file=joinpath(folder, "err.csv")
	CSV.write(file, pa.err)
end


"""
Save Param
"""
function save(pa::IBD, folder; tgridg=nothing, tgrid=nothing)
	!(isdir(folder)) && error("invalid directory")

	save(pa.om, folder, tgridg=tgridg, tgrid=tgrid)
	save(pa.optm, folder, tgridg=tgridg, tgrid=tgrid)

	# finally save err
	err!(pa) # compute err in cal 
	file=joinpath(folder, "err.csv")
	CSV.write(file, pa.err)
end



function save(pa::ObsModel, folder; tgridg=nothing, tgrid=nothing)
	!(isdir(folder)) && error("invalid directory")


	(tgridg===nothing) && (tgridg = Grid.M1D(0.0, (pa.ntg-1)*1.0, pa.ntg))
	(tgrid===nothing) && (tgrid = Grid.M1D(0.0, (pa.nt-1)*1.0, pa.nt))

	# save original g
	file=joinpath(folder, "gobs.csv")
	CSV.write(file,DataFrame(hcat(tgridg.x, pa.g)))

	# save g for imagesc plots in TikZ
	file=joinpath(folder, "imgobs.csv")
	CSV.write(file,DataFrame(hcat(repeat(tgridg.x,outer=pa.nr),repeat(1:pa.nr,inner=pa.ntg),vec(pa.g))),)

	# save original s
	file=joinpath(folder, "sobs.csv")
	CSV.write(file,DataFrame(hcat(tgrid.x,pa.s)))

	# save data
	file=joinpath(folder, "dobs.csv")
	CSV.write(file,DataFrame(hcat(tgrid.x, pa.d)))
end

function save(pa::OptimModel, folder; tgridg=nothing, tgrid=nothing)
	!(isdir(folder)) && error("invalid directory")

	(tgridg===nothing) && (tgridg = Grid.M1D(0.0, (pa.ntg-1)*1.0, pa.ntg))
	(tgrid===nothing) && (tgrid = Grid.M1D(0.0, (pa.nt-1)*1.0, pa.nt))

	save_obscal(pa.obs.g, pa.cal.g, tgridg.x, "goptm", folder)
	save_obscal(pa.obs.d, pa.cal.d, tgrid.x, "doptm", folder)
	save_obscal(pa.obs.s, pa.cal.s, tgrid.x, "soptm", folder)

	save_cross(pa.obs.d,pa.cal.d, "doptm", folder)
	save_cross(pa.obs.g,pa.cal.g, "goptm", folder)
	save_cross(pa.obs.s,pa.cal.s, "soptm", folder)

end

	
function savex(pa::OptimModel, folder; tgridg=nothing, tgrid=nothing)
	!(isdir(folder)) && error("invalid directory")

	(tgridg===nothing) && (tgridg = Grid.M1D(0.0, (pa.ntg-1)*1.0, pa.ntg))
	(tgrid===nothing) && (tgrid = Grid.M1D(0.0, (pa.nt-1)*1.0, pa.nt))

	xtgridg=Grid.M1D_xcorr(tgridg); 
	xtgrid=Grid.M1D_xcorr(tgrid)

	xgobs=Conv.xcorr(pa.obs.g)[1]
	xgcal=Conv.xcorr(pa.cal.g)[1]

	save_obscal(xgobs, xgcal, xtgridg.x, "xg", folder)

	# compute cross-correlations without blind Decon
	xdobs=Conv.xcorr(pa.obs.d,Conv.P_xcorr(pa.nt, pa.nr, cglags=[pa.ntg-1, pa.ntg-1]))[1]
	xdcal=Conv.xcorr(pa.cal.d,Conv.P_xcorr(pa.nt, pa.nr, cglags=[pa.ntg-1, pa.ntg-1]))[1]
	xsobs=(Conv.xcorr(pa.obs.s,Conv.P_xcorr(pa.nt, 1, cglags=[pa.ntg-1, pa.ntg-1]))[1])
	xscal=(Conv.xcorr(pa.cal.s,Conv.P_xcorr(pa.nt, 1, cglags=[pa.ntg-1, pa.ntg-1]))[1])

	save_obscal(xdobs, xdcal, xtgridg.x, "xdat", folder)
	save_obscal(xsobs, xscal, xtgridg.x, "xs", folder)


	save_cross(xdobs,xdcal, "xdat", folder)
	save_cross(xsobs,xscal, "xs", folder)
	save_cross(xgobs,xgcal, "xg", folder)
end



function save_cross(a,b, name, folder; num=1000)
	(length(a) ≠ length(b)) && error("size mismatch")

	# x plots of data
	nd=length(b);
	fact=(nd>num) ? round(Int,nd/num) : 1
	b=b[1:fact:nd]
	a=a[1:fact:nd]
	datmat=hcat(vec(b), vec(a))
	scale!(datmat, inv(maximum(abs, datmat)))

	file=joinpath(folder, string(name, "cross.csv"))
	CSV.write(file,DataFrame( datmat))
end



function save_obscal(a,b, x, name, folder)
	(size(a) ≠ size(b)) && error("size mismatch")
	nr=size(a,2)
	nt=size(a,1)

	# save obs
	file=joinpath(folder, string(name,"obs.csv"))
	CSV.write(file,DataFrame(hcat(x, a)))
	# save cal
	file=joinpath(folder, string(name, "cal.csv"))
	CSV.write(file,DataFrame(hcat(x, b)))


	# save g for imagesc plots in TikZ
	file=joinpath(folder, string("im", name, "obs.csv"))
	CSV.write(file,DataFrame(hcat(repeat(x,outer=nr),repeat(1:nr,inner=nt),vec(a))))

	file=joinpath(folder, string("im", name, "cal.csv"))
	CSV.write(file,DataFrame(hcat(repeat(x,outer=nr),repeat(1:nr,inner=nt),vec(b))))

end

