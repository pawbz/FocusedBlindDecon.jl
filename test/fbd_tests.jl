function convergence_test(tol=1e-3)
	global pa
	FBD.fbd!(pa);
	@test Misfits.error_after_scaling(pa.pfibd.optm.cal.g,pa.pfibd.optm.obs.g)[1] < tol
	@test Misfits.error_after_scaling(pa.pfibd.optm.cal.s[1:pa.plsbd.om.nts-1],pa.pfibd.optm.obs.s[1:pa.plsbd.om.nts-1])[1] < tol
	@test Misfits.error_after_scaling(pa.plsbd.optm.cal.g,pa.plsbd.optm.obs.g)[1] < tol
	@test Misfits.error_after_scaling(pa.plsbd.optm.cal.s,pa.plsbd.optm.obs.s)[1] < tol

end


ENV["FOCUSBD_OPTG"]="optim"
ENV["FOCUSBD_OPTS"]="optim"
FBD.__init__()
pa=FBD.random_problem();
convergence_test(1e-3)


ENV["FOCUSBD_OPTG"]="iterativesolvers"
ENV["FOCUSBD_OPTS"]="iterativesolvers"
FBD.__init__()
pa=FBD.random_problem();
convergence_test(1e-3)

ENV["FOCUSBD_OPTG"]="iterativesolvers"
ENV["FOCUSBD_OPTS"]="iterativesolvers"
FBD.__init__(stf=true)
pa=FBD.random_problem();
convergence_test(1e-3)


