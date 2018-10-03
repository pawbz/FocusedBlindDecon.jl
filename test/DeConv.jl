using Revise
using Signals
using Inversion
using DeConv
using Base.Test
using ForwardDiff
using BenchmarkTools


# a simple DeConv memory test
ntg = 200
nr = 30
tfact=20;
gobs=zeros(ntg, nr)
Signals.DSP.toy_green!(gobs,bfrac=0.2, nevents=3,   afrac=[2.0, 0.5, 0.25]);
nt = ntg*tfact
δt=0.0001; # results of independent of this
tgridnoise=range(0., stop=Float64(ntg*tfact), length=ntg*tfact)
gprecon, gweights, sprecon=DeConv.create_white_weights(ntg, nt, nr)
sobs=randn(nt);
pa=DeConv.Param(ntg, nt, nr, gobs=gobs,fft_threads=true, gprecon=gprecon,     snorm_flag=false, sobs=sobs, verbose=false, gweights=gweights, mode=:ibd, fftwflag=FFTW.MEASURE);
DeConv.update_func_grad!(pa,goptim=[:ls], gαvec=[1.]);
DeConv.initialize!(pa)
DeConv.update_all!(pa, max_reroundtrips=1, max_roundtrips=1, roundtrip_tol=1e-6)
#DeConv.remove_gprecon!(pa, all=true)
#DeConv.remove_gweights!(pa, all=true)
#DeConv.update_all!(pa, max_reroundtrips=1, max_roundtrips=1, roundtrip_tol=1e-6)



rrrr
##


ntg = 3
nr = 2
nt = 5
gobs=randn(ntg, nr)
sobs=randn(nt);
gprecon1, gweights1, sprecon1=DeConv.create_weights(ntg,
		nt, gobs, αexp=100., cflag=true)
randn!(gprecon1)
randn!(sprecon1)

pa1=DeConv.Param(ntg, nt, nr, gobs=gobs,
	sobs=sobs, snorm_flag=false, verbose=false,
	gprecon=gprecon1, gweights=gweights1, sprecon=sprecon1, mode=:bd
	)


gprecon2, gweights2, sprecon2=DeConv.create_white_weights(ntg, nt, nr)
randn!(gprecon2)

pa2=DeConv.Param(ntg, nt, nr, gobs=gobs,     sobs=sobs,
		  snorm_flag=false, verbose=false,      mode=:ibd,
		  )
DeConv.add_gprecon!(pa2, gprecon2)
DeConv.add_gweights!(pa2, gweights2)
DeConv.add_sprecon!(pa2, sprecon2)


for pa in [pa1, pa2]
	println("... tests of, ", string(pa.mode))
	@time for attrib in [:g, :s]
		println("... tests of, ", string(attrib))
		pa.attrib_inv=attrib
		x=randn(DeConv.ninv(pa))
		if(pa.mode==:ibd && pa.attrib_inv==:s)
			x[1]=1.0
		end
		xa=similar(x)
		DeConv.x_to_model!(x, pa)
		DeConv.model_to_x!(xa, pa)
		# note that data is not linear when norm flag
		!pa.snorm_flag && @test x ≈ xa

		# compute Fx
		last_x=similar(x)
		DeConv.F!(pa, x)
		dcal=pa.cal.d

		# compute F^a dcala
		dcala=randn(size(pa.cal.d))
		gx=similar(x)
		DeConv.Fadj!(pa, x, gx, dcala)

		# note that data is not linear when norm_flag
		!pa.snorm_flag && @test dot(x, gx) ≈ dot(dcal, dcala)

		# compute gradient
		g1=similar(x);
		DeConv.func_grad!(g1, x,pa)
		g2=similar(x);
		@time Inversion.finite_difference!(x -> DeConv.func_grad!(nothing, x, pa), x, g2, :central)
		println(g1)
		println(g2)
		println(g2-g1)
		@test g1 ≈ g2
	end
end

rrrrrrr

# a simple DeConv test
ntg = 50
nr = 40
nt = 5000

gobs=randn(ntg, nr)
sobs=randn(nt)

pa=DeConv.Param(ntg, nt, nr, gobs=gobs, sobs=sobs, snorm_flag=false,verbose=false)
storages=randn(size(pa.xs))
storageg=randn(size(pa.xg))
# memory tests
using BenchmarkTools
println("===========")
pa.attrib_inv=:s
@btime (randn!(pa.last_xs); DeConv.F!(pa, pa.xs,  ));
@btime DeConv.Fadj!(pa, pa.xs, storages, pa.ddcal)
@btime DeConv.x_to_model!(pa.xs, pa);
@btime DeConv.model_to_x!(pa.xs, pa);
@btime DeConv.update_s!(pa, pa.xs,  pa.dfs)
@btime DeConv.func_grad!(storages, pa.xs,pa)
pba.attrib_inv=:g
@btime DeConv.F!(pa, pa.xg,);
@btime DeConv.Fadj!(pa, pa.xg,storageg, pa.ddcal)
@btime DeConv.x_to_model!(pa.xg, pa);
@btime DeConv.model_to_x!(pa.xg, pa);
@btime DeConv.update_g!(pa, pa.xg,  pa.dfg)
@btime DeConv.func_grad!(storageg, pa.xg,  pa)


# check if there are any model discrepancies

begin
	gobs=randn(ntg, nr)
	sobs=randn(nt)
	gobs[1,:]=0.0; # mute
	gprecon, gweights, sprecon=DeConv.create_weights(ntg, nt, gobs,
		αexp=10., cflag=true)
	paDeConv=DeConv.Param(ntg, nt, nr, gobs=gobs, gweights=gweights,
	    fft_threads=false, snorm_flag=false, sobs=sobs, verbose=false, gprecon=gprecon,
	    sprecon=sprecon);
end

begin
	storages=randn(size(paDeConv.xs))
	storageg=randn(size(paDeConv.xg))
	copy!(paDeConv.cal.s, sobs)
	copy!(paDeConv.cal.g, gobs)
end

begin
	paDeConv.attrib_inv=:s
	DeConv.model_to_x!(paDeConv.xs, paDeConv);
	@time DeConv.F!(paDeConv, paDeConv.xs,  );
	@test paDeConv.cal.d ≈ paDeConv.obs.d
end

begin
	@time DeConv.func_grad!(storages, paDeConv.xs, paDeConv)
	@test (storages ≈ zeros(storages))
end



begin
	paDeConv.attrib_inv=:g
	DeConv.model_to_x!(paDeConv.xg, paDeConv);
	@time DeConv.F!(paDeConv, paDeConv.xg,  );
	@test paDeConv.cal.d ≈ paDeConv.obs.d
end
begin
	@time DeConv.func_grad!(storageg, paDeConv.xg, paDeConv)
	@test (storageg ≈ zeros(storageg))
end


# final test
@time DeConv.update_all!(pa)
f, α = JuMIT.Misfits.error_after_scaling(pa.cal.s, sobs)
@test (f<1e-3)
f, α = JuMIT.Misfits.error_after_scaling(pa.cal.g, gobs)
@test (f<1e-3)
