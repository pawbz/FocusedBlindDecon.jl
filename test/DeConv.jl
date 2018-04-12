using Revise
using Signals
using Grid
using Inversion
using DeConv
using Base.Test
using ForwardDiff
using BenchmarkTools


# a simple DeConv memory test
ntgf = 200
nr = 30
tfact=20;
gfobs=zeros(ntgf, nr)
Signals.DSP.toy_green!(gfobs,bfrac=0.2, nevents=3,   afrac=[2.0, 0.5, 0.25]);
nt = ntgf*tfact
δt=0.0001; # results of independent of this
tgridnoise=Grid.M1D(0., Float64(ntgf*tfact), ntgf*tfact)
gfprecon, gfweights, wavprecon=DeConv.create_white_weights(ntgf, nt, nr)
wavobs=randn(nt);
pa=DeConv.Param(ntgf, nt, nr, gfobs=gfobs,fft_threads=true, gfprecon=gfprecon,     wavnorm_flag=false, wavobs=wavobs, verbose=false, gfweights=gfweights, mode=:ibd);
DeConv.update_func_grad!(pa,gfoptim=[:ls], gfαvec=[1.]);
DeConv.initialize!(pa)
#@time DeConv.update_all!(pa, max_reroundtrips=1, max_roundtrips=1, roundtrip_tol=1e-6)
DeConv.remove_gfprecon!(pa, all=true)
DeConv.remove_gfweights!(pa, all=true)
#@time DeConv.update_all!(pa, max_reroundtrips=1, max_roundtrips=1, roundtrip_tol=1e-6)



##


ntgf = 3
nr = 2
nt = 5
gfobs=randn(ntgf, nr)
wavobs=randn(nt);
gfprecon1, gfweights1, wavprecon1=DeConv.create_weights(ntgf,
		nt, gfobs, αexp=100., cflag=true)
randn!(gfprecon1)
randn!(wavprecon1)

pa1=DeConv.Param(ntgf, nt, nr, gfobs=gfobs,
	wavobs=wavobs, wavnorm_flag=false, verbose=false,
	gfprecon=gfprecon1, gfweights=gfweights1, wavprecon=wavprecon1, mode=:bd
	)


gfprecon2, gfweights2, wavprecon2=DeConv.create_white_weights(ntgf, nt, nr)
randn!(gfprecon2)

pa2=DeConv.Param(ntgf, nt, nr, gfobs=gfobs,     wavobs=wavobs,
		  wavnorm_flag=false, verbose=false,      mode=:ibd,
		  )
DeConv.add_gfprecon!(pa2, gfprecon2)
DeConv.add_gfweights!(pa2, gfweights2)
DeConv.add_wavprecon!(pa2, wavprecon2)


for pa in [pa1, pa2]
	println("... tests of, ", string(pa.mode))
	@time for attrib in [:gf, :wav]
		println("... tests of, ", string(attrib))
		pa.attrib_inv=attrib
		x=randn(DeConv.ninv(pa))
		if(pa.mode==:ibd && pa.attrib_inv==:wav)
			x[1]=1.0
		end
		xa=similar(x)
		DeConv.x_to_model!(x, pa)
		DeConv.model_to_x!(xa, pa)
		# note that data is not linear when norm flag
		!pa.wavnorm_flag && @test x ≈ xa

		# compute Fx
		last_x=similar(x)
		DeConv.F!(pa, x)
		dcal=pa.cal.d

		# compute F^a dcala
		dcala=randn(size(pa.cal.d))
		gx=similar(x)
		DeConv.Fadj!(pa, x, gx, dcala)

		# note that data is not linear when norm_flag
		!pa.wavnorm_flag && @test dot(x, gx) ≈ dot(dcal, dcala)

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
ntgf = 50
nr = 40
nt = 5000

gfobs=randn(ntgf, nr)
wavobs=randn(nt)

pa=DeConv.Param(ntgf, nt, nr, gfobs=gfobs, wavobs=wavobs, wavnorm_flag=false,verbose=false)
storagewav=randn(size(pa.xwav))
storagegf=randn(size(pa.xgf))
# memory tests
using BenchmarkTools
println("===========")
pa.attrib_inv=:wav
@btime (randn!(pa.last_xwav); DeConv.F!(pa, pa.xwav,  ));
@btime DeConv.Fadj!(pa, pa.xwav, storagewav, pa.ddcal)
@btime DeConv.x_to_model!(pa.xwav, pa);
@btime DeConv.model_to_x!(pa.xwav, pa);
@btime DeConv.update_wav!(pa, pa.xwav,  pa.dfwav)
@btime DeConv.func_grad!(storagewav, pa.xwav,pa)
pba.attrib_inv=:gf
@btime DeConv.F!(pa, pa.xgf,);
@btime DeConv.Fadj!(pa, pa.xgf,storagegf, pa.ddcal)
@btime DeConv.x_to_model!(pa.xgf, pa);
@btime DeConv.model_to_x!(pa.xgf, pa);
@btime DeConv.update_gf!(pa, pa.xgf,  pa.dfgf)
@btime DeConv.func_grad!(storagegf, pa.xgf,  pa)


# check if there are any model discrepancies

begin
	gfobs=randn(ntgf, nr)
	wavobs=randn(nt)
	gfobs[1,:]=0.0; # mute
	gfprecon, gfweights, wavprecon=DeConv.create_weights(ntgf, nt, gfobs,
		αexp=10., cflag=true)
	paDeConv=DeConv.Param(ntgf, nt, nr, gfobs=gfobs, gfweights=gfweights,
	    fft_threads=false, wavnorm_flag=false, wavobs=wavobs, verbose=false, gfprecon=gfprecon,
	    wavprecon=wavprecon);
end

begin
	storagewav=randn(size(paDeConv.xwav))
	storagegf=randn(size(paDeConv.xgf))
	copy!(paDeConv.cal.wav, wavobs)
	copy!(paDeConv.cal.gf, gfobs)
end

begin
	paDeConv.attrib_inv=:wav
	DeConv.model_to_x!(paDeConv.xwav, paDeConv);
	@time DeConv.F!(paDeConv, paDeConv.xwav,  );
	@test paDeConv.cal.d ≈ paDeConv.obs.d
end

begin
	@time DeConv.func_grad!(storagewav, paDeConv.xwav, paDeConv)
	@test (storagewav ≈ zeros(storagewav))
end



begin
	paDeConv.attrib_inv=:gf
	DeConv.model_to_x!(paDeConv.xgf, paDeConv);
	@time DeConv.F!(paDeConv, paDeConv.xgf,  );
	@test paDeConv.cal.d ≈ paDeConv.obs.d
end
begin
	@time DeConv.func_grad!(storagegf, paDeConv.xgf, paDeConv)
	@test (storagegf ≈ zeros(storagegf))
end


# final test
@time DeConv.update_all!(pa)
f, α = JuMIT.Misfits.error_after_scaling(pa.cal.wav, wavobs)
@test (f<1e-3)
f, α = JuMIT.Misfits.error_after_scaling(pa.cal.gf, gfobs)
@test (f<1e-3)
