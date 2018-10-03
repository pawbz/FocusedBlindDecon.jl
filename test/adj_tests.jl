pa=DC.simple_problem()

function adjtest()
	x=randn(size(F,2))
	y=randn(size(F,1))
	a=LinearAlgebra.dot(y,F*x)
	b=LinearAlgebra.dot(x,adjoint(F)*y)
	c=LinearAlgebra.dot(x, transpose(F)*F*x)
	println("adjoint test: ", a, "\t", b)       
	@test isapprox(a,b,rtol=1e-6)
	println("must be positive: ", c)
	@test c>0.0
end


@testset "IBD" begin
	global p=pa.pfibd
	for attrib in [DC.S(), DC.G()]
		global F=DC.operator(p, attrib);
		adjtest()
	end
end

@testset "BD" begin
	global p=pa.plsbd
	for attrib in [DC.S(), DC.G()]
		global F=DC.operator(p, attrib);
		adjtest()
	end
end


