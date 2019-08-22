# Load packages.
using Conv
using FocusedBlindDecon
using Gadfly

# We consider an illustrative synthetic experiment with the following parameters.
ntg=30 # number of time samples in `g`
nr=20 # number of receivers
nt=40*ntg # time samples in records `d`
nts=nt # samples in `s`
#+

# The aim is to reconstruct the true `g`, i.e., `gobs`, that we are going to design below. This design 
# is of particular interest e.g., in seismic inversion and room acoustics as they reveal 
# the arrival of energy, propagated from an impulsive source, at the receivers.
# Here, the arrivals curve linearly and hyperbolically, depending on `c` and have onsets 
# depending on `bfrac`. Their amplitudes are determined by `afrac`.
gobs=zeros(ntg,nr) # allocate
FBD.toy_direct_green!(gobs, c=4.0, bfrac=0.20, afrac=1.0); # add arrival 1
FBD.toy_reflec_green!(gobs, c=1.5, bfrac=0.35, afrac=-0.6); # add arrival 2
plotg=(x,args...)->spy(x, Guide.xlabel("channel"), Guide.ylabel("time"),args...) # define a plot recipe 
p1=plotg(gobs, Guide.title("True g"))

# The source signature `s` for the experiment is arbitrary: we simply use a Gaussian random signal.
sobs=randn(nts)
plot(y=sobs,x=1:nts,Geom.line, Guide.title("arbitrary source"), Guide.xlabel("time"))

# The next task is to generate synthetic observed records `dobs`: 
# first lets construct a linear operator `S`; then applying `S` on `g` will result in measurements `d`.
# This task can be skipped if measured `dobs` are already available. 
S=Conv.S(sobs, gsize=[ntg,nr], dsize=[nt,nr], slags=[nts-1,0]);
dobs=reshape(S*vec(gobs), nt, nr);
#+

# We first need to allocate a parameter variable `pa`, where the inputs `gobs` and `sobs` are optional.  
pa=P_fbd(ntg, nt, nr, nts, dobs=dobs, gobs=gobs, sobs=sobs)
#+

# The we perform LSBD i.e., least-squares fitting, without regularization, of `dobs` to jointly 
# optimize the arrays `g` and `s`. 
FBD.lsbd!(pa)
#+

# We extract `g` from `pa` and plot to notice that it doesn't match `gobs`.
p2=plotg(pa[:g], Guide.title("LSBD g"))

# Instead, we perform FBD that uses the focusing functionals to regularize `lsbd!`. 
FBD.fbd!(pa)
#+

# Notice that the extract impulse responses are closer to `gobs`, except for a scaling factor and an overall translation in time.
p3=plotg(pa[:g], Guide.title("FBD g"))


