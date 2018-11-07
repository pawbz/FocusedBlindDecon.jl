# Focused Blind Deconvolution (FBD)

A multi-channel blind deconvolution (BD) example is here. Choose the length of the Green's functions, source signal and data vectors:
```julia
ntg=20 # length of Green's functions
nr=40 # number of receivers
tfact=80 # 
nt = ntg*tfact # length of data
```
Then, we create some toy Green's functions:
```julia
gobs=randn(ntg, nr)
sobs=randn(nt)
```
Allocation of memory necessary to perform BD
```julia
bdpa=DeConv.BD(ntg, nt, nr, gobs=gobs, sobs=sobs);
```

```julia
plotobsmodel(pa.om)
```

```julia
DeConv.bd!(pa)
```
