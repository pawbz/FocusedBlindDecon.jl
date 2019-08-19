[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://pawbz.github.io/FocusedBlindDecon.jl/dev)
2
[![Build Status](https://travis-ci.org/pawbz/FocusedBlindDecon.jl.svg?branch=master)](https://travis-ci.org/pawbz/FocusedBlindDecon.jl)
3
[![codecov](https://codecov.io/gh/pawbz/FocusedBlindDecon.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/pawbz/FocusedBlindDecon.jl)

# Focused Blind Deconvolution (FBD)

## Attribution
If you make use of this code, please cite the following paper.
```latex
@article{bharadwaj2019focused,
  title={Focused blind deconvolution},
  author={Bharadwaj, Pawan and Demanet, Laurent and Fournier, Aim{\'e}},
  journal={IEEE Transactions on Signal Processing},
  volume={67},
  number={12},
  pages={3168--3180},
  year={2019},
  publisher={IEEE}
}
```

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
bdpa=FocusedBlindDecon.BD(ntg, nt, nr, gobs=gobs, sobs=sobs);
```

```julia
plotobsmodel(pa.om)
```

```julia
FocusedBlindDecon.bd!(pa)
```
