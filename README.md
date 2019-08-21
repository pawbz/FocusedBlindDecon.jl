[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://pawbz.github.io/FocusedBlindDecon.jl/dev)
[![Build Status](https://travis-ci.org/pawbz/FocusedBlindDecon.jl.svg?branch=master)](https://travis-ci.org/pawbz/FocusedBlindDecon.jl)
[![codecov](https://codecov.io/gh/pawbz/FocusedBlindDecon.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/pawbz/FocusedBlindDecon.jl)

# Focused Blind Deconvolution (FBD)

`FocusedBlindDecon` is a Julia package that implements the following methods.

**`lsbd!`** Least-squares blind deconvolution.

**`fibd!`** Focused interferometric blind deconvolution.

**`fpr!`** Focused phase retrieval.

**`fbd!`** Focused blind deconvolution.

These methods are described in the article below; if you make use of this code, please cite it.
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


## Installation
At the moment, `FocusedBlindDecon` depends on two unregistered packages `Misfits` and `Conv`.
For complete installation, enter these package manager commands in the REPL:
```julia
using Pkg
Pkg.add(PackageSpec(name="Misfits",url="https://github.com/pawbz/Misfits.jl.git"))
Pkg.add(PackageSpec(name="Misfits",url="https://github.com/pawbz/Conv.jl.git"))
Pkg.add(PackageSpec(name="GeoPhyInv",url="https://github.com/pawbz/FocusedBlindDecon.jl.git"))
```

## Documentation & Tutorials

A detailed documentation and some tutorials are available here: [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://pawbz.github.io/FocusedBlindDecon.jl/dev)
