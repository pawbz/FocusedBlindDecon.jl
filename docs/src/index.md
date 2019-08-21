# FocusedBlindDecon

Geophysicists rely on seismic data to understand the Earth’s subsurface.
Data from seismic receivers `d{T,2}(nt,nr)` contains two types of information convoluted
into a single signal: information about the source of the signal (source effects `s{T,1}(nts)`)
and information about the subsurface features it passed
through on its way to the receiver (path effects `g{T,2}(ntg,nr)`).  
Conventional methods for separating out the two types of
information rely on assumptions which may not be completely
accurate: extracting source effects requires assumptions about the path,
and extracting path effects requires assumptions about the source.

Similarly, in room
acoustics, the speech signal `s` recorded as `d` at a
microphone array is distorted as sound is reverberated 
due to `g` i.e., the reflection of walls, furniture and other objects.
Speech recognition and compression is simpler when
the reverberated records `d` at the microphones are
factorized into the distortions `g` and the clean speech signal `s`.


Focused Blind Deconvolution (FBD) performs the above-mentioned factorization and extracts
either the 
source or path information without relying on their assumptions, instead:
* it compares data `d` from the same source picked up by multiple receivers, and identifies similarities and differences among them;
* similarities among the signals can be identified as source effects `s`, while dissimilarities indicate path effects `g`.


`FocusedBlindDecon` is a Julia package corresponding to the article:
```
Bharadwaj, Pawan, Laurent Demanet, and Aimé Fournier. "Focused blind deconvolution." 
IEEE Transactions on Signal Processing 67.12 (2019): 3168-3180.
```
This package uses the fast Fourier transform `FFTW.jl` on the zero-padded
signals in order to perform multi-dimensional cross-correlations and convolutions. After installation,
the package has to be initialized either to utilize either of the two packages: `IterativeSolvers.jl` or `Optim.jl`,
for solving the linear systems. For example, execute one of the following commands.
```julia
using FocusedBlindDecon # start import package, also aliased as FBD
FBD.__init__(optg="optim", opts="optim") # uses Optim while solving for g and s
FBD.__init__(optg="iterativesolvers", opts="optim") # uses Optim while solving for s, and IterativeSolvers for g
```
By default, `FBD.__init__()` chooses the solvers from `IterativeSolvers.jl` for optimized performance. 

The functionality of this package revolves around the mutable `P_fbd` type. Firstly, most of the memory necessary to perform a given optimization is allocated while creating an instance of `P_fbd`, denoted as `pa`.
Then this instance is input to in-place functions (e.g., `lsbd!`, `fbd!`, `fibd!`)  
which as per Julia convention ends with an exclamation mark, to actually perform the optimizations.
Finally, the outputs of the optimizations can be easily accessed from `pa` e.g., `pa[:s]` returns the estimated source. 
Details of these methods are provided in the next section.
