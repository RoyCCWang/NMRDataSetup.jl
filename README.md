# NMRDataSetup.jl
This package is designed to work given the 1D 1H NMR time series free-induction decay (FID) data from Bruker spectrometers. The main function is `setupBruker1Dspectrum`, which requires the user to specify the folder path where the binary *fid* file and text file *acqu* (alternatively, *acqus*) are.

The following parameters must be present in the *acqu* (or *acqus*) file:
```
$TD # total number of data samples acquired.
$SW # spectral window, in ppm.
$SFO1 # carrier freuqnecy. Used as spectrometer frequency.
SO1 # offset frequency. Used to estimate roughly where the 0 ppm is.
$SW_h # sampling frequency.

$BYTORDA # control variable on endianess.
$DTYPA # control variable on data type of the FID binary file.
```

Please refer to a Bruker manual for the meaning of these parameters.

# Usage
Please see `/examples/load_experiment.jl` for an example.

# Install
From a Julia REPL or script, add the custom registry before adding the package:

```
using Pkg

Pkg.Registry.add(url = "https://github.com/RoyCCWang/RWPublicJuliaRegistry")

Pkt.add("NMRDataSetup)
```

# Citation
Our work is undergoing peer review.

# License
This project is licensed under the Mozilla Public License v2.0; see the LICENSE file for details. Individual source files may contain the following tag instead of the full license text:
```
SPDX-License-Identifier: MPL-2.0
```

Using SPDX enables machine processing of license information based on the SPDX License Identifiers and makes it easier for developers to see at a glance which license they are dealing with.
