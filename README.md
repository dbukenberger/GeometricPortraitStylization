<img src='https://bukenberger.net/papers/hendrix.svg' align="right" width="220" height="220">

# Geometric Portrait Stylization
[![PDF](https://img.shields.io/badge/PDF-green)](https://bukenberger.net/pdfs/2024_bukenberger_geometric_portrait_stylization.pdf)
[![VMV Paper](https://img.shields.io/badge/DOI-10.2312%2Fvmv%20241203-blue)](https://doi.org/10.2312/vmv.20241203)

With this implementation you can generate geometric stylizations of portraits and scenic input images as described in the paper.

## Dependencies

<img src='https://bukenberger.net/papers/kahlo.svg' align="right" width="220" height="220">

Used libraries can be easily installed with `pip`.

**Required**
* [NumPy](https://github.com/numpy/numpy) for vectorized arrays.
* [OpenCV](https://github.com/opencv/opencv-python) for image processing.
* [dlib](https://github.com/davisking/dlib) for face landmark detection.

The `dlib` face detector also requires a pretrained model for the landmark detection.
If you have the [requests](https://github.com/psf/requests) lib installed, it will download the model automatically.
Otherwise, you can lookup the url and target path in the `util.py` file and download the model manually.

Further common utility methods are bundled in my [drbutil](https://github.com/dbukenberger/drbutil) which you can also install as a library using `pip install drbutil`.
However, if not installed, it will be downloaded automatically and imported from source.

## Run Examples
* In the main directory you can run `python runExamples.py` to generate example results.
This script contains exemplary setups to recreate results from the paper and an explanation of the used parameters.
Here you can also set the result directory and if you want to show previews during the computation.
* Results are stored as `.png` and `.svg` files, respectively.

## Citation
You can cite the paper with:
```
@inproceedings{bukenberger2024geometric,
	booktitle = {Vision, Modeling, and Visualization},
	editor = {Linsen, Lars and Thies, Justus},
	title = {{Geometric Portrait Stylization}},
	author = {Bukenberger, Dennis R.},
	year = {2024},
	publisher = {The Eurographics Association},
	ISBN = {978-3-03868-247-9},
	DOI = {10.2312/vmv.20241203}
}
```
