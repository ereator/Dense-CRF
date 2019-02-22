### This project is deprecated. The dense graphical models were re-implemented and optimized in the [Direct Graphical Models library](https://github.com/Project-10/DGM)

# DenseCRF + Direct Graphical Models

This is a modification of the [DenseCRF Code](http://graphics.stanford.edu/projects/densecrf/), which now supports OpenCV and DGM libraries.

This software pertains to the research described in the NIPS 2011 paper: _Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials_, by Philipp Krähenbühl and Vladlen Koltun. If you're using this code in a publication, please cite our paper.

This software is provided for research purposes, with absolutely no warranty or suggested support, and use of it most follow the BSD license agreement, at the top of each source file. *Please do not contact the authors for assistance with installing, understanding or running the code. However if you think you have found an interesting bug, the authors would be grateful if you could pass on the information.

## How to compile the code
Dependencies:
 - cmake  http://www.cmake.org/
 - OpenCV http://opencv.org/
 - DGM	  http://research.project-10.de/dgm/

## How to run the example
An example on how to use the DenseCRF can be found in examples/dense_inference.cpp. The example loads an image and some annotations.
It then uses a very simple classifier to compute a unary term based on those annotations. A dense CRF with both color dependent and color independent terms find the final accurate labeling.

Please note that this implementation is slightly slower than the one used to in our NIPS 2011 paper. Mainly because I tried to keep the code clean and easy to understand.

