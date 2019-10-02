This document provides a brief overview our INT4 submission for MLPerf inference 0.5.   Our submission is derived from earlier research at 
NVIDIA to assess the performance and accuracy of INT4 inference on Turing. 

The code uses a ResNet50-v1.5 pipeline, where the weights have been fine tuned to allow accurate inference using INT4 residual layers.  The 
code loads the fine tuned network from the “model” directory, which is used to drive the computation.  Internally, the code aggressively fuses 
layers to produce an efficient high performance inference engine which is used to process the Load Gen data.

The remainder of this document is organized as follows.   Section 1 describes how to run the inference engine and various command line arguments.
Section 2 discusses re-linking the tools with alternate LloadGgen routines.  Section 3 describes the model format and computations performed. 
Section 4 describes the process used to fine tune the model weights.

1.       RUNNING AND COMMAND LINE ARGUMENTS
Following files are relevant for INT4 harness: 
<TBD – The paths may get changed as per submission guidelines>
-          benchmarks/ResNet50-Int4/int4_offline.a
ResNet50 INT4 benchmark implementation provided as binary.
-          Harness/harness_offline/harness_offline_int4/inc/SampleLibrary.h
-          Harness/harness_offline/harness_offline_int4/src/SampleLibrary.cc
These files are made available to show how QSL is implemented by the harness.  The binary file int4_offline.a already comes pre-compiled with this implementation and as such these files are for reference only and don’t participate in re-compilation.
-          Harness/harness_offline/harness_offline_int4/Makefile
This file provides a recipe to produce int4_offline executable by linking int4_offline.a with mlperf_loadgen.so which is separately built and is located at (inference/loadgen/build/ lib.linux-x86_64-2.7). 
-          Harness/harness_offline/harness_offline_int4/model
This is described in detail later in Section 3.
-          Harness/harness_offline/harness_offline_int4/int4_offline
The INT4 harness executable.
Useful instructions for executing INT4 harness:
-          ./int4_offline -h/--help shows the available command line options
 
Useful options:
-b / --batch_size <n> : This is equivalent to perfSampleCount in loadgen terminology and defines how many images are processed per batch. (Max supported value: Tesla T4: 512, Titan RTX: 1024)
-p / --tensorPath <path>: Disk location <path> for sample images
-m / --mapPath <path>: Disk location <path> for val_map.txt file which contains filenames and labels.
-a <filename>: Load the config number for each conv layer from <f>.
--test-mode: Loadgen test mode.  {SubmissionRun, PerformanceOnly, AccuracyOnly}

Loadgen test/log settings related options:
Most of the loadgen supported test and log settings can be passed as command line arguments.  The prefix lgts is used for loadgen test setting parameters, while prefix lgls is used for loadgen log setting parameters.
Example command line:
-          int4_offline -b 512 -a autoconfig_bs256 --test-mode PerformanceOnly --tensorPath /path/to/sample/images --mapPath ../../../data_maps/imagenet/val_map.txt
 
2.       RE-LINKING WITH ALTERNATE LOADGEN TOOLS
 
There might be a desire to run the INT4 harness with a user-specified loadgen library. 

It is assumed that the user has pre-compiled mlperf_loadgen.so by following these steps (ref: https://github.com/mlperf/inference/blob/master/loadgen/README_BUILD.md):

git clone --recurse-submodules https://github.com/mlperf/inference.git mlperf_inference

LOADGEN_DIR=<Path> (e.g. ${PWD}/mlperf_inference/inference/loadgen)
CUDA_PATH=<Path for CUDA toolkit e.g. /usr/local/cuda-10.1>
cd $LOADGEN_ DIR
CFLAGS="-std=c++14 -O3" python setup.py bdist_wheel

To produce a new int4_offline executable, the following command can be used:
            	cd harness/harness_offline/harness_offline_int4
            	make -j CUDA=${CUDA_PATH} LOADGEN_PATH=${LOADGEN_DIR} clean
            	make -j CUDA=${CUDA_PATH} LOADGEN_PATH=${LOADGEN_DIR} all

3.       THE MODEL
The INT4 ResNet50 network consists of a pipeline of layers, described by files in the “model” directory.  At the top level, we have the following layers:

|Main Pipeline|
|---|
|First Layer Convolution (7x7, with C=3, K=64)|
|ReLU|
|Quantize 1|
|MaxPool|
|Layer 1 (sequence of 3 residual networks)|
|Layer 2 (sequence of 4 residual networks)|
|Layer 3 (Sequence of 6 residual networks)|
|Layer 4 (Sequence of 3 residual networks)|
|AvgPool|
|Fully Connected|
|Scale Layer|
 

Layer 1 through Layer 4 represent sequences of residual networks, where each residual network consists of the following:

|Residual Network|
|---|
|Resample/Downsample Pipeline|
|Conv1 (1x1 conv, typically with C=4*K)|
|ReLU layer|
|Quantize 1|
|Conv2 (3x3 conv, with C=K)|
|ReLU layer|
|Quantize 2|
|Conv3 (1x1 conv, with 4*C=K)|
|Eltwise Add|
|ReLU Layer|
|Quantize 3|
 
|The Resample/Downsample pipeline consists of three|
|---|
|Residual Resample/Downsample Pipeline|
|optional Convolution|
|optional ReLU layer|
|Quantize/Dequantize Layer|
 

For each layer, there is a text file that describes the layer structure and then within the “model/npdata” directory, there are files with actual filter weight, scale and bias values.   Filter weight files are just binary little-endian data files with a KCSR layout (K being the outer-most dimension, R being the inner-most).

For example, the “model/conv1” file contains the following text:

<pre>
Conv2d                          <i>layer type: convolution layer</i>
compute_mode=s8s8s32  	        <i>s8 activations, s8 weights, s32 accumulation/output</i>
in_channels=3                 	<i>3 input channel (r, g, b)</i>
out_channels=64               	<i>64 output channels</i>
kernel_size=2:7 7            	<i>kernel size tuple (2 dimension) 7x7</i>
stride=2:2 2                    <i>stride tuple (2 dimensions) 2, 2</i>
padding=2:3 3                 	<i>padding tuple (2 dimensions) 3, 3</i>
dilation=2:1 1                	<i>dilation tuple (2 dimensions) 1, 1</i>
groups=1                      	<i>groups parameter is 1</i>
bias=s16,1:64                 	<i>bias vector, s16 datatype, 1 dimension, size 64</i>
!model/npdata/conv1.bias     	<i>location of the bias data</i>
weight=s8,4:64 3 7 7         	<i>weight tensor, s8 data, 4 dims, KCRS</i>
!model/npdata/conv1.weight   	<i>location of the weight data</i>
</pre>

These files are largely self-explanatory and have the same semantics as the standard ResNet layers.  The one exception is a quantize / dequantize layer.   Here’s an example quantize, “model/quantize1”:

Quantize                             			Layer type: quantize layer
compute_mode=u31u16u4        			u31 inputs, u16 weights, u4 outputs
output_bits=4                        			output is 4 bits wide
shift_bits=16                        			described below
max_requant=71                      			maximum value in the u16 data
requant_factor=u16,4:1 64 1 1   			scale tensor, u16 data, 64 input/output chans
!model/npdata/quantize1.requant_factor 		Location of scale data

For each input, a quantize layer does fixed point arithmetic and computes:

half = 2^shift_bits / 2
out[channel] = (±in[channel] * requant_factor[channel] ± half) / 2^shift_bits
clamp out[channel] to appropriate range, in this example u4 (0..15)

Quantization layers can also be used to de-quantize, for example, “model/layer1_0_downsample_2”, which has a compute_mode of “s8u16s32”, 31 output bits and a shift_bits of 0.   The quantization layer rounds positive value ties towards +inf and negative value ties towards -inf.


4.       FINE TUNING THE MODEL WEIGHTS
Post training quantizing to int4 can’t preserve enough accuracy, so the network needs to be fine tuned. Fake quantization is added into forward propagation. Since quantization is either not differentiable or has derivative 0 depends on the input value, we use Straight Through Estimator (STE) to approximate its derivative during backward propagation: .
Quantization of each tensor is defined by range. We determine the range by 99.999% percentile calibration for both activation and weight. Values outside the range are clamped. A hHistogram of weights is collected on the pretrained model weights. For activation, we fed 512 images in training set to collect the histogram. After determininged the range offline, we fine tune the quantized network for 15 epochs. The pretrained model is from torchvision.
 

