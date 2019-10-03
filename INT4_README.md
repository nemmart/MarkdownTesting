This document provides a brief overview our INT4 submission for MLPerf inference 0.5.   Our submission is derived from earlier research at 
NVIDIA to assess the performance and accuracy of INT4 inference on Turing. 

The code uses a ResNet50-v1.5 pipeline, where the weights have been fine tuned to allow accurate inference using INT4 residual layers.  The 
code loads the fine tuned and quantized network from the “model” directory, which is used to drive the computation.  It's worth noting we use
a full ResNet50-v1.5 pipeline, no layers have been pruned or altered, other than using low precision activations and weights.  Internally, the
code does fuse layers to save bandwidth and produce an efficient high performance inference engine, which is used to process the LoadGen data.

The remainder of this document is organized as follows.   Section 1 describes how to run the inference engine and various command line arguments.
Section 2 discusses re-linking the tools with alternate LoadGen routines.  Section 3 describes the model format and computations performed. 
Section 4 describes the process used to fine tune the model weights.

##1. FILES LOCATIONS AND RUNNING THE SOFTWARE

Most of the files are in the folder "Harness/harness_offline/harness/offline_int4".  Here you'll find:

|File|Description|
|---|---|
|int4_offline|The INT4 harness executable|
|model|The INT4 model directory, described in the next section|
|Makefile|This file provides a recipe to produce int4_offline executable by linking int4_offline.a with mlperf_loadgen.so 
          which is separately built and is located at (inference/loadgen/build/ lib.linux-x86_64-2.7)|



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
 
##2. MODEL DESCRIPTION

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
Conv2d                                    <i>layer type: convolution layer</i>
compute_mode=s8s8s32  	                  <i>s8 activations, s8 weights, s32 accumulation/output</i>
in_channels=3                 	          <i>3 input channel (r, g, b)</i>
out_channels=64               	          <i>64 output channels</i>
kernel_size=2:7 7            	          <i>kernel size tuple (2 dimension) 7x7</i>
stride=2:2 2                              <i>stride tuple (2 dimensions) 2, 2</i>
padding=2:3 3                 	          <i>padding tuple (2 dimensions) 3, 3</i>
dilation=2:1 1                	          <i>dilation tuple (2 dimensions) 1, 1</i>
groups=1                      	          <i>groups parameter is 1</i>
bias=s16,1:64                 	          <i>bias vector, s16 datatype, 1 dimension, size 64</i>
!model/npdata/conv1.bias     	          <i>location of the bias data</i>
weight=s8,4:64 3 7 7         	          <i>weight tensor, s8 data, 4 dims, KCRS</i>
!model/npdata/conv1.weight   	          <i>location of the weight data</i>
</pre>

These files are largely self-explanatory and have the same semantics as the standard ResNet layers.  The one exception is a quantize / dequantize layer.   Here’s an example quantize, “model/quantize1”:

<pre>
Quantize                                  <i>Layer type: quantize layer</i>
compute_mode=u31u16u4                     <i>u31 inputs, u16 weights, u4 outputs</i>
output_bits=4                             <i>output is 4 bits wide</i>
shift_bits=16                             <i>described below</i>
max_requant=71                            <i>maximum value in the u16 data</i>
requant_factor=u16,4:1 64 1 1   	  <i>scale tensor, u16 data, 64 input/output chans</i>
!model/npdata/quantize1.requant_factor    <i>Location of scale data</i>
</pre>

For each input, a quantize layer does fixed point arithmetic and computes:

```
      half = 2^shift_bits / 2
      out[channel] = (±in[channel] * requant_factor[channel] ± half) / 2^shift_bits
      clamp out[channel] to appropriate range, in this example u4 (0..15)
```

Quantization layers can also be used to de-quantize, for example, “model/layer1_0_downsample_2”, which has a compute_mode of “s8u16s32”, 31 output bits and a shift_bits of 0.   The quantization layer rounds positive value ties towards +inf and negative value ties towards -inf.

##3. HOW THE MODEL WAS FINE TUNED

For INT8 inference, the standard technique is to take the trained network, run a calibration dataset and use the results to
set the quantization parameters for the layers.  Unfortunately, using this technique for INT4 doesn't work well.  Too much
information is lost during the quantization and the network accuracy suffers badly.   To resolve this problem, we start with a
pre-trained model (the standard torchvision model).  We run a calibration dataset and collect range and histogram data for both
activations and model weights.   Next, we add "fake" FP32 quantization layers to model.  The initial parameters for these 
quantization layers are set using the collected range data.  Note, it's still an all FP32 model.  Then we run training epochs 
to fine tune the network and quantization layers. The process works as follows:

1)  Run the calibration dataset through the "quantized" FP32 model and collect histogram data
2)  Use KL divergence to choose new quantization ranges that minimize information loss between the histograms
3)  Adjust the quantization layers in the model with the new ranges
4)  Run an epoch of the training dataset through the quantized model and back propagate the errors, using a
    Straight Through Estimator (STE) for the quantization layers

Continue training until the accuracy reaches acceptable levels, typically about 15 epochs.  Once complete, the model is fine
tuned and a quantized INT4 model can be generated using the range data from the "fake" quantization layers.  For more 
information about the fine tuning process, please see: ....

##4. RE-LINKING WITH ALTERNATE LOADGEN TOOLS
 
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

