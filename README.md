### XMP 2.0 Alpha Release (June 2018)

The XMP 2.0 library provides a set of APIs for doing fixed size, unsigned multiple precision integer arithmetic in CUDA.   The library provides
these APIs under the name Cooperative Groups Big Numbers (CGBN).   The idea is that a cooperative group of threads will work together to represent
and process operations on each big numbers.   This alpha release targets the most important sizes:  1024 bits through 15360 bits in 1024 bit 
increments and operates with a warp (32 threads) per CGBN group.   In the future we will support multiple CGBN groups per instance warp and a
more granular set of sizes.  

### Why use CGBN?

CGBN imposes some constraints on the developer (discussed below), but within those constraints, it's **_really_** fast. 

In the following table, we compare the speed-up of CGBN on running a Tesla V100 (Volta) GPU vs. an Intel Xeon 16-Core E5-2997a running at 2.6 GHz 
with GMP 6.1.2 and OpenMP for parallelization:

|_operation_| 1024 bits | 2048 bits | 3072 bits | 4096 bits | 6144 bits | 8192 bits |_average speed-up_|
|-----------|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:----------------:|
|add        | 26.5      | 23.3      | 26.2      | 26.1      | 27.2      | 28.5      | 26.3             |
|sub        | 22.7      | 24.5      | 26.4      | 26.3      | 26.8      | 27.9      | 25.8             |
|mul        | 10.4      | 13.7      | 16.9      | 16.5      | 17.5      | 16.0      | 15.2             |
|mont_reduce| 12.7      | 15.0      | 18.9      | 22.2      | 26.0      | 26.5      | 20.2             |
|powm_odd   | 14.8      | 17.8      | 17.6      | 19.1      | 19.2      | 17.8      | 17.7             |
|div_qr     | 3.1       | 3.5       | 5.3       | 6.3       | 7.8       | 7.8       | 5.6              |
|sqrt       | 3.6       | 2.9       | 3.9       | 4.0       | 4.4       | 4.6       | 3.9              |
|gcd        | 3.2       | 4.4       | 4.9       | 4.8       | 4.3       | 3.9       | 4.3              |
|mod inv    | 2.5       | 3.2       | 3.6       | 3.3       | 3.1       | 3.0       | 3.1              |

These performance results were generated with the perf_tests tools provided with the library.

### Installation

To install this package, create a directory for the CGBN files, and untar the CGBN-<date>.tar.gz package.

CGBN relies on two open source packages which must be installed before running the CGBN makefile.   These are the GNU Multiple Precision Library (GMP)
and the Google Test framework (gtest).   If GMP is not installed as a local package on your system, you can built a local copy for your use as follows.

* Download GMP from http://www.gmplib.org
* Create a directory to hold the include files and library files
* Set the environment variable GMP_HOME to be your
* Configure GMP with `./configure --prefix=$GMP_HOME`
* Build and install GMP normally (we recommend that you also run make test).

If GMP is installed on your local system on the standard include and library paths, no action is needed.

CGBN also requires the Google Test framework source.  If this is installed on your system, set the environment variable GTEST_HOME to point to the source,
if it's not installed, we provide a `make download-gtest` in the main CGBN makefile that will download and unpack the Google Test framework into the CGBN
directory, where all the makefiles will find it automatically.

Once GMP and the Google Test framework are set up, the CGBN samples, unit tests, and performance tests can be built with `make _arch_` where _arch_ is one 
of kepler, maxwell, pascal, volta.   The compilation takes several minutes due to the large number of kernels that must built.

### Running Unit Tests

Once the unit tests have been compiled for the correct architecture, simply run the tester in the unit_tests directory.  This will run all tests on CUDA
device zero.  To use a different GPU, set the environment variable CUDA_VISIBLE_DEVICES.

### Running the Performance Tests

Once the performance tests have been compiled for the correct architecture, simply run the xmp_tester.  This will performance test a number of core CGBN 
APIs, print the information in a easily readily form and write a **_gpu\_throughput\_report.csv_** file.   To generate GMP performance results for the
same tests, run `make gmp-run` or `make gmp-numactl-run`.   The latter uses **numactl** to bind the GMP threads to a single CPU, for socket to socket
comparisons.   The make targets gmp-run and gmp-numactl-run both print the report in a readable format as well as generate a **_cpu\_throughput\_report.csv_**
file.   Speedups are easily computed by loading the two .csv files into a spreadsheet.

### Development - Getting Started



### Limitations
