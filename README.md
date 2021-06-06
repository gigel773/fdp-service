# fdp-service
This is a part of complex biometric system that is responsible for video-processing using several IP-cameras. The main purpose of this part is to prepare each frame from the video for classification.

### Build
Requirements:
 - Linux OS or Windows OS
 - CMake 
 - C++ compiler with C++17 support
 - TBB
 - OpenVINO
 - OpenCV (with extra modules)
 - cuDNN
 - CUDA
 - Boost
 - cpprestsdk
 - OpenSSL
 - google-benchmark
 - GTest
 
In order to build a project you may want to execute the further list of commands (within the root of the repo):
```shell
mkdir build
cd build
cmake ..
cmake --build .
```

### Performance tests
Project includes performance tests that are based on google-benchmark framework. These tests require testing photoset that is searched in path `../photos`. 

In order to run the tests you'll have to build target `nipb` and run executable file, e.g.:
```shell
./nipb --benchmark_repetitions=5
```
