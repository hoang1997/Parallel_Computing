#pragma once
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#ifndef __HOSTFUNCTIONS__
#define __HOSTFUNCTIONS__


#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif
#include <iostream>
#include <string>
#include <vector>

using namespace std;


class HostFunctions
{
private:
	size_t input_elements;
	size_t input_size;
	cl::Context context;
	cl::Program program;
	cl::CommandQueue queue;
	vector<int> data;
	size_t local_size;
	

public:
	HostFunctions( size_t inputElements, size_t inputSize, cl::Context cont, cl::CommandQueue q, cl::Program p);
	vector<int> localFunctions (vector<int> data, size_t local_size, int dataSize, string kernelName);
	vector<int> sortFunction(vector<int> data, size_t local_size, int dataSize);
	double varianceFunction(vector<int> data, size_t local_size, int dataSize, int mean);
	string GetFullProfilingInfo(const cl::Event& evnt, int resolution);
};

#endif