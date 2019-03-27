#include "HostFunctions.h"
#include <string>
#include <iostream>
#include <sstream>

HostFunctions::HostFunctions( size_t inputElements, size_t inputSize, cl::Context cont, cl::CommandQueue q, cl::Program p)
{
	//initialise all the arguments to the hostfunctions instance;
	input_elements = inputElements;
	input_size = inputSize;
	context = cont;
	queue = q;
	program = p;
	
}

//LOCAL KERNEL//
vector<int> HostFunctions::localFunctions(vector<int> data, size_t local_size, int dataSize, string kernelName) {
	//create output vector with size of data + padding which was set when reading the file
	vector<int> output(data.size());
	
	//set output_size variable in bytes
	size_t output_size = data.size() * sizeof(int);

	
	//Create Buffers and events;
	cl::Buffer input_buffer(context, CL_MEM_READ_ONLY, input_size);
	cl::Buffer output_buffer(context, CL_MEM_READ_WRITE, output_size);
	cl::Event write_event;
	cl::Event read_event;
	cl::Event prof_event;

	//DEVICE OPERATIONS
	//copy data to the input buffer and initialise other arrays on device memory
	queue.enqueueWriteBuffer(input_buffer , CL_TRUE, 0, input_size, &data[0], NULL, &write_event);
	queue.enqueueFillBuffer(output_buffer, 0, 0, output_size);//zero B buffer on device memory
	
	//Create kernel and set arguments
	cl::Kernel localKernel(program, kernelName.c_str());
	localKernel.setArg(0, input_buffer);
	localKernel.setArg(1, output_buffer);
	localKernel.setArg(2, cl::Local(local_size * sizeof(int)));//local memory size

	//call all kernels in a sequence
	queue.enqueueNDRangeKernel(localKernel, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event);
	//Copy the result from device to host
	queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, output_size, &output[0], NULL, &read_event);
	
	//Print out all the profiling information, read and write time to buffers, and kernel execution time
	cout << "PROFILING INFORMATION:\nKernel Name: " << kernelName << endl;
	cout << "Read time [ns]: " << read_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
		read_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
	cout << "Write time [ns]: " << write_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
		write_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
	cout << "Kernel execution time [ns]: " <<
		prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
		prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() <<"\n"<< endl;
	
	//return output vector to menu system
	return output;
}

//SORT KERNEL//
vector<int>  HostFunctions::sortFunction(vector<int> data, size_t local_size, int dataSize) {

	vector<int> output(data.size());
	size_t output_size = data.size() * sizeof(int);//size in bytes
	
	//Create one input buffer 
	cl::Buffer input_buffer(context, CL_MEM_READ_ONLY, input_size);
	cl::Event write_event;
	cl::Event read_event;
	cl::Event prof_event;

	//device operations

	//copy array A to and initialise other arrays on device memory
	queue.enqueueWriteBuffer(input_buffer, CL_TRUE, 0, input_size, &data[0], NULL, &write_event);

	//Setup and execute all kernels (i.e. device code)
	cl::Kernel kernel_1 = cl::Kernel(program, "sort_odd");
	kernel_1.setArg(0, input_buffer);
	cl::Kernel kernel_2 = cl::Kernel(program, "sort_even");
	kernel_2.setArg(0, input_buffer);
	
	//Loop through the execution of both odd and even kernels to sort
	for (int i = 0; i < data.size() / 2; i++)
	{
		queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event);
		queue.enqueueNDRangeKernel(kernel_2, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event);
		
	}
	//Read input buffer to output vector
	queue.enqueueReadBuffer(input_buffer, CL_TRUE, 0, output_size, &output[0], NULL, &read_event);

	//Print out all the profiling information, read and write time to buffers, and kernel execution time
	cout << "PROFILING INFORMATION:\nKernel Name: variance_local" << endl;
	cout << "Read time [ns]: " << read_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
		read_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
	cout << "Write time [ns]: " << write_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
		write_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
	cout << "Kernel execution time [ns]: " <<
		prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
		prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << "\n" << endl;
	
	//return output vector to menu system
	return output;
}

//SUM OF SQUARED DIFFERENCE KERNEL//
double HostFunctions::varianceFunction(vector<int> data, size_t local_size, int dataSize, int mean) {
	//Create output vector
	vector<int> output(data.size());
	
	//Create output size in bytes
	size_t output_size = data.size() * sizeof(int);//size in bytes

	//Create buffers and events
	cl::Buffer input_buffer(context, CL_MEM_READ_ONLY, input_size);
	cl::Buffer output_buffer(context, CL_MEM_READ_WRITE, output_size);
	cl::Event prof_event;
	cl::Event read_event;
	cl::Event write_event;
	
	//DEVICE OPERATIONS

	//copy vector to buffer and initialise other arrays on device memory
	queue.enqueueWriteBuffer(input_buffer, CL_TRUE, 0, input_size, &data[0], NULL, &write_event);
	queue.enqueueFillBuffer(output_buffer, 0, 0, output_size);//zero B buffer on device memory

	//Set Kernel Arguments
	cl::Kernel kernel = cl::Kernel(program, "variance_local");
	kernel.setArg(0, input_buffer);
	kernel.setArg(1, output_buffer);
	kernel.setArg(2, cl::Local(local_size * sizeof(int)));//local memory size
	kernel.setArg(3, mean);
	kernel.setArg(4, dataSize);

	//Queue the kernel to be run 
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event);
	//Copy the result from device to host
	queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, output_size, &output[0], NULL, &read_event);
	//Print out the execution time 
	cout << "PROFILING INFORMATION:\nKernel Name: variance_local"<< endl;
	cout << "Read time [ns]: " << read_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
		read_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
	cout << "Write time [ns]: " << write_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
		write_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
	cout << "Kernel execution time [ns]: " <<
		prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
		prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << "\n" << endl;
	
	return sqrt(double(output[0] / dataSize) / 10);
}


string HostFunctions::GetFullProfilingInfo(const cl::Event& evnt, int resolution) {
	stringstream sstream;

	sstream << "Queued " << (evnt.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>() - evnt.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>()) / resolution;
	sstream << ", Submitted " << (evnt.getProfilingInfo<CL_PROFILING_COMMAND_START>() - evnt.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>()) / resolution;
	sstream << ", Executed " << (evnt.getProfilingInfo<CL_PROFILING_COMMAND_END>() - evnt.getProfilingInfo<CL_PROFILING_COMMAND_START>()) / resolution;
	sstream << ", Total " << (evnt.getProfilingInfo<CL_PROFILING_COMMAND_END>() - evnt.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>()) / resolution;

	switch (resolution) {
	case 1: sstream << " [ns]"; break;
	case 1000: sstream << " [us]"; break;
	
	default: break;
	}

	return sstream.str();
}
