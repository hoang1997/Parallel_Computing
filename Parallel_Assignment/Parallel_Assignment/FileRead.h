#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#pragma once
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
using namespace std;

class FileRead
{
private: 
	
	vector<int> *data;
	string fileName;
	int data_size;
	size_t lSize;
public:
	FileRead(string file, size_t local_size);
	~FileRead();
	
	vector<int> readData();

	string getFileName();

	vector<int> getData();

	int getDataSize();
	
};

