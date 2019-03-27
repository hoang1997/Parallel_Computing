#pragma once
#include <iostream>
#include <vector>
#include <string>
#include "HostFunctions.h"
#include "FileRead.h"

using namespace std;

class MenuSystem
{

private:
	FileRead *read;
	string file_name;
	vector<int> data;
	size_t input_size;
	size_t input_elements;
	HostFunctions *hostClient;
	size_t local_size;

public:
	MenuSystem(size_t localSize, cl::Context cont, cl::CommandQueue q, cl::Program p);
	~MenuSystem();
	string chooseDataset();
	void chooseFunction();

};

