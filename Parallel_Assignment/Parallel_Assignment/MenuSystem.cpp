#include "MenuSystem.h"
#include <cstdlib>
#include <chrono>
#include <iostream>
MenuSystem::MenuSystem(size_t localSize, cl::Context cont, cl::CommandQueue q, cl::Program p)
{
	//Initalise all the OPENCL arguments and initialise fileread instance to read file
	//Initialise instance of HostFunctions Class to run the kernel functions
	file_name = chooseDataset();
	read = new FileRead(file_name, localSize);
	data = read->readData();
	input_elements = data.size();//number of input elements
	input_size = data.size() * sizeof(int);//size in bytes
	local_size = localSize;

	hostClient = new HostFunctions(input_elements, input_size, cont, q, p);
}

MenuSystem::~MenuSystem()
{
	cout << "Program End" << endl;
}

string MenuSystem::chooseDataset() {
	//Choose which dataset the user wants to use, return the file name string
	int choice;
	cout << "Please choose which dataset you would like to use:\n\n 1. temp_lincolnshire_short.txt\n 2. temp_lincolnshire.txt\n" << endl;
	cout << "Choice: ";
	cin >> choice;
	switch (choice) {
	case 1:
		system("CLS");
		return "temp_lincolnshire_short.txt";
	case 2:
		system("CLS");
		return "temp_lincolnshire.txt";
	default:
		return "ERROR";
	}
}

void MenuSystem::chooseFunction() {
	//Initialise variables including chrono clocks to calculate duration of operations
	bool loop = true;
	double value;
	vector<int> outputVector;
	using clock = chrono::system_clock;
	using ms = chrono::milliseconds;

	//create padding size for sorting purposes, due to all the padding being pushed to the start 
	size_t padding_size = read->getDataSize() % local_size;

	//loop through the choices
	while (loop == true) {
		int choice;
		cout << "Please choose which function you would like to use:\n 1. Min\n 2. Max\n 3. Mean\n 4. Median\n 5. Upper Quartile\n 6. Lower Quartile\n 7. Standrd Deviation\n" << endl;
		cout << "Choice: ";
		cin >> choice;
		cout << "\n";
		auto before = chrono::system_clock::now();
		switch(choice) {
			case 0: 
				loop = false;
				break;
		
			case 1: 
				//initialise start time of operation
				before = chrono::system_clock::now();
				value = double(hostClient->localFunctions(data, local_size, read->getDataSize(), "reduce_min_local")[0]) / 10;
				cout <<"Min: "<<value<<"\n"<<endl;
				break;
		
			case 2: 
				//initialise start time of operation
				before = chrono::system_clock::now();
				value = double(hostClient->localFunctions(data, local_size,read->getDataSize(),"reduce_max_local")[0]) / 10;
				cout <<"Max: "<< value << "\n" << endl;
				break;

			case 3:
				//initialise start time of operation
				before = chrono::system_clock::now();
				value = (double(hostClient->localFunctions(data, local_size, read->getDataSize(),"reduce_add")[0])/read->getDataSize())/10;
				cout <<"Mean: "<< value << "\n" << endl;
				break;

			case 4:
				//initialise start time of operation
				before = chrono::system_clock::now();
				//check to see if sorted vector already exists;
				if (outputVector.size() > 0)
				{
					//print the median
					cout << "Median: " << double(outputVector[padding_size + (read->getDataSize() / 2)]) / 10 << "\n" << endl;
				}
				else {
					//if sorted array doesnt exist then run the sort function
					outputVector = hostClient->sortFunction(data, local_size, read->getDataSize());
					//print median by dividing by 2
					cout << "Median: " << double(outputVector[padding_size + (read->getDataSize() / 2)]) / 10 << "\n" << endl;
				}
				
				break;

			case 5:
				//initialise start time of operation
				before = chrono::system_clock::now();
				//check to see if sorted vector already exists;
				if (outputVector.size() > 0)
				{
					//print the upper quartile
					cout << "Upper Quartile: " << double(outputVector[padding_size + (read->getDataSize() * 0.75)]) / 10 << "\n" << endl;
				}
				else {
					//if sorted array doesnt exist then run the sort function
					outputVector = hostClient->sortFunction(data, local_size, read->getDataSize());
					//print upper quartile value by multiplying the data size (minus the padding) by 0.75
					cout << "Upper Quartile: " << double(outputVector[padding_size + (read->getDataSize() * 0.75)]) / 10 << "\n" << endl;
				}
				
				break;

			case 6:
				//initialise start time of operation
				before = chrono::system_clock::now();
				//check to see if sorted vector already exists;
				if (outputVector.size() > 0)
				{
					//print the lower quartile
					cout << "Lower Quartile: " << double(outputVector[padding_size + (read->getDataSize() * 0.25)]) / 10 << "\n" << endl;
				}
				else {
					//if sorted array doesnt exist then run the sort function
					outputVector = hostClient->sortFunction(data, local_size, read->getDataSize());
					//print lower quartile value by multiplying the data size (minus the padding) by 0.25
					cout << "Lower Quartile: " << double(outputVector[padding_size + (read->getDataSize() * 0.25)]) / 10 << "\n" << endl;
				}
				break;

			case 7:
				//initialise start time of operation 
				before = chrono::system_clock::now();
				//get the sum of the dataset
				int sum = hostClient->localFunctions(data, local_size, read->getDataSize(), "reduce_add")[0];
				//get the mean of the dataset
				int mean = sum / read->getDataSize();
				//calculate standard deviation of dataset using mean 
				double std = hostClient->varianceFunction(data, local_size, read->getDataSize(), mean);
				cout <<"Standard Deviation: "<< std <<"\n" <<endl;
				break;
		}
		const auto duration = chrono::duration_cast<ms>(clock::now() - before);
		cout << "Calculation Time: " << double(duration.count()) << "[ns]" << endl;
		system("PAUSE");
		system("CLS");
	}
}

