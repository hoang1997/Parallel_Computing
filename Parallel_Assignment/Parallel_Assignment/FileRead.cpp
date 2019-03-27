#include "FileRead.h"
#include <chrono>


FileRead::FileRead(string file, size_t local_size)
{
	//initialise the file name and local size
	fileName = file;
	lSize = local_size;
}


FileRead::~FileRead()
{
}

//returns file name
string FileRead::getFileName() {
	cout << "\nUsing: " + fileName + "\n" << endl;
	return fileName;
}

//reads and pads the data to a vector 
vector<int> FileRead::readData() {
	string line;
	ifstream infile(fileName);
	data = new vector<int>;
	//get start time
	using clock = chrono::system_clock;
	using ms = chrono::milliseconds;
	auto before = chrono::system_clock::now();
	//loop through each line of the file
	while (getline(infile, line))
	{
		//set string value
		string d;
		//get position of the last space
		size_t found = line.find_last_of(" ");
		//set the value 
		d = line.substr(found + 1);
		//push the value to the data and time by 10 (for accuracy, will convert to double or float at the end)
		data->push_back((stof(d))*10);
	}
	//calculate duration of reading and setting the file
	const auto duration = chrono::duration_cast<ms>(clock::now() - before);
	
	//print load time
	cout << "Load Time: " << double(duration.count())/1000 << "[ms]"<< endl;
	data_size = data->size();

	//set the padding size;
	size_t padding_size = data->size() % lSize;

	//if the input vector is not a multiple of the local_size
	//insert additional neutral elements (0 for addition) so that the total will not be affected
	if (padding_size) {
		//create an extra vector with neutral values
		std::vector<int> A_ext(lSize - padding_size, 0);
		//append that extra vector to our input
		data->insert(data->end(), A_ext.begin(), A_ext.end());
	}

	return *data;
}

vector<int> FileRead::getData() {
	return *data;
}

int FileRead::getDataSize() {
	return data_size;
}


