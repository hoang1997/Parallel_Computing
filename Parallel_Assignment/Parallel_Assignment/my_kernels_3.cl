
//REDUCE_ADD_LOCAL
kernel void reduce_add(global const int* input, global int* output, local int* scratch) {
	//get local, global ids and the local size
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	//set local to global
	scratch[lid] = input[id];
	//wait for all threads to be done
	barrier(CLK_LOCAL_MEM_FENCE);

	//loop through all local values
	for (int i = 1; i < N; i *= 2) {
		
		//add next value to current value
		if (!(lid % (i * 2)) && ((lid + i) < N))
			scratch[lid] += scratch[lid + i];
		//wait for all threads to finish
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	

	//atomic add all values
	if (!lid) {
		atomic_add(&output[0], scratch[lid]);
	}
}

// REDUCE_MIN_LOCAL
kernel void reduce_min_local(global const int* input, global int* output, local int* scratch)
{
	//set local id
	int lid = get_local_id(0);
	//set local thread value to global memory data
	scratch[lid] = input[get_global_id(0)];
	//wait for all threads to finish
	barrier(CLK_LOCAL_MEM_FENCE);
	//loop through local memory and bitshift i 
	for (int i = get_local_size(0) / 2; i > 0; i >>= 1)
	{
		if (lid < i)
		{
			//check to see if value is lower than next
			int next = scratch[lid + i];
			int current = scratch[lid];
			scratch[lid] = (current < next) ? current : next;
		}
		//wait for all threads to finish
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//set min value
	if (!lid)
		atomic_min(&output[0], scratch[lid]);
}


// REDUCE_MAX_LOCAL
kernel void reduce_max_local(__global const int* input, __global int* output, __local int* scratch)
{
	//set local id
	int lid = get_local_id(0);
	//set local id data to input data
	scratch[lid] = input[get_global_id(0)];
	//wait for all threads to finish
	barrier(CLK_LOCAL_MEM_FENCE);
	//loop through the local memory by bitshifting 1
	for (int i = get_local_size(0) / 2; i > 0; i >>= 1)
	{
		// Ensure that i does not decrement beyond the local id 
		if (lid < i)
		{
			//check which is greater
			int next = scratch[lid + i];
			int current = scratch[lid];
			scratch[lid] = (current > next) ? current : next;
		}
		//wait for all threads to finish
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//set first value of output to max value
	if (!lid)
		atomic_max(&output[0], scratch[lid]);
}


// SUM OF SQUARED DIFFERENCE KERNEL
kernel void variance_local(global const int* input, global int* output, local int* scratch, int mean, int dataSize)
{
	//get global id and local id
	int id = get_global_id(0);
	int lid = get_local_id(0);
	
	//set local data to input data
	scratch[lid] = input[id];

	//check to see if the global id is less than the data size without the padding
	if (id < dataSize)
	{
		// Calculate the input[id] - mean and copy its squared value to the scratch at index lid.
		int diff = input[id] - mean;
		scratch[lid] = (diff * diff) / 10;
	}

	// Wait for all threats to finish calculating the squared difference for each data 
	barrier(CLK_LOCAL_MEM_FENCE);

	//Loop through local memory and iterate by bitshifting i
	for (int i = get_local_size(0) / 2; i > 0; i >>= 1)
	{
		// Ensure that i does not decrement beyond the local id 
		if (lid < i)
			//similiar to reduce add
			scratch[lid] += scratch[lid + i];

		// Wait for all threads to finish
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//atomic add all the local memory data to the first position of output
	if (!lid)
		atomic_add(&output[0], scratch[lid]);
	
}


//SORTING KERNEL GLOBAL USING ODD EVEN SORT
void swap(global int* A, global int* B) {
	if (*A > *B) {
		int t = *A;
		*A = *B;
		*B = t;
	}
}

kernel void sort_even(global int* A) {
	int id = get_global_id(0);
	int N = get_global_size(0);

	if (id % 2 == 0 && id + 1 < N) //even
		swap(&A[id], &A[id + 1]);

}

kernel void sort_odd(global int* A) {
	int id = get_global_id(0);
	int N = get_global_size(0);

	if (id % 2 == 1 && id + 1 < N) //even
		swap(&A[id], &A[id + 1]);

}


