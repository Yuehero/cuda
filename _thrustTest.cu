#pragma once
#include"cuda_helper.cuh"
#include<thrust/host_vector.h>
#include<thrust/device_vector.h>
#include<thrust/generate.h>
#include<thrust/sort.h>
#include<thrust/copy.h>
#include<thrust/sequence.h>
#define NUM_ELEM (1024*1024)
#define NUM_ELEM_START (1024*1024)
#define NUM_ELEM_END (1024*1024*10)
void _sortThrustTest(){
	thrust::host_vector<u32> host_arr(NUM_ELEM);
	thrust::generate(host_arr.begin(),host_arr.end(),rand);
	thrust::device_vector<u32> device_arr = host_arr;
	thrust::sort(device_arr.begin(),device_arr.end());
	thrust::sort(host_arr.begin(),host_arr.end());
	thrust::host_vector<u32> host_sorted_arr = device_arr;
	bool flag = false;
	for (u32 i = 0; i < NUM_ELEM; i++){
		if (host_sorted_arr[i] != host_arr[i])
			flag = true;
	}
	if (flag == false)
		printf("> Test passed.\n");
	else
		printf("> Test failed.\n");
}
long int reduce_serial(const int *__restrict__ const host_raw_ptr,const int num_elements){
	long int sum = 0;
	for (int i = 0; i < num_elements; i++)
		sum += host_raw_ptr[i];
	return sum;
}
long int reduce_openmp(const int *__restrict__ const host_raw_ptr, const int num_elements){
	long int sum = 0;
#pragma omp parallel for reduction(+:sum) num_threads(4)
	for (int i = 0; i < num_elements; i++)
		sum += host_raw_ptr[i];
	return sum;
}

void _reduceThrustTest(){
	int num_devices;
	checkCudaErrors(cudaGetDeviceCount(&num_devices));
	printf("\n> GPU num: %d.", num_devices);
	int cur_device = 0;
	checkCudaErrors(cudaSetDevice(cur_device));
	struct cudaDeviceProp device_prop;
	checkCudaErrors(cudaGetDeviceProperties(&device_prop, cur_device));
	printf("\n> Using CUDA Device %u. Device ID: %s on PCI-E %d",cur_device,device_prop.name,device_prop.pciBusID);

	for (unsigned long num_elem = NUM_ELEM_START; num_elem < NUM_ELEM_END; num_elem*=2){
	
		const size_t nbytes= sizeof(int)*num_elem;

		printf("\n> Reducing %lu data items (%lu MB)", num_elem, (nbytes / 1024 / 1024));
	
		float c2d_t, reduce_d_t, reduce_h_t, reduce_h_mp_t, reduce_h_serial_t;
		cudaEvent_t c2d_start, c2d_stop;
		cudaEvent_t sort_d_start, sort_d_stop;

		checkCudaErrors(cudaEventCreate(&c2d_start));
		checkCudaErrors(cudaEventCreate(&c2d_stop));
		checkCudaErrors(cudaEventCreate(&sort_d_start));
		checkCudaErrors(cudaEventCreate(&sort_d_stop));

		thrust::host_vector<int> host_arr(num_elem);
		thrust::sequence(host_arr.begin(),host_arr.end());

		//copy to device
		checkCudaErrors(cudaEventRecord(c2d_start,0));
		thrust::device_vector<int> device_arr = host_arr;
		checkCudaErrors(cudaEventRecord(c2d_stop, 0));
		checkCudaErrors(cudaEventSynchronize(c2d_stop));

		// sort on device
		checkCudaErrors(cudaEventRecord(sort_d_start, 0));
		const long int sum_device = thrust::reduce(device_arr.begin(),device_arr.end());
		checkCudaErrors(cudaEventRecord(sort_d_stop, 0));
		checkCudaErrors(cudaEventSynchronize(sort_d_stop));

		//sort on host
		clock_t start, stop;
		start = clock();
		const long int sum_host = thrust::reduce(host_arr.begin(),host_arr.end());
		stop = clock();
		reduce_h_t = stop - start;

		// allocate host memory
		int * const host_raw_ptr_2 = (int*)malloc(nbytes);
		int * p2 = host_raw_ptr_2;
		for (int i = 0; i < num_elem; i++)
			*p2++ = host_arr[i];

		// host_openmp
		start = clock();
		const long int sum_host_openmp = reduce_openmp(host_raw_ptr_2,num_elem);
		stop = clock();
		reduce_h_mp_t = stop - start;

		// host_serial
		start = clock();
		const long int sum_host_serial = reduce_serial(host_raw_ptr_2, num_elem);
		stop = clock();
		reduce_h_serial_t = stop - start;

		free(host_raw_ptr_2);

		if ((sum_device==sum_host)&& (sum_host_serial == sum_host_openmp))
			printf("\n> reduction matched");
		else
			printf("\n> reduction failed");
    
		checkCudaErrors(cudaEventElapsedTime(&c2d_t, c2d_start, c2d_stop));
		checkCudaErrors(cudaEventElapsedTime(&reduce_d_t, sort_d_start, sort_d_stop));
		printf("\n> copy to device  : %0.2fms", c2d_t);
		printf("\n> reduce on device: %0.2fms", reduce_d_t);
		printf("\n> total on device : %0.2fms", reduce_d_t + c2d_t);
		printf("\n> Thrust reduce on host: %0.2fms", reduce_h_t);
		printf("\n> serial reduce on host: %0.2fms", reduce_h_serial_t);
		printf("\n> openMP reduce on host: %0.2fms", reduce_h_mp_t);

		checkCudaErrors(cudaEventDestroy(c2d_start));
		checkCudaErrors(cudaEventDestroy(c2d_stop));
		checkCudaErrors(cudaEventDestroy(sort_d_start));
		checkCudaErrors(cudaEventDestroy(sort_d_stop));
	}
	checkCudaErrors(cudaDeviceReset());
}
