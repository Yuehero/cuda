#pragma once
#include"cuda_helper.cuh"
#define SIZE  (1024*1024)
__shared__ unsigned int d_bin_data_shared[256];// num threads/per block


__global__ void histogramGpu(unsigned char * d_hist_data, unsigned int * d_bin_data){

	unsigned int idx = blockDim.x *blockIdx.x + threadIdx.x;
	unsigned int idy = blockDim.y *blockIdx.y + threadIdx.y;
	unsigned int tid = gridDim.x*blockDim.x*idy + idx;

	unsigned char value = d_hist_data[tid];//1 byte per thread
	atomicAdd(&(d_bin_data[value]), 1);
}
__global__ void histogramGpu_Shared(unsigned char * d_hist_data, unsigned int * d_bin_data){
	d_bin_data_shared[threadIdx.x] = 0;
	unsigned int idx = blockDim.x *blockIdx.x + threadIdx.x;
	unsigned int idy = blockDim.y *blockIdx.y + threadIdx.y;
	unsigned int tid = gridDim.x*blockDim.x*idy + idx;
	
	unsigned char value = d_hist_data[tid];//1 byte per thread
	__syncthreads();

	atomicAdd(&(d_bin_data_shared[value]), 1);
	
	__syncthreads();
	atomicAdd(&(d_bin_data[threadIdx.x]), d_bin_data_shared[threadIdx.x]);

}
__global__ void histogramGpu_Consol(unsigned char * d_hist_data, unsigned int * d_bin_data){

	unsigned int idx = blockDim.x *blockIdx.x + threadIdx.x;
	unsigned int idy = blockDim.y *blockIdx.y + threadIdx.y;
	unsigned int tid = gridDim.x*blockDim.x*idy + idx;
	unsigned int value_32 = d_hist_data[tid]; //4 byte per thread

	atomicAdd(&(d_bin_data[(value_32 & 0x000000ff)]), 1);
	atomicAdd(&(d_bin_data[((value_32 & 0x0000ff00) >> 8)]), 1);
	atomicAdd(&(d_bin_data[((value_32 & 0x00ff0000) >> 16)]), 1);
	atomicAdd(&(d_bin_data[((value_32 & 0xff000000) >> 24)]), 1);

}

__global__ void histogramGpu_CShared(unsigned char * d_hist_data, unsigned int * d_bin_data){

	// clear shared memory
	d_bin_data_shared[threadIdx.x] = 0;
	unsigned int idx = blockDim.x *blockIdx.x + threadIdx.x;
	unsigned int idy = blockDim.y *blockIdx.y + threadIdx.y;
	unsigned int tid = gridDim.x*blockDim.x*idy + idx;
	unsigned int value_32 = d_hist_data[tid]; //4 byte per thread
	__syncthreads();

	atomicAdd(&(d_bin_data_shared[((value_32 & 0x000000ff))]), 1);
	atomicAdd(&(d_bin_data_shared[((value_32 & 0x0000ff00) >> 8)]), 1);
	atomicAdd(&(d_bin_data_shared[((value_32 & 0x00ff0000) >> 16)]), 1);
	atomicAdd(&(d_bin_data_shared[((value_32 & 0xff000000) >> 24)]), 1);

	__syncthreads();
	atomicAdd(&(d_bin_data[threadIdx.x]), d_bin_data_shared[threadIdx.x]);

}

unsigned char * big_random_block(int size)
{
	unsigned char* data = (unsigned char*)malloc(size);
	if (!data)
	{
		printf("Memery allocate failed!\n");
		return NULL;
	}
	for (int i = 0; i < size; i++)
	{
		data[i] = rand();
	}

	return data;
}
void _histogramTest(){
	// host 
	unsigned char *h_hist_data;
	unsigned int  *h_bin_data;
	checkCudaErrors(cudaHostAlloc(&h_hist_data, sizeof(char)*SIZE, cudaHostAllocDefault));
	checkCudaErrors(cudaHostAlloc(&h_bin_data, sizeof(int) * 256, cudaHostAllocDefault));
	//Initialization
	for (int i = 0; i < SIZE; i++)
	{
		h_hist_data[i] = rand();
	}
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));


	//device
	unsigned char *d_hist_data;
	unsigned int * d_bin_data;
	checkCudaErrors(cudaMalloc(&d_hist_data, sizeof(char)*SIZE));
	checkCudaErrors(cudaMalloc(&d_bin_data, sizeof(int) * 256));

	//H2D
	checkCudaErrors(cudaMemcpy(d_hist_data, h_hist_data, SIZE*sizeof(char), cudaMemcpyHostToDevice));

	int nIter = 10;
	dim3 block(256);
	dim3 grid(SIZE / 256);
	checkCudaErrors(cudaEventRecord(start, 0));

	for (int i = 0; i < nIter; i++){
		checkCudaErrors(cudaMemset(d_bin_data, 0, 256 * sizeof(int)));
		histogramGpu_Shared << < grid, block >> >(d_hist_data, d_bin_data);

	}
	// Compute and print the performance
	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	float elapsedTime = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));

	printf("Time to generate: %3.1f ms\n", elapsedTime);
	float msec = elapsedTime / nIter;
	double flops = SIZE*2;
	double gigaFlops = (flops * 1.0e-9f) / (msec / 1000.0f);
	printf("Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops\n",
		gigaFlops, msec, flops);

	// test the Kernel2
	cudaEvent_t start1, stop1;
	checkCudaErrors(cudaEventCreate(&start1));
	checkCudaErrors(cudaEventCreate(&stop1));
	checkCudaErrors(cudaEventRecord(start1, 0));

	for (int i = 0; i < nIter; i++){
		checkCudaErrors(cudaMemset(d_bin_data, 0, 256 * sizeof(int)));
		histogramGpu_CShared << < grid, block >> >(d_hist_data, d_bin_data);

	}
	// Compute and print the performance
	checkCudaErrors(cudaEventRecord(stop1, 0));
	checkCudaErrors(cudaEventSynchronize(stop1));
	float elapsedTime1 = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&elapsedTime1, start1, stop1));
	printf("Time to generate: %3.1f ms\n", elapsedTime1);
	msec = elapsedTime1 / nIter;
	flops = SIZE * 5;
	gigaFlops = (flops * 1.0e-9f) / (msec / 1000.0f);
	printf("Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops\n",
		gigaFlops, msec, flops);

	//D2H
	checkCudaErrors(cudaMemcpy(h_bin_data, d_bin_data, 256 * sizeof(int), cudaMemcpyDeviceToHost));
	//验证与基于CPU计算得到的结果是相同的
	for (int i = 0; i < SIZE; i++)
		h_bin_data[h_hist_data[i]]--;
	for (int i = 0; i < 256; i++){
		if (h_bin_data[i] != 0)
			printf("Failure at %d!\n", i);
	}
	//free

	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(stop));
	checkCudaErrors(cudaEventDestroy(start1));
	checkCudaErrors(cudaEventDestroy(stop1));
	checkCudaErrors(cudaFree(d_hist_data));
	checkCudaErrors(cudaFree(d_bin_data));
	checkCudaErrors(cudaFreeHost(h_hist_data));
	checkCudaErrors(cudaFreeHost(h_bin_data));
}
//int main(){
//	cudaDeviceReset();
//	checkCudaErrors(cudaSetDevice(0));
//	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1); //L1
//
//	//cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
//	srand((unsigned)time(NULL));
//	_histogramTest();
//	return 0;
//}



