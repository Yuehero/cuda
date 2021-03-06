#pragma once
#include"cuda_helper.cuh"

__global__ void addKernel(int *c, const int *a, const int *b)
{
 
   int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

// Helper function for using CUDA to add vectors in parallel.
void addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
 
    // Choose which GPU to run on, change this on a multi-GPU system.
	checkCudaErrors(cudaSetDevice(0));

    // Allocate GPU buffers for three vectors (two input, one output)    .
	checkCudaErrors(cudaMalloc((void**)&dev_c, size * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&dev_a, size * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&dev_b, size * sizeof(int)));


    // Copy input vectors from host memory to GPU buffers.
	checkCudaErrors(cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice));
	

    // Launch a kernel on the GPU with one thread for each element.
    addKernel <<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
	checkCudaErrors(cudaGetLastError());
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
	checkCudaErrors(cudaDeviceSynchronize());

    // Copy output vector from GPU buffer to host memory.
	checkCudaErrors(cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(dev_c));
	checkCudaErrors(cudaFree(dev_a));
	checkCudaErrors(cudaFree(dev_b));
}
void _addTest(){
	const int arraySize = 5;
	const int a[arraySize] = { 1, 2, 3, 4, 5 };
	const int b[arraySize] = { 10, 20, 30, 40, 50 };
	int c[arraySize] = { 0 };

	// Add vectors in parallel.
	addWithCuda(c, a, b, arraySize);

	printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
		c[0], c[1], c[2], c[3], c[4]);
	checkCudaErrors(cudaDeviceReset());
}

//int main(){
//	_addTest();
//	return 0;
//}
