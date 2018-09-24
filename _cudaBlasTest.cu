#pragma once
#include"cuda_helper.cuh"
#include <cublas_v2.h>
#define CUBLAS_CALL(x){const cublasStatus_t a=(x);if (a!=CUBLAS_STATUS_SUCCESS){printf("\nCUBLAS Error(err_num=%d) \n", a);cudaDeviceReset();assert(0);}}

#define CUDA_CALL(x){const cudaError_t a=(x);if (a!=cudaSuccess){printf("\nCUDA Error(err_num=%d) \n",cudaGetErrorString(a),a);cudaDeviceReset();assert(0);}}

__host__ void print_arr(const float * __restrict__ const data1,
	const float * __restrict__ const data2,
	const float * __restrict__ const data3,
	const int num_element, const char *const prefix){
	printf("\n %s", prefix);
	for (int i = 0; i < num_element; i++)
		printf("\n%2d: %2.4f %2.4f %2.4f", i + 1, data1[i], data2[i], data3[i]);
}
template<class T>
__host__ void fillHostArray(T* data, const u32 num_elements){
	for (u32 i = 0; i < num_elements; i++)
		data[i] = rand() % (num_elements + 1);
}
void _cublasTest(){
	const int num_elem = 8;
	const size_t nbytes = sizeof(float)*num_elem;

	float * host_src_ptr_A;
	float * host_dst_ptr;
	float * host_dst_ptr_A;
	float * host_dst_ptr_B;

	float * device_src_ptr_A;
	float * device_src_ptr_B;
	float * device_dst_ptr;

	checkCudaErrors(cudaHostAlloc(&host_src_ptr_A, nbytes, cudaHostAllocDefault));
	checkCudaErrors(cudaHostAlloc(&host_dst_ptr, nbytes, cudaHostAllocDefault));
	checkCudaErrors(cudaHostAlloc(&host_dst_ptr_A, nbytes, cudaHostAllocDefault));
	checkCudaErrors(cudaHostAlloc(&host_dst_ptr_B, nbytes, cudaHostAllocDefault));

	checkCudaErrors(cudaMalloc(&device_src_ptr_A, nbytes));
	checkCudaErrors(cudaMalloc(&device_src_ptr_B, nbytes));
	checkCudaErrors(cudaMalloc(&device_dst_ptr, nbytes));

	memset(host_dst_ptr, 0, nbytes);
	memset(host_dst_ptr_A, 0, nbytes);
	memset(host_dst_ptr_B, 0, nbytes);

	//init cublas
	cublasHandle_t cublas_handle;
	CUBLAS_CALL(cublasCreate(&cublas_handle));
	//print libiary version

	int version;
	CUBLAS_CALL(cublasGetVersion(cublas_handle, &version));
	printf("\n Using CUBLAS Version: %d", version);

	fillHostArray(host_src_ptr_A, num_elem);
	print_arr(host_src_ptr_A, host_dst_ptr_B, host_dst_ptr, num_elem, "before set");

	const int num_rows = num_elem;
	const int num_cols = 1;
	const size_t elem_size = sizeof(float);

	//copy matrix to device

	CUBLAS_CALL(cublasSetMatrix(num_rows, num_cols, elem_size, host_src_ptr_A, num_rows, device_src_ptr_A, num_rows));

	//clear device memory
	checkCudaErrors(cudaMemset(device_dst_ptr, 0, nbytes));
	checkCudaErrors(cudaMemset(device_src_ptr_B, 0, nbytes));

	//saxpy  y=ax+y;
	const int stride = 1;
	float alpha = 2.0f;
	CUBLAS_CALL(cublasSaxpy(cublas_handle, num_elem, &alpha, device_src_ptr_A, stride, device_src_ptr_B, stride));

	alpha = 3.0f;
	CUBLAS_CALL(cublasSaxpy(cublas_handle, num_elem, &alpha, device_src_ptr_A, stride, device_dst_ptr, stride));

	// calculate the index of the max of each matrix writing the result directly to host memory
	int host_max_idxA, host_max_idxB, host_max_idx_dst;
	CUBLAS_CALL(cublasIsamax(cublas_handle, num_elem, device_src_ptr_A, stride, &host_max_idxA));
	CUBLAS_CALL(cublasIsamax(cublas_handle, num_elem, device_src_ptr_B, stride, &host_max_idxB));
	CUBLAS_CALL(cublasIsamax(cublas_handle, num_elem, device_dst_ptr, stride, &host_max_idx_dst));

	//calculate each sum of each matrix, writing the result directly to host memory

	float host_sumA, host_sumB, host_sum_dst;
	CUBLAS_CALL(cublasSasum(cublas_handle, num_elem, device_src_ptr_A, stride, &host_sumA));
	CUBLAS_CALL(cublasSasum(cublas_handle, num_elem, device_src_ptr_B, stride, &host_sumB));
	CUBLAS_CALL(cublasSasum(cublas_handle, num_elem, device_dst_ptr, stride, &host_sum_dst));

	//copy device version back to host to print out
	CUBLAS_CALL(cublasGetMatrix(num_rows, num_cols, elem_size, device_src_ptr_A,
		num_rows, host_dst_ptr_A, num_rows));
	CUBLAS_CALL(cublasGetMatrix(num_rows, num_cols, elem_size, device_src_ptr_B,
		num_rows, host_dst_ptr_B, num_rows));
	CUBLAS_CALL(cublasGetMatrix(num_rows, num_cols, elem_size, device_dst_ptr,
		num_rows, host_dst_ptr, num_rows));
	//make sure any async calls above are complete before we use the host data
    
	const int default_stream = 0;
	checkCudaErrors(cudaStreamSynchronize(default_stream));

	//print out the arrays	
	print_arr(host_dst_ptr_A, host_dst_ptr_B, host_dst_ptr, num_elem, "after set"); 
	printf("\nIDX of max values: %d, %d, %d", host_max_idxA, host_max_idxB, host_max_idx_dst);
	printf("\nSUM of max values: %2.2f, %2.2f, %2.2f", host_sumA, host_sumB, host_sum_dst);

	//free
	CUBLAS_CALL(cublasDestroy(cublas_handle));
	checkCudaErrors(cudaFreeHost(host_src_ptr_A));
	checkCudaErrors(cudaFreeHost(host_dst_ptr));
	checkCudaErrors(cudaFreeHost(host_dst_ptr_B));
	checkCudaErrors(cudaFreeHost(host_dst_ptr_A));
	checkCudaErrors(cudaFree(device_src_ptr_A));
	checkCudaErrors(cudaFree(device_src_ptr_B));
	checkCudaErrors(cudaFree(device_dst_ptr));
	checkCudaErrors(cudaDeviceReset());
}

//int main(int argc, char **argv)
//{
//	printf("> %s Starting...\n\n", argv[0]);
//	srand((unsigned)time(NULL));
//
//	_cublasTest();
//
//	exit(EXIT_SUCCESS);
//}