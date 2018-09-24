#pragma once
#include"cuda_helper.cuh"

#define NUM_ELEMENTS 4096


/********interleaved type**********************/
//define an interleaved type
typedef struct{
	u32 a;
	u32 b;
	u32 c;
	u32 d;
}INTERLEAVED_T;
//define an array based on the interleaved structure
typedef INTERLEAVED_T INTERLAEVED_ARRAY_T[NUM_ELEMENTS];

/********Non_interleaved type*******************/
typedef u32 ARRAY_MEMBER_T[NUM_ELEMENTS];
typedef struct{
	ARRAY_MEMBER_T a;
	ARRAY_MEMBER_T b;
	ARRAY_MEMBER_T c;
	ARRAY_MEMBER_T d;
}NON_INTERLEAVED_T;


__host__ void add_test_non_interleaved_cpu(
	NON_INTERLEAVED_T * const host_dest_ptr,
	const NON_INTERLEAVED_T * const host_src_ptr,
	const u32 iter, const u32 num_elements){


	clock_t start, stop;
	start = clock();
	
	for (u32 tid = 0; tid < num_elements; tid++){
		for (u32 i = 0; i < iter; i++){
			host_dest_ptr->a[tid] += host_src_ptr->a[tid];
			host_dest_ptr->b[tid] += host_src_ptr->b[tid];
			host_dest_ptr->c[tid] += host_src_ptr->c[tid];
			host_dest_ptr->d[tid] += host_src_ptr->d[tid];
		}
	}
	stop = clock();
	double delta = (double)(stop - start) / CLOCKS_PER_SEC;
	printf("Noninterleave CPU excute time is %0.4f ms\n", delta*1000);
}



__host__ void add_test_interleaved_cpu_0(
	INTERLAEVED_ARRAY_T  host_dest_ptr,
	const INTERLAEVED_ARRAY_T  const host_src_ptr,
	const u32 iter, const u32 num_elements){

	clock_t start, stop;
	start = clock();
	for (u32 tid = 0; tid < num_elements; tid++){
		for (u32 i = 0; i < iter; i++){
			host_dest_ptr[tid].a += host_src_ptr[tid].a;
			host_dest_ptr[tid].b += host_src_ptr[tid].b;
			host_dest_ptr[tid].c += host_src_ptr[tid].c;
			host_dest_ptr[tid].d += host_src_ptr[tid].d;

		}
	}
	stop = clock();
	double delta = (double)(stop - start) / CLOCKS_PER_SEC;
	printf("interleave CPU excute time is %0.4f ms\n", delta*1000);
}	
__host__ void add_test_interleaved_cpu(
	INTERLEAVED_T*  host_dest_ptr,
	const INTERLEAVED_T* const host_src_ptr,
	const u32 iter, const u32 num_elements){

	clock_t start, stop;
	start = clock();
	for (u32 tid = 0; tid < num_elements; tid++){
		for (u32 i = 0; i < iter; i++){
			host_dest_ptr[tid].a += host_src_ptr[tid].a;
			host_dest_ptr[tid].b += host_src_ptr[tid].b;
			host_dest_ptr[tid].c += host_src_ptr[tid].c;
			host_dest_ptr[tid].d += host_src_ptr[tid].d;

		}
	}
	stop = clock();
	double delta = (double)(stop - start) / CLOCKS_PER_SEC;
	printf("interleave CPU excute time is %0.4f ms\n", delta*1000);
}
                   
__global__ void add_kernel_interleaved(
	INTERLEAVED_T * const dev_dest_ptr,
	const INTERLEAVED_T * const dev_src_ptr,
	const u32 iter, const u32 num_elements){

	const u32 tid= threadIdx.x + blockDim.x * blockIdx.x;

    if (tid < num_elements){
		for (u32 i = 0; i < iter; i++){
			dev_dest_ptr[tid].a += dev_src_ptr[tid].a;
			dev_dest_ptr[tid].b += dev_src_ptr[tid].b;
			dev_dest_ptr[tid].c += dev_src_ptr[tid].c;
			dev_dest_ptr[tid].d += dev_src_ptr[tid].d;
		}	
	}

}

__global__ void add_kernel_Non_interleaved(
	NON_INTERLEAVED_T * const dev_dest_ptr,
	const NON_INTERLEAVED_T * const dev_src_ptr,
	const u32 iter, const u32 num_elements){

	const u32 tid = threadIdx.x + blockDim.x * blockIdx.x;

	if (tid < num_elements){
		for (u32 i = 0; i < iter; i++){
			dev_dest_ptr->a[tid] += dev_src_ptr->a[tid];
			dev_dest_ptr->b[tid] += dev_src_ptr->b[tid];
			dev_dest_ptr->c[tid] += dev_src_ptr->c[tid];
			dev_dest_ptr->d[tid] += dev_src_ptr->d[tid];
		}
	}
}

void _globalAddCudaTest(){

	const u32 iter = 1000;
	const u32 num_elements = NUM_ELEMENTS;
	const u32 threadsPerBlock = 256;
	const u32 blocksPerGrid = (num_elements+ threadsPerBlock -1)/threadsPerBlock;

	//for interleaved test
	INTERLEAVED_T * host_dest_ptr;
	INTERLEAVED_T * host_src_ptr;

	INTERLEAVED_T * dev_dest_ptr;
	INTERLEAVED_T * dev_src_ptr;

	cudaEvent_t start, stop;

	checkCudaErrors(cudaHostAlloc(&host_dest_ptr, sizeof(INTERLEAVED_T)*num_elements, cudaHostAllocDefault));
	checkCudaErrors(cudaHostAlloc(&host_src_ptr, sizeof(INTERLEAVED_T)*num_elements, cudaHostAllocDefault));
	checkCudaErrors(cudaMalloc(&dev_dest_ptr, sizeof(INTERLEAVED_T)*num_elements));
	checkCudaErrors(cudaMalloc(&dev_src_ptr, sizeof(INTERLEAVED_T)*num_elements));
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	for (int i = 0; i < num_elements;i++){
	host_src_ptr[i].a = 3;
	host_src_ptr[i].b = 2;
	host_src_ptr[i].c = 1;
	host_src_ptr[i].d = 2;
	}

	
	//add_test_interleaved_cpu(host_dest_ptr, host_src_ptr, iter, num_elements);
	
	checkCudaErrors(cudaMemcpy(dev_src_ptr, host_src_ptr, sizeof(INTERLEAVED_T)*num_elements, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaEventRecord(start, 0));

	add_kernel_interleaved << < blocksPerGrid, threadsPerBlock >> >(dev_dest_ptr,dev_src_ptr,iter,num_elements);

	checkCudaErrors(cudaEventRecord(stop, 0));

	checkCudaErrors(cudaEventSynchronize(stop));

	float elapsedTime = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
	printf("interleave Kernel excute time is %0.4f ms\n", elapsedTime);

	checkCudaErrors(cudaMemcpy(host_dest_ptr, dev_dest_ptr, sizeof(INTERLEAVED_T)*num_elements, cudaMemcpyDeviceToHost));

    //for noninterleaved test

	NON_INTERLEAVED_T * n_host_dest_ptr;
	NON_INTERLEAVED_T * n_host_src_ptr;

	NON_INTERLEAVED_T * n_dev_dest_ptr;
	NON_INTERLEAVED_T * n_dev_src_ptr;

	cudaEvent_t nstart, nstop;

	checkCudaErrors(cudaHostAlloc(&n_host_dest_ptr, sizeof(NON_INTERLEAVED_T),cudaHostAllocDefault));
	checkCudaErrors(cudaHostAlloc(&n_host_src_ptr, sizeof(NON_INTERLEAVED_T), cudaHostAllocDefault));
	checkCudaErrors(cudaMalloc(&n_dev_dest_ptr, sizeof(NON_INTERLEAVED_T)));
	checkCudaErrors(cudaMalloc(&n_dev_src_ptr, sizeof(NON_INTERLEAVED_T)));

	checkCudaErrors(cudaEventCreate(&nstart));
	checkCudaErrors(cudaEventCreate(&nstop));

	for (int i = 0; i < num_elements; i++){
		n_host_src_ptr->a[i] = 3;
		n_host_src_ptr->b[i] = 2;
		n_host_src_ptr->c[i] = 1;
		n_host_src_ptr->d[i] = 2;
	}

	//add_test_non_interleaved_cpu(n_host_dest_ptr, n_host_src_ptr, iter, num_elements);
	

	checkCudaErrors(cudaMemcpy(n_dev_src_ptr, n_host_src_ptr, sizeof(NON_INTERLEAVED_T),cudaMemcpyHostToDevice));

	checkCudaErrors(cudaEventRecord(nstart,0));
	
	add_kernel_Non_interleaved << < blocksPerGrid, threadsPerBlock >> >(n_dev_dest_ptr, n_dev_src_ptr, iter, num_elements);

	checkCudaErrors(cudaEventRecord(nstop, 0));
	checkCudaErrors(cudaEventSynchronize(nstop));
	float nelapsedTime = 0.0f;

	checkCudaErrors(cudaEventElapsedTime(&nelapsedTime, nstart, nstop));

	printf("Non_interleave Kernel excute time is %0.4f ms\n", nelapsedTime);

	checkCudaErrors(cudaMemcpy(n_host_dest_ptr, n_dev_dest_ptr, sizeof(INTERLAEVED_ARRAY_T), cudaMemcpyDeviceToHost));


	checkCudaErrors(cudaFreeHost(host_dest_ptr));
	checkCudaErrors(cudaFreeHost(host_src_ptr));
	checkCudaErrors(cudaFreeHost(n_host_dest_ptr));
	checkCudaErrors(cudaFreeHost(n_host_src_ptr));
	checkCudaErrors(cudaFree(n_dev_dest_ptr));
	checkCudaErrors(cudaFree(n_dev_src_ptr));
	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(stop));
	checkCudaErrors(cudaEventDestroy(nstart));
	checkCudaErrors(cudaEventDestroy(nstop));
}
//int main(){
//	cudaDeviceReset();
//	checkCudaErrors(cudaSetDevice(0));
//	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1); //L1
//	//cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
//	srand((unsigned)time(NULL));
//
//	_globalAddCudaTest();
//	return 0;
//}