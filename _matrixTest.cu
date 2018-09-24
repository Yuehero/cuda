#pragma once
#include"cuda_helper.cuh"

typedef unsigned int u32;

#define BlOCK_SIZE  64

typedef struct{

	unsigned int uiWA, uiHA, uiWB, uiHB, uiWC, uiHC;

}sMatrixSize;

__global__ void GEMM(const float *src_A, const float *src_B, float *des_C, int height_A, int width_A, int width_B)
{

	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.x;

	const int offset = y * (gridDim.x * blockDim.x) + x;      // girdDim.x * blockDim.x = width_B
	if (offset > height_A * width_B)
		return;

	const int a_begin = blockIdx.y * (blockDim.y * width_A);
	const int a_step = blockDim.x;
	const int a_end = a_begin + width_A - 1;

	const int b_begin = blockIdx.x * blockDim.x;
	const int b_step = blockIdx.y * width_B;

	register float c_tmp;
	for (int index_a = a_begin, int index_b = b_begin; 
		 index_a < a_end; index_a += a_step, index_b += b_step){
	
		__shared__ float tmp_a[BlOCK_SIZE][BlOCK_SIZE];
		__shared__ float tmp_b[BlOCK_SIZE][BlOCK_SIZE];

		tmp_a[threadIdx.y][threadIdx.x] = src_A[index_a + threadIdx.y * width_A + threadIdx.x];
		tmp_b[threadIdx.y][threadIdx.x] = src_B[index_b + threadIdx.y * width_B + threadIdx.x];
	
		__syncthreads();

		for (int i = 0; i < BlOCK_SIZE; i++)
			c_tmp += tmp_a[threadIdx.y][i] * tmp_b[i][threadIdx.x];
	
		__syncthreads();
	}

	int c_bigSize = blockIdx.y * (blockDim.y * width_B) + blockDim.x * blockIdx.x;
	int c_offset = c_bigSize + (threadIdx.y * width_B) + threadIdx.x;

	des_C[c_offset] = c_tmp;


}  
__host__ void _cpuMatrixMul(const float * mSrcA, const float * mSrcB, float * mDstC, u32 uiHA, u32 uiWA, u32 uiWB){
	for (u32 i = 0; i < uiHA; i++){
		for (u32 j = 0; j < uiWB; j++){
			for (u32 k = 0; k < uiWA; k++){
				mDstC[i*uiWB + j] += mSrcA[uiWA*i + k] * mSrcB[uiWB*k + j];
			}
		}
	}
}
__global__ void _gpuMatrixMul(const float * mSrcA, const float * mSrcB, float * mDstC, u32 uiWA, u32 uiWB){

	u32 tx = threadIdx.x;
	u32 ty = threadIdx.y;
	u32 bx = blockIdx.x;
	u32 by = blockIdx.y;

	u32 aBegin = by* uiWA * BlOCK_SIZE;
	u32 aStep = BlOCK_SIZE;
	u32 aEnd = aBegin + uiWA - 1;

	u32 bBegin = bx * BlOCK_SIZE;
	u32 bStep = uiWB * BlOCK_SIZE;

	register float cSub;
	for (u32 a = aBegin, b = bBegin; a < aEnd; a += aStep, b += bStep){

		__shared__ float Ashared[BlOCK_SIZE][BlOCK_SIZE];
		__shared__ float Bshared[BlOCK_SIZE][BlOCK_SIZE];

		Ashared[ty][tx] = mSrcA[a + ty*uiWA + tx];
		Bshared[ty][tx] = mSrcB[b + ty*uiWB + tx];

		__syncthreads();

		for (u32 k = 0; k < BlOCK_SIZE; k++){
			cSub += Ashared[ty][k] * Bshared[k][tx];
		}

		__syncthreads();
	}
	u32 c = uiWB*by*BlOCK_SIZE + BlOCK_SIZE*bx;
	mDstC[c + tx + ty*uiWB] = cSub;
}
__host__ void randomInit(float *data, int size)
{
	for (int i = 0; i < size; ++i)
		data[i] = rand() / (float)RAND_MAX;
}
__host__ void _dispDeviceInfrom(){
	int num_devices;
	checkCudaErrors(cudaGetDeviceCount(&num_devices));
	printf("> GPU num: %d.\n", num_devices);

	int cur_device = 0;
	//select correct devices
	checkCudaErrors(cudaSetDevice(cur_device));
	struct cudaDeviceProp device_prop;
	checkCudaErrors(cudaGetDeviceProperties(&device_prop, cur_device));
	printf("> ID:%d %s.\n", cur_device, device_prop.name);
}
__host__ void _matrixTest(){

	sMatrixSize matrixSize;
	//host
	float * h_mSrcA;
	float * h_mSrcB;
	float * h_mDstC;
	float * h_mDstC_cpu;
	//device
	float * d_mSrcA;
	float * d_mSrcB;
	float * d_mDstC;
	// 3*4 srcA,4*2 srcB,3*2 dst
	matrixSize.uiHA = 3 * BlOCK_SIZE;
	matrixSize.uiWA = 4 * BlOCK_SIZE;
	matrixSize.uiHB = 4 * BlOCK_SIZE;
	matrixSize.uiWB = 2 * BlOCK_SIZE;
	matrixSize.uiHC = 3 * BlOCK_SIZE;
	matrixSize.uiWC = 2 * BlOCK_SIZE;

	u32 size_A = matrixSize.uiHA*matrixSize.uiWA;
	u32 size_B = matrixSize.uiHB*matrixSize.uiWB;
	u32 size_C = matrixSize.uiHC*matrixSize.uiWC;

	checkCudaErrors(cudaHostAlloc(&h_mSrcA, sizeof(float)*size_A, cudaHostAllocDefault));
	checkCudaErrors(cudaHostAlloc(&h_mSrcB, sizeof(float)*size_B, cudaHostAllocDefault));
	checkCudaErrors(cudaHostAlloc(&h_mDstC, sizeof(float)*size_C, cudaHostAllocDefault));
	h_mDstC_cpu = (float*)malloc(sizeof(float)*size_C);

	checkCudaErrors(cudaMalloc(&d_mSrcA, sizeof(float)*size_A));
	checkCudaErrors(cudaMalloc(&d_mSrcB, sizeof(float)*size_B));
	checkCudaErrors(cudaMalloc(&d_mDstC, sizeof(float)*size_C));

	randomInit(h_mSrcA, size_A);
	randomInit(h_mSrcB, size_C);

	checkCudaErrors(cudaMemcpy(d_mSrcA, h_mSrcA, sizeof(float)*size_A, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_mSrcB, h_mSrcB, sizeof(float)*size_B, cudaMemcpyHostToDevice));
	//cpu
	_cpuMatrixMul(h_mSrcA, h_mSrcB, h_mDstC_cpu, matrixSize.uiHA, matrixSize.uiWA, matrixSize.uiWB);

	dim3 threads(BlOCK_SIZE, BlOCK_SIZE);
	dim3 Grid(matrixSize.uiWB / threads.x, matrixSize.uiHA / threads.y);


	_gpuMatrixMul << <Grid, threads >> >(d_mSrcA, d_mSrcB, d_mDstC, matrixSize.uiWA, matrixSize.uiWB);

	checkCudaErrors(cudaMemcpy(h_mDstC, d_mDstC, sizeof(float)*size_C, cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_mSrcA));
	checkCudaErrors(cudaFree(d_mSrcB));
	checkCudaErrors(cudaFree(d_mDstC));
	checkCudaErrors(cudaFreeHost(h_mSrcA));
	checkCudaErrors(cudaFreeHost(h_mSrcB));
	checkCudaErrors(cudaFreeHost(h_mDstC));

	free(h_mDstC_cpu);

	checkCudaErrors(cudaDeviceReset());

}


