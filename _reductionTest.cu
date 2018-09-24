#pragma once
#include"cuda_helper.cuh"

#define N 1024*10
const u32 threadsPerBlock = 256;
const u32 blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

template<class T>
__global__ void reductionGpu_0(T *d_idata, T* d_odata) {

	__shared__ T sdata[threadsPerBlock];

	u32 tid = threadIdx.x;

	u32 index = threadIdx.x + blockIdx.x * blockDim.x;

	sdata[tid] = d_idata[index];

	__syncthreads();

	for (u32 stride = 1; stride < blockDim.x; stride *= 2) {

		if (tid % (2 * stride) == 0)         //exists warp divergence
			sdata[tid] += sdata[tid + stride]; // sdata[0] += sdata[1]; sdata[2] += sdata[3]; 不连续

		__syncthreads();
	}
	if (tid == 0)
		d_odata[blockIdx.x] = sdata[0];
}

template<class T>
__global__ void reductionGpu_1(T *d_idata, T* d_odata) {

	__shared__ T sdata[threadsPerBlock];

	u32 tid = threadIdx.x;

	u32 index = threadIdx.x + blockIdx.x * blockDim.x;

	sdata[tid] = d_idata[index];

	__syncthreads();

	for (u32 stride = 1; stride < blockDim.x; stride *= 2) {

		u32 i = 2 * stride * tid;             // all of threads in a warp will excute this instruction, so the warp diverage not exits

		if (i < blockDim.x) {

			sdata[i] += sdata[i + stride];
		}

		__syncthreads();
	}
	if (tid == 0)
		d_odata[blockIdx.x] = sdata[0];
}

template<class T>
__global__ void reductionGpu_2(T *d_idata, T* d_odata) {

	__shared__ T sdata[threadsPerBlock];

	u32 tid = threadIdx.x;

	u32 index = threadIdx.x + blockDim.x*blockIdx.x;

	sdata[tid] = d_idata[index];

	__syncthreads();

	for (u32 s = blockDim.x / 2; s > 0; s >>= 1) {

		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	if (tid == 0)
		d_odata[blockIdx.x] = sdata[0];
}

template<class T>
__global__ void reductionGpu_3(T *d_idata, T* d_odata) {

	__shared__ T sdata[threadsPerBlock];

	u32 tid = threadIdx.x;

	u32 index = threadIdx.x + blockDim.x*blockIdx.x;

	sdata[tid] = d_idata[index];

	__syncthreads();

	for (u32 s = blockDim.x / 2; s > 32; s >>= 1) {

		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	// tid<32 will not excute the syncthreads(); 
	// warp sync
	if (tid < 32) {
		/*每个线程块中的线程束是按照锁步方式（lockstep）执行每条指令,因此当线程块
		中活动线程数低于硬件线程束的大小时可以无须再调用__syncthreads()来同步。
		编写线程束同步代码时，必须对共享内存的指针使用volatile关键字修饰，
		否则可能会由于编译器的优化行为改变内存的操作顺序从而使结果不正确。*/
		volatile T *warpSum = sdata;
		if (blockDim.x > 32)
			warpSum[tid] += warpSum[tid + 32];
		warpSum[tid] += warpSum[tid + 16];
		warpSum[tid] += warpSum[tid + 8];
		warpSum[tid] += warpSum[tid + 4];
		warpSum[tid] += warpSum[tid + 2];
		warpSum[tid] += warpSum[tid + 1];
		if (tid == 0)
			d_odata[blockIdx.x] = warpSum[0];
	}
}
template<class T>
__global__ void reductionGpu_4(T *d_idata, T* d_odata) {

	__shared__ T sdata[threadsPerBlock];


	u32 tid = threadIdx.x;

	u32 index = threadIdx.x + ( blockDim.x * 2)* blockIdx.x;

	T mySum = (index <N) ? d_idata[index] : 0;

	if (index + blockDim.x < N)
		mySum += d_idata[index + blockDim.x];
	sdata[tid] = mySum;
	__syncthreads();

	for (u32 s = blockDim.x / 2; s > 0; s >>= 1) {

		if (tid < s) {
			sdata[tid] = mySum = mySum + sdata[tid + s];
		}
		__syncthreads();
	}
	if (tid == 0)
		d_odata[blockIdx.x] = sdata[0];
}


//const int COL = 1024;
//const int ROW = 16;
//typedef float mytpe[COL];
//
//const int threadsPerBlock_ = 256;
//const int blocksPerGrid_ = (COL + threadsPerBlock - 1) / threadsPerBlock;
//
//dim3 threads_(threadsPerBlock_, 1);
//dim3 grid_((COL + threadsPerBlock - 1) / threads_.x, ROW / threads_.y);
//
//__global__ void reductionGpu_matrix(mytpe *d_idata, mytpe *d_odata) {
//	int tid = threadIdx.x;
//	int x = threadIdx.x + blockIdx.x * blockDim.x;
//	int y = threadIdx.y + blockIdx.y * blockDim.y; 
//
//
//	__shared__ float cache[ROW][threadsPerBlock_];
//
//	while (tid < COL && y < ROW){
//
//	   cache[y][tid] = d_idata[y][x];
//	   __syncthreads();
//	}	
//	
//	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1){
//		cache[y][tid] += cache[y][tid + stride];
//
//		__syncthreads();	
//	}
//
//	if (tid == 0)
//		d_odata[y][blockIdx.x] = cache[y][0];
//}
void _reductionCudaTest() {

	//define host & device
	float * h_idata;
	float * h_odata;
	float * d_idata;
	float * d_odata;

	//define CudaEvents
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	cudaEvent_t start1, stop1;
	checkCudaErrors(cudaEventCreate(&start1));
	checkCudaErrors(cudaEventCreate(&stop1));

	cudaEvent_t start2, stop2;
	checkCudaErrors(cudaEventCreate(&start2));
	checkCudaErrors(cudaEventCreate(&stop2));

	cudaEvent_t start3, stop3;
	checkCudaErrors(cudaEventCreate(&start3));
	checkCudaErrors(cudaEventCreate(&stop3));


	//malloc host & device
	checkCudaErrors(cudaHostAlloc(&h_idata, sizeof(float)*N, cudaHostAllocDefault));
	checkCudaErrors(cudaHostAlloc(&h_odata, sizeof(float)*blocksPerGrid, cudaHostAllocDefault));
	checkCudaErrors(cudaMalloc(&d_idata, sizeof(float)*N));
	checkCudaErrors(cudaMalloc(&d_odata, sizeof(float)*blocksPerGrid));

	//initialization host input data

	for (u32 i = 0; i < N; i++)
		h_idata[i] = 1;

	//H2D
	checkCudaErrors(cudaMemcpy(d_idata, h_idata, sizeof(float)*N, cudaMemcpyHostToDevice));
	u32 nIter = 10;


	/************************************************/
	/********************kernel 4 *******************/
	/************************************************/
	checkCudaErrors(cudaEventRecord(start, 0));
	for (u32 i = 0; i < nIter; i++){
		checkCudaErrors(cudaMemset(d_odata, 0, blocksPerGrid * sizeof(int)));
		reductionGpu_4 << < blocksPerGrid, threadsPerBlock >> > (d_idata, d_odata);
	}
	checkCudaErrors(cudaMemcpy(h_odata, d_odata, sizeof(float)*blocksPerGrid, cudaMemcpyDeviceToHost));

	float sum0 = 0.f;
	for (u32 i = 0; i < blocksPerGrid; i++)
		sum0 += h_odata[i];
	printf("Final sum is %3.1f\n", sum0);

	// Compute and print the performance
	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	float elapsedTime_0 = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&elapsedTime_0, start, stop));
	printf("reductionGpu_4 Kernel excute time is %0.4f ms\n", elapsedTime_0 / nIter);



	/************************************************/
	/********************kernel 1 *******************/
	/************************************************/
	checkCudaErrors(cudaEventRecord(start1, 0));
	for (u32 i = 0; i < nIter; i++){
		checkCudaErrors(cudaMemset(d_odata, 0, blocksPerGrid * sizeof(int)));
		reductionGpu_1 << < blocksPerGrid, threadsPerBlock >> > (d_idata, d_odata);
	}
	checkCudaErrors(cudaMemcpy(h_odata, d_odata, sizeof(float)*blocksPerGrid, cudaMemcpyDeviceToHost));

	float sum1 = 0.f;
	for (u32 i = 0; i < blocksPerGrid; i++)
		sum1 += h_odata[i];
	printf("Final sum is %3.1f\n", sum1);

	// Compute and print the performance
	checkCudaErrors(cudaEventRecord(stop1, 0));
	checkCudaErrors(cudaEventSynchronize(stop1));
	float elapsedTime_1 = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&elapsedTime_1, start1, stop1));
	printf("reductionGpu_1 Kernel excute time is %0.4f ms\n", elapsedTime_1 / nIter);




	/************************************************/
	/********************kernel 2 *******************/
	/************************************************/
	checkCudaErrors(cudaEventRecord(start2, 0));
	for (u32 i = 0; i < nIter; i++){
		checkCudaErrors(cudaMemset(d_odata, 0, blocksPerGrid * sizeof(int)));
		reductionGpu_2 << < blocksPerGrid, threadsPerBlock >> > (d_idata, d_odata);
	}
	checkCudaErrors(cudaMemcpy(h_odata, d_odata, sizeof(float)*blocksPerGrid, cudaMemcpyDeviceToHost));

	float sum2 = 0.f;
	for (u32 i = 0; i < blocksPerGrid; i++)
		sum2 += h_odata[i];
	printf("Final sum is %3.1f\n", sum1);
	// Compute and print the performance
	checkCudaErrors(cudaEventRecord(stop2, 0));
	checkCudaErrors(cudaEventSynchronize(stop2));
	float elapsedTime_2 = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&elapsedTime_2, start2, stop2));
	printf("reductionGpu_2 Kernel excute time is %0.4f ms\n", elapsedTime_2 / nIter);




	/************************************************/
	/********************kernel 3 *******************/
	/************************************************/
	checkCudaErrors(cudaEventRecord(start3, 0));
	for (u32 i = 0; i < nIter; i++){
		checkCudaErrors(cudaMemset(d_odata, 0, blocksPerGrid * sizeof(int)));
		reductionGpu_3 << < blocksPerGrid, threadsPerBlock >> > (d_idata, d_odata);
	}
	checkCudaErrors(cudaMemcpy(h_odata, d_odata, sizeof(float)*blocksPerGrid, cudaMemcpyDeviceToHost));

	float sum3 = 0.f;
	for (u32 i = 0; i < blocksPerGrid; i++)
		sum3 += h_odata[i];
	printf("Final sum is %3.1f\n", sum3);
	// Compute and print the performance
	checkCudaErrors(cudaEventRecord(stop3, 0));
	checkCudaErrors(cudaEventSynchronize(stop3));
	float elapsedTime_3 = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&elapsedTime_3, start3, stop3));
	printf("reductionGpu_3 Kernel excute time is %0.4f ms\n", elapsedTime_3 / nIter);



	//Free
	checkCudaErrors(cudaFreeHost(h_idata));
	checkCudaErrors(cudaFreeHost(h_odata));
	checkCudaErrors(cudaFree(d_idata));
	checkCudaErrors(cudaFree(d_odata));
	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(stop));
	checkCudaErrors(cudaEventDestroy(start1));
	checkCudaErrors(cudaEventDestroy(stop1));
	checkCudaErrors(cudaEventDestroy(start2));
	checkCudaErrors(cudaEventDestroy(stop2));
	checkCudaErrors(cudaEventDestroy(start3));
	checkCudaErrors(cudaEventDestroy(stop3));
}


int main(){
	cudaDeviceReset();
	checkCudaErrors(cudaSetDevice(0));
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1); //L1

	//cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	srand((unsigned)time(NULL));
	_reductionCudaTest();
	return 0;
}