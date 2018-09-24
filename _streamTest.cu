#pragma once
#include"cuda_helper.cuh"


#define MAX_NUM_DEVICES (1)
#define NUM_ELEM (1024*1024)
#define FULL_DATA_SIZE (NUM_ELEM*12)

// define one stream per GPU
__host__ void fillHostArray(u32* data, const u32 num_elements){
	for (u32 i = 0; i < num_elements; i++)
		data[i] = rand() % (num_elements + 1);
}
__host__ void checkArray(char *device_perfix, u32*data, const u32 num_elements){
	bool error = false;
	for (u32 i = 0; i < num_elements; i++)
		if (data[i] != i * 2){
			printf("%s error: %u %u", device_perfix, i, data[i]);
			error = true;
		}
	if (error == false){
		printf("%s array check passed.", device_perfix);
	}
}
__global__ void gpuKernel(u32 *data){
	int tid = threadIdx.x + blockDim.x*blockIdx.x;
	data[tid] *= 2;
}
void _streamTest_0(){

	cudaStream_t stream[MAX_NUM_DEVICES];
	char device_perfix[MAX_NUM_DEVICES][300];
	u32*gpu_data[MAX_NUM_DEVICES];
	u32*cpu_src_data[MAX_NUM_DEVICES];
	u32*cpu_dest_data[MAX_NUM_DEVICES];

	//cudaEvent_t kernel_start_event[MAX_NUM_DEVICES];
	//cudaEvent_t mempcy_to_start_event[MAX_NUM_DEVICES];
	//cudaEvent_t mempcy_from_start_event[MAX_NUM_DEVICES];
	//cudaEvent_t mempcy_from_stop_event[MAX_NUM_DEVICES];

	//float time_kernel_ms[MAX_NUM_DEVICES];
	//float time_copy_to_ms[MAX_NUM_DEVICES];
	//float time_copy_from_ms[MAX_NUM_DEVICES];
	//float time_exec_ms[MAX_NUM_DEVICES];


	const int shared_memory_usage = 0;
	const size_t single_gpu_chunk_size = sizeof(u32)*NUM_ELEM;

	const u32 threadsPerBlock = 256;
	const u32 blocksPerGrid = (NUM_ELEM + threadsPerBlock - 1) / threadsPerBlock;

	int num_devices;
	checkCudaErrors(cudaGetDeviceCount(&num_devices));
	if (num_devices > MAX_NUM_DEVICES)
		num_devices = MAX_NUM_DEVICES;

	for (int cur_device = 0; cur_device < num_devices; cur_device++){
		//select correct devices
		checkCudaErrors(cudaSetDevice(cur_device));
		struct cudaDeviceProp device_prop;
		checkCudaErrors(cudaGetDeviceProperties(&device_prop, cur_device));
		printf(&device_perfix[cur_device][0], "\nID:%d %s", cur_device, device_prop.name);

		//create a stream
		checkCudaErrors(cudaStreamCreate(&stream[cur_device]));
		//allocate device memory
		checkCudaErrors(cudaMalloc(&gpu_data[cur_device], single_gpu_chunk_size));
		//allocate host memory
		checkCudaErrors(cudaHostAlloc(&cpu_src_data[cur_device], single_gpu_chunk_size, cudaHostAllocDefault));
		checkCudaErrors(cudaHostAlloc(&cpu_dest_data[cur_device], single_gpu_chunk_size, cudaHostAllocDefault));

		fillHostArray(cpu_src_data[cur_device], NUM_ELEM);

		//asynchronous
		//checkCudaErrors(cudaEventRecord(mempcy_to_start_event[cur_device], 0));
		checkCudaErrors(cudaMemcpyAsync(gpu_data[cur_device], cpu_src_data[cur_device], single_gpu_chunk_size, cudaMemcpyHostToDevice, stream[cur_device]));

		//checkCudaErrors(cudaEventRecord(kernel_start_event[cur_device], stream[cur_device]));
		gpuKernel << <blocksPerGrid, threadsPerBlock, shared_memory_usage, stream[cur_device] >> >(gpu_data[cur_device]);

		//checkCudaErrors(cudaEventRecord(mempcy_from_start_event[cur_device], stream[cur_device]));
		checkCudaErrors(cudaMemcpyAsync(cpu_dest_data[cur_device], gpu_data[cur_device], single_gpu_chunk_size, cudaMemcpyDeviceToHost, stream[cur_device]));

		//checkCudaErrors(cudaEventRecord(mempcy_from_stop_event[cur_device], 0));
	}
	//free
	for (int cur_device = 0; cur_device < num_devices; cur_device++){
		//select the correct device
		checkCudaErrors(cudaSetDevice(cur_device));
		//wait for all commands in stream is complete
		checkCudaErrors(cudaStreamSynchronize(stream[cur_device]));

		//checkCudaErrors(cudaEventElapsedTime(&time_copy_to_ms[cur_device], mempcy_to_start_event[cur_device], kernel_start_event[cur_device]));

		//checkCudaErrors(cudaEventElapsedTime(&time_kernel_ms[cur_device], kernel_start_event[cur_device], mempcy_from_start_event[cur_device]));

		//checkCudaErrors(cudaEventElapsedTime(&time_copy_from_ms[cur_device], mempcy_from_start_event[cur_device], mempcy_from_stop_event[cur_device]));

		//checkCudaErrors(cudaEventElapsedTime(&time_exec_ms[cur_device], mempcy_to_start_event[cur_device], mempcy_from_stop_event[cur_device]));

		//printf("> %s Copy to  \t:%.2f ms", device_perfix[cur_device], time_copy_to_ms[cur_device]);

		//printf("> %s Kernel   \t:%.2f ms", device_perfix[cur_device], time_kernel_ms[cur_device]);

		//printf("> %s Copy Back\t:%.2f ms", device_perfix[cur_device], time_copy_from_ms[cur_device]);

		//printf("> %s Execution \t:%.2f ms", device_perfix[cur_device], time_exec_ms[cur_device]);

		//checkArray(device_perfix[cur_device],cpu_dest_data[cur_device],NUM_ELEM);

		checkCudaErrors(cudaStreamDestroy(stream[cur_device]));
		checkCudaErrors(cudaFree(gpu_data[cur_device]));
		checkCudaErrors(cudaFreeHost(cpu_src_data[cur_device]));
		checkCudaErrors(cudaFreeHost(cpu_dest_data[cur_device]));


		//checkCudaErrors(cudaEventDestroy(kernel_start_event[cur_device]));
		//checkCudaErrors(cudaEventDestroy(mempcy_to_start_event[cur_device]));
		//checkCudaErrors(cudaEventDestroy(mempcy_from_start_event[cur_device]));
		//checkCudaErrors(cudaEventDestroy(mempcy_from_stop_event[cur_device]));
		cudaDeviceReset();
	}
}

__global__ void gpuKernel_1(const u32 *src_arr0, const u32* src_arr1, u32 *const dst_arr){
	int tid = threadIdx.x + blockIdx.x*blockDim.x;

	if (tid < NUM_ELEM){
		register u32 idx0 = (tid + 1) % 256;
		register u32 idx1 = (tid + 2) % 256;
		register float as = (src_arr0[tid] + src_arr0[idx0] + src_arr0[idx1]) / 3.0f;
		register float bs = (src_arr1[tid] + src_arr1[idx0] + src_arr1[idx1]) / 3.0f;
		dst_arr[tid] = (as + bs) / 2;
	}

}
void _multipleStreamTest(){

	int num_devices;
	checkCudaErrors(cudaGetDeviceCount(&num_devices));

	printf("> GPU num: %d.\n", num_devices);

	int cur_device = 0;
	//select correct devices
	checkCudaErrors(cudaSetDevice(cur_device));
	struct cudaDeviceProp device_prop;
	checkCudaErrors(cudaGetDeviceProperties(&device_prop, cur_device));
	printf("> ID:%d %s.\n", cur_device, device_prop.name);

	if (!device_prop.deviceOverlap){
		printf("> Device will not handle overlaps, so no speed up from stream\n");
	}
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	u32 const nstream = 2;
	cudaStream_t stream[nstream];
	for (u32 str = 0; str < nstream; str++)
		checkCudaErrors(cudaStreamCreate(&stream[str]));

	u32 *h_src_arr0, *h_src_arr1, *h_dst_arr;
	u32 *d_src_arr0[nstream], *d_src_arr1[nstream], *d_dst_arr[nstream];

	const u32 threadsPerBlock = 512;
	const u32 blocksPerGrid = (NUM_ELEM + threadsPerBlock - 1) / threadsPerBlock;

	const int shared_memory_usage = 0;
	const size_t Hsize = sizeof(u32)*FULL_DATA_SIZE;
	const size_t Dsize = sizeof(u32)*NUM_ELEM;

	checkCudaErrors(cudaHostAlloc(&h_src_arr0, Hsize, cudaHostAllocDefault));
	checkCudaErrors(cudaHostAlloc(&h_src_arr1, Hsize, cudaHostAllocDefault));
	checkCudaErrors(cudaHostAlloc(&h_dst_arr, Hsize, cudaHostAllocDefault));

	for (int str = 0; str < nstream; str++){
		checkCudaErrors(cudaMalloc(&d_src_arr0[str], Dsize));
		checkCudaErrors(cudaMalloc(&d_src_arr1[str], Dsize));
		checkCudaErrors(cudaMalloc(&d_dst_arr[str], Dsize));
	}

	fillHostArray(h_src_arr0, FULL_DATA_SIZE);
	fillHostArray(h_src_arr1, FULL_DATA_SIZE);

	checkCudaErrors(cudaEventRecord(start, 0));

	for (int i = 0; i < FULL_DATA_SIZE; i += NUM_ELEM * nstream){

		for (int str = 0; str < nstream; str++){
			checkCudaErrors(cudaMemcpyAsync(d_src_arr0[str], h_src_arr0 + i + NUM_ELEM*str, Dsize, cudaMemcpyHostToDevice, stream[str]));
			checkCudaErrors(cudaMemcpyAsync(d_src_arr1[str], h_src_arr1 + i + NUM_ELEM*str, Dsize, cudaMemcpyHostToDevice, stream[str]));

			gpuKernel_1 << <blocksPerGrid, threadsPerBlock, shared_memory_usage, stream[str] >> >
				(d_src_arr0[str], d_src_arr1[str], d_dst_arr[str]);

			checkCudaErrors(cudaMemcpyAsync(h_dst_arr + i + NUM_ELEM*str, d_dst_arr[str], Dsize, cudaMemcpyDeviceToHost, stream[str]));
		}
	}

	//for (int i = 0; i < FULL_DATA_SIZE; i += NUM_ELEM * nstream){

	//	checkCudaErrors(cudaMemcpyAsync(d_src_arr0[0], h_src_arr0 + i, Dsize, cudaMemcpyHostToDevice, stream[0]));
	//	checkCudaErrors(cudaMemcpyAsync(d_src_arr1[0], h_src_arr1 + i, Dsize, cudaMemcpyHostToDevice, stream[0]));

	//	checkCudaErrors(cudaMemcpyAsync(d_src_arr0[1], h_src_arr0 + i + NUM_ELEM, Dsize, cudaMemcpyHostToDevice, stream[1]));
	//	checkCudaErrors(cudaMemcpyAsync(d_src_arr1[1], h_src_arr1 + i + NUM_ELEM, Dsize, cudaMemcpyHostToDevice, stream[1]));

	//	gpuKernel_1 << <blocksPerGrid, threadsPerBlock, shared_memory_usage, stream[0] >> >
	//		(d_src_arr0[0], d_src_arr1[0], d_dst_arr[0]);

	//	checkCudaErrors(cudaMemcpyAsync(h_dst_arr + i, d_dst_arr[0], Dsize, cudaMemcpyDeviceToHost, stream[0]));

	//	gpuKernel_1 << <blocksPerGrid, threadsPerBlock, shared_memory_usage, stream[1] >> >
	//		(d_src_arr0[1], d_src_arr1[1], d_dest_arr[1]);

	//	checkCudaErrors(cudaMemcpyAsync(h_dst_arr + i + NUM_ELEM, d_dst_arr[1], Dsize, cudaMemcpyDeviceToHost, stream[1]));

	//}
	for (int str = 0; str < nstream; str++)
		checkCudaErrors(cudaStreamSynchronize(stream[str]));
	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));

	float elapsedTime = 0.f;
	checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
	printf("> Time taken: %3.1f ms\n", elapsedTime);

	checkCudaErrors(cudaFreeHost(h_src_arr0));
	checkCudaErrors(cudaFreeHost(h_src_arr1));
	checkCudaErrors(cudaFreeHost(h_dst_arr));
	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(stop));

	for (int str = 0; str < nstream; str++){
		checkCudaErrors(cudaStreamDestroy(stream[str]));
		checkCudaErrors(cudaFree(d_src_arr0[str]));
		checkCudaErrors(cudaFree(d_src_arr1[str]));
		checkCudaErrors(cudaFree(d_dst_arr[str]));
	}

}


//int main(int argc, char **argv)
//{
//	printf("> %s Starting...\n\n", argv[0]);
//	srand((unsigned)time(NULL));
//
//	_multipleStreamTest();
//
//	exit(EXIT_SUCCESS);
//}