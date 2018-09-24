#pragma once
#include"cuda_helper.cuh"

#define NUM_ELEMENTS 4096
#define BlockSize    256
#define MAX_NUM_LISTS (BlockSize)

//for kernel merge_array_4
#define REDUCTION_SIZE 8
#define REDUCTION_SIZE_BIT_SHIFT 3
#define MAX_ACTIVE_REDUCTIONS ((MAX_NUM_LISTS)/REDUCTION_SIZE)


/****************************************************************************
*******************************HOST******************************************
*****************************************************************************/
// 32*N
__host__ void cpu_sort(u32 * const src_data, u32 num_elements){
	static u32 cpu_tmp_0[NUM_ELEMENTS];
	static u32 cpu_tmp_1[NUM_ELEMENTS];
	/*clock_t start_cpu, stop_cpu;
	start_cpu = clock();*/
	// first cycle is 32 bits
	for (u32 bit = 0; bit < 32; bit++){
		u32 base_cnt_0 = 0;
		u32 base_cnt_1 = 0;
		// first cycle is num_elements
		for (u32 i = 0; i < num_elements; i++){
			const u32 bit_mask = 1 << bit;
			if (src_data[i] & bit_mask > 0){
				cpu_tmp_1[base_cnt_1] = src_data[i];
				base_cnt_1++;
			}
			else{
				cpu_tmp_0[base_cnt_0] = src_data[i];
				base_cnt_0++;
			}
		}
		//copy data back to src_data first_zero_list
		for (u32 i = 0; i < base_cnt_0; i++)
			src_data[i] = cpu_tmp_0[i];
		//copy data back to src_data then_one_list
		for (u32 i = 0; i < base_cnt_1; i++)
			src_data[base_cnt_0 + i] = cpu_tmp_1[i];
	}
	//stop_cpu = clock();
	//float delta = (float)(stop_cpu - start_cpu) / CLOCKS_PER_SEC;

	//printf("radixSortCpu excute time is %0.4f ms\n", delta * 1000);

}
__host__ u32 find_min(const u32 * src_arr, u32 *const list_index,
	const u32 num_lists, const u32 num_elements_per_list){

	u32 min_val = 0xFFFFFFFF;
	u32 min_idx = 0;

	for (u32 i = 0; i < num_lists; i++){
		if (list_index[i] < num_elements_per_list){
			const u32 src_idx = i + (list_index[i] * num_lists);
			const u32 data = src_arr[src_idx];
			if (data <= min_val){
				min_val = data;
				min_idx = i;
			}
		}
	}
	list_index[min_idx]++;
	return min_val;
}
__host__ void cpu_merge(const u32 * src_arr, u32 *const dest_arr,
	const u32 num_lists, const u32 num_elements){

	const u32 num_elements_per_list = num_elements / num_lists;
	u32 list_index[NUM_ELEMENTS];
	for (u32 list = 0; list < num_lists; list++)
		list_index[list] = 0;
	for (u32 i = 0; i < num_elements; i++)
		dest_arr[i] = find_min(src_arr, list_index, num_lists, num_elements_per_list);
}
__host__ void result_compare(const u32 * gpu_arr, const u32 * cpu_arr,
	const u32 num_elements){
	bool flag = true;
	for (u32 i = 0; i < num_elements; i++){
		if (gpu_arr[i] != cpu_arr[i]){
			flag = false;
			break;
		}
	}
	if (flag == true)
		printf("> output of host and device are same, test pass.\n\n");
	else
		printf("> output of host and device are difft, test fail.\n\n");
}
/****************************************************************************
*******************************DEVICE****************************************
*****************************************************************************/
__device__ void radix_sort_0(u32 * const sort_tmp, const u32 num_lists,
	const u32 num_elements, const u32 tid, u32 * const sort_tmp_0, u32 * const sort_tmp_1){
	// Sort into num_list, lists
	// Apply radix sort on 32 bits of data
	for (u32 bit = 0; bit < 32; bit++){
		u32 base_cnt_0 = 0;
		u32 base_cnt_1 = 0;
		const u32 bit_mask = (1 << bit);
		for (u32 i = 0; i < num_elements; i += num_lists){
			const u32 elem = sort_tmp[i + tid];
			if ((elem & bit_mask) > 0){
				sort_tmp_1[base_cnt_1 + tid] = elem;
				base_cnt_1 += num_lists;
			}
			else{
				sort_tmp_0[base_cnt_0 + tid] = elem;
				base_cnt_0 += num_lists;
			}
		}
		// Copy data back to source - first the zero list
		for (u32 i = 0; i < base_cnt_0; i += num_lists)
			sort_tmp[i + tid] = sort_tmp_0[i + tid];
		// Copy data back to source - then the one list
		for (u32 i = 0; i < base_cnt_1; i += num_lists)
			sort_tmp[base_cnt_0 + i + tid] = sort_tmp_1[i + tid];
	}
	__syncthreads();
}
__device__ void radix_sort_1(u32 * const sort_tmp, u32 * const sort_tmp_1,
	const u32 num_lists, const u32 num_elements, const u32 tid){
	for (u32 bit = 0; bit < 32; bit++){
		u32 base_cnt_0 = 0;
		u32 base_cnt_1 = 0;
		u32 bit_mask = 1 << bit;
		for (u32 i = 0; i < num_elements; i += num_lists){
			const u32 elem = sort_tmp[i + tid];
			if ((elem & bit_mask) > 0){
				sort_tmp_1[base_cnt_1 + tid] = elem;
				base_cnt_1 += num_lists;
			}
			else{
				sort_tmp[base_cnt_0 + tid] = elem;
				base_cnt_0 += num_lists;
			}
		}
		for (u32 i = 0; i < base_cnt_1; i += num_lists)
			sort_tmp[i + base_cnt_0 + tid] = sort_tmp_1[i + tid];
	}
	__syncthreads();
}
__device__ void copy_data_to_shared(const u32 * src_data, u32 * const sort_tmp,//const u32 * src_data 常量指针,p可以改变,*p 不可以改变
	const u32 num_lists, const u32 num_elements, const u32 tid){
	for (u32 i = 0; i < num_elements; i += num_lists)
		sort_tmp[i + tid] = src_data[i + tid]; // 全局内存以行的形式load 到shredmemory 
	// sort_tmp 会被频繁的访问
	__syncthreads();
}
// only one thread to compute
__device__ void merge_array_1(const u32 * src_data, u32 * const dest_data,
	const u32 num_lists, const u32 num_elements, const u32 tid){
	__shared__ u32 list_index[MAX_NUM_LISTS];
	// mutiple threads
	list_index[tid] = 0;
	__syncthreads();
	//sigle thread
	if (tid == 0){
		const u32 num_elements_per_list = num_elements / num_lists;
		for (u32 i = 0; i < num_elements; i++){
			u32 min_val = 0xFFFFFFFF;
			u32 min_idx = 0;
			for (u32 list = 0; list < num_lists; list++){
				if (list_index[list] < num_elements_per_list){
					const u32 src_idx = list + (list_index[list] * num_lists);
					const u32 data = src_data[src_idx];
					if (data <= min_val){
						min_val = data;
						min_idx = list;
					}
				}
			}
			list_index[min_idx]++;
			dest_data[i] = min_val;
		}
	}
}
/******num_lists threads parallel compute, but actomicMin is just used for interger not float*/
__device__ void merge_array_2(const u32 * src_data, u32 * const dest_data,
	const u32 num_lists, const u32 num_elements, const u32 tid){
	__shared__ u32 list_index[MAX_NUM_LISTS];
	// mutiple threads
	list_index[tid] = 0;
	__syncthreads();
	//sigle thread
	const u32 num_elements_per_list = num_elements / num_lists;
	for (u32 i = 0; i < num_elements; i++){
		// create a value shared with the other threads
		__shared__ u32 min_val;
		__shared__ u32 min_tid;
		// use a tmp register for work purposes
		u32 data;
		// threadsPerBlocks= num_lists
		if (list_index[tid] < num_elements_per_list){
			const u32 src_idx = tid + (list_index[tid] * num_lists);
			data = src_data[src_idx];
		}
		else
			data = 0xFFFFFFFF;
		//have thread0 clear the min values
		if (tid == 0){
			min_val = 0xFFFFFFFF;
			min_tid = 0xFFFFFFFF;
		}
		// wait for all threads();
		__syncthreads();

		// have every thread try to store it's value into
		// min_val. Only the thread with lowest value will win
		atomicMin(&min_val, data);
		// make sure every threads have taken their turn
		__syncthreads();

		// if this thread was the one with the minimum
		if (min_val == data){
			// check for equal values
			// lowest tid wins adn does the wirte
			atomicMin(&min_tid, tid);
		}
		//make sure all threads have taken their turn
		__syncthreads();
		//if this thread has the lowest tid
		if (tid == min_tid){
			// incremen the list pointer for this thread
			list_index[tid]++;
			//stroe the winning value
			dest_data[i] = data;
		}
	}
}
// reduction slower than merge_array_2
__device__ void merge_array_3(const u32 * src_data, u32 * const dest_data,
	const u32 num_lists, const u32 num_elements, const u32 tid){

	const u32 num_elements_per_list = num_elements / num_lists;
	__shared__ u32 list_index[MAX_NUM_LISTS];
	__shared__ u32 reduction_val[MAX_NUM_LISTS];
	__shared__ u32 reduction_idx[MAX_NUM_LISTS];

	// Clear working sets
	list_index[tid] = 0;
	reduction_val[tid] = 0;
	reduction_idx[tid] = 0;
	// make sure every threads have taken their turn
	__syncthreads();

	// every time to find the min value to load the dest_data[i]
	for (u32 i = 0; i < num_elements; i++){
		// we need num_list/2 active threads
		u32 tid_max = num_lists >> 1;
		// use a tmp register for work purposes
		u32 data;
		if (list_index[tid] < num_elements_per_list){
			u32 src_id = tid + (list_index[tid] * num_lists);
			data = src_data[src_id];
		}
		// if list_index[tid]==num_elements_per_list,means the list of tid 
		// have already load to dest_data, they have wined before
		else
			data = 0xFFFFFFFF;

		// store the data and idx to reduction shared memory
		reduction_val[tid] = data;
		reduction_idx[tid] = tid;
		//wait for all threads have taken their turn
		__syncthreads();
		//time to reduction from num_lists to the thread0 for the min value
		while (tid_max != 0){
			//gradually reduce tid_max from num_list/2 to zero
			if (tid < tid_max){
				// every thread which (< tid_max) have compared the value with the other half
				// to find the more lowest value
				// calculate the index of the other half
				const u32 val2_idx = tid + tid_max;
				// read in the other half
				const u32 val2 = reduction_val[val2_idx];
				// if the half is bigger, store the smaller value and idx
				if (val2 < reduction_val[tid]){
					reduction_val[tid] = val2;
					reduction_idx[tid] = reduction_idx[val2_idx];
				}
			}
			// divide tid_max
			tid_max = tid_max/2;
			__syncthreads();
		}
		if (tid == 0){
			// increment the list pointer for this thread
			list_index[reduction_idx[0]]++;
			// store the winning value
			dest_data[i] = reduction_val[0];
		}
		// wait for tid 0
		__syncthreads();
	}
}
// add reduction and actomicMin 
// use multiple threads for merge,does reduction into a warp and then into a sigle value
// 不理解，需要继续看
__device__ void merge_array_4(const u32 * src_data, u32 * const dest_data,
	const u32 num_lists, const u32 num_elements, const u32 tid){
	//read initial value from the list(num_list=threadsPerBlock)
	u32 data = src_data[tid];
	//shared memory index
	const u32 s_idx = tid >> REDUCTION_SIZE_BIT_SHIFT;
    //calculate number of the 1st stage reduction
	const u32 num_reductions = num_lists >> REDUCTION_SIZE_BIT_SHIFT;
	const u32 num_elements_per_list = num_elements / num_lists;
	//declear a number of list pointers and set to the start of the list
	__shared__ u32 list_index[MAX_NUM_LISTS];
	list_index[tid] = 0;
    //interate over all elements
	for (u32 i = 0; i < num_elements; i++){
	// create a value shared with the other threads
		__shared__ u32 min_val[MAX_ACTIVE_REDUCTIONS];
		__shared__ u32 min_tid;
	// have one thread from warp0 clear the min_value
		if (tid < num_lists){
		// write a very large value so the first thread wins the min
			min_val[s_idx] = 0xFFFFFFFF;
			min_tid = 0xFFFFFFFF;
		}
		// wait for the warp0 to clear min vals
		__syncthreads();
	//have every thread try to store its value into min_val for it's
	//own reduction elements. only the thread with the lowest value will win
		atomicMin(&min_val[s_idx],data);

	//if we have more than one reduction then do an addition reduction step
		if (num_reductions > 0){
		// wait for all threads
			__syncthreads();
		// have every thread in warp0 do an additional min over all the partial
		// min to date
			if (tid < num_reductions)	
				atomicMin(&min_val[0],min_val[tid]);
		// make sure each thread have taken their turn
			__syncthreads();
		}
		//if this thread is the minimum
		if (min_val[0] == data){
		//check for the equal values lowest tid win 
			atomicMin(&min_tid,tid);
		}
		__syncthreads();
		if (tid == min_tid){
			list_index[tid]++;
			dest_data[i] = data;
			if (list_index[tid] < num_elements_per_list){
				data = src_data[tid + list_index[tid] * num_lists];
			}
			else
				data = 0xFFFFFFFF;
		}
		__syncthreads();
	}
	 

}
__global__ void sortGpu(u32 * const src_data, u32 num_lists, u32 num_elements){ //u32 * const src_data 指针常量 *p可以改变
	__shared__ u32 sort_tmp[NUM_ELEMENTS];
	__shared__ u32 sort_tmp_1[NUM_ELEMENTS];
	u32 tid = threadIdx.x + blockDim.x * blockIdx.x;
	copy_data_to_shared(src_data, sort_tmp, num_lists, num_elements, tid);
	radix_sort_1(sort_tmp, sort_tmp_1, num_lists, num_elements, tid);
	merge_array_3(sort_tmp, src_data, num_lists, num_elements, tid);  //src_data 被改变了
}


/****************************************************************************
*******************************TEST******************************************
*****************************************************************************/
void _sortTest(){

	/*************************Define************ ***********************/
	const u32 num_elements = NUM_ELEMENTS;
	const u32 threadsPerBlock = BlockSize;
	const u32 blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
	//define host & device
	u32 * h_idata;
	u32 * d_idata;
	u32 * h_odata;
	u32 * h_odata_gpu;
	//define CudaEvents
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	/*************************Malloc host & device ***********************/
	checkCudaErrors(cudaHostAlloc(&h_idata, sizeof(u32)*num_elements, cudaHostAllocDefault));
	checkCudaErrors(cudaHostAlloc(&h_odata, sizeof(u32)*num_elements, cudaHostAllocDefault));
	checkCudaErrors(cudaHostAlloc(&h_odata_gpu, sizeof(u32)*num_elements, cudaHostAllocDefault));
	checkCudaErrors(cudaMalloc(&d_idata, sizeof(u32)*num_elements));

	/*************************Initialization input data***********************/
	for (u32 i = 0; i < num_elements; i++)
		h_idata[i] = rand() % (num_elements + 1);
	/*********************************H2D***************************************/
	checkCudaErrors(cudaMemcpy(d_idata, h_idata, sizeof(u32)*num_elements, cudaMemcpyHostToDevice));

	/*******************************CPU EXECUTE***********************************/
	clock_t start_cpu, end_cpu;
	start_cpu = clock();
	cpu_sort(h_idata, num_elements);
	cpu_merge(h_idata, h_odata, num_elements, num_elements);
	end_cpu = clock();
	double delta = (double)(end_cpu - start_cpu) / CLOCKS_PER_SEC;
	printf("> cpu excute time:%0.4f ms\n\n", delta * 1000);

	/*******************************GPU EXECUTE***********************************/
	checkCudaErrors(cudaEventRecord(start, 0));
	sortGpu << < blocksPerGrid, threadsPerBlock >> > (d_idata, threadsPerBlock, num_elements);
	// Compute and print the performance
	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	float elapsedTime_0 = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&elapsedTime_0, start, stop));
	printf("> radixSortGpu Kernel excute time:%0.4fms\n\n", elapsedTime_0);

	/*******************************D2H******************************************/
	checkCudaErrors(cudaMemcpy(h_odata_gpu, d_idata, sizeof(u32)*num_elements, cudaMemcpyDeviceToHost));
	result_compare(h_odata_gpu, h_odata, num_elements);

	/*******************************Memory Free***********************************/
	checkCudaErrors(cudaFreeHost(h_idata));
	checkCudaErrors(cudaFreeHost(h_odata));
	checkCudaErrors(cudaFreeHost(h_odata_gpu));
	checkCudaErrors(cudaFree(d_idata));
	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(stop));
	cudaDeviceReset();
}
//int main(int argc, char **argv)
//{
//
//	printf("%s Starting...\n\n", argv[0]);
//
//	int deviceCount = 0;
//	checkCudaErrors(cudaGetDeviceCount(&deviceCount));
//
//	if (deviceCount == 0)
//		printf("> There are no available device(s) that support CUDA\n\n");
//	else
//		printf("> Detected %d CUDA Capable device(s)\n\n", deviceCount);
//
//	checkCudaErrors(cudaSetDevice(0));
//	struct cudaDeviceProp device_prop;
//	checkCudaErrors(cudaGetDeviceProperties(&device_prop,0));
//
//	//printf("\n shared memory:%u\n",device_prop.sharedMemPerBlock);
//	//cudaDeviceSetCacheConfig(cudaFuncCachePreferL1); //L1
//	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
//
//	srand((unsigned)time(NULL));
//
//	_sortTest();
//
//	// finish
//	exit(EXIT_SUCCESS);
//}