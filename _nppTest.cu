//#include"cuda_helper.cuh"
//
//#include"npp.h"
//#include"nppcore.h"
//#include"nppdefs.h"
//#include"nppi.h"
//#include"npps.h"
//#include"nppversion.h"
//#define NPP_CALL(x){const NppStatus a=(x);if (a!=NPP_SUCCESS){printf("\nNPP Error(err_num=%d) \n", a);cudaDeviceReset();ASSERT(0);}}
//
//#define NUM_STREAMS  4
//typedef unsigned char u8;
//
//#include <npp.h>
//#pragma comment(lib, "cudart.lib")
//#pragma comment(lib, "nppi.lib")
//
//template<class T>
//__host__ void fillHostArray(T* data, const u32 num_elements){
//	for (u32 i = 0; i < num_elements; i++)
//		data[i] = rand() % (num_elements + 1);
//}
//void _nppTest(){
//
//	checkCudaErrors(cudaSetDevice(0));
//	const u32 num_bytes = (1024u * 255u*256)*sizeof(Npp8u);
//
//	Npp8u *host_src_ptr0;
//	Npp8u *host_src_ptr1;
//	Npp8u *host_dst_ptr0[NUM_STREAMS];
//
//
//	Npp8u *device_src_ptr0[NUM_STREAMS];
//	Npp8u *device_src_ptr1[NUM_STREAMS];
//	Npp8u *device_dst_ptr0[NUM_STREAMS];
//	
//
//	checkCudaErrors(cudaHostAlloc(&host_src_ptr0,num_bytes,cudaHostAllocDefault));
//	checkCudaErrors(cudaHostAlloc(&host_src_ptr1, num_bytes, cudaHostAllocDefault));
//	
//
//	cudaStream_t stream[NUM_STREAMS];
//	cudaEvent_t start, stop;
//	checkCudaErrors(cudaEventCreate(&start));
//	checkCudaErrors(cudaEventCreate(&stop));
//	for (u32 str = 0; str < NUM_STREAMS; str++){
//		checkCudaErrors(cudaHostAlloc(&host_dst_ptr0[str], num_bytes, cudaHostAllocDefault));
//		checkCudaErrors(cudaMalloc(&device_src_ptr0[str],num_bytes));
//		checkCudaErrors(cudaMalloc(&device_src_ptr1[str], num_bytes));
//		checkCudaErrors(cudaMalloc(&device_dst_ptr0[str], num_bytes));
//		checkCudaErrors(cudaStreamCreate(&stream[str]));
//	
//	}
//	fillHostArray(host_src_ptr0, num_bytes);
//	fillHostArray(host_src_ptr1, num_bytes);
//
//	checkCudaErrors(cudaEventRecord(start,0));
//
//	for (u32 str = 0; str < NUM_STREAMS; str++){
//	
//		nppSetStream(stream[str]);
//		checkCudaErrors(cudaMemcpyAsync(device_src_ptr0[str], host_src_ptr0,num_bytes,cudaMemcpyHostToDevice,stream[str]));
//		checkCudaErrors(cudaMemcpyAsync(device_src_ptr1[str], host_src_ptr1, num_bytes, cudaMemcpyHostToDevice, stream[str]));
//		nppsXor_8u(device_src_ptr0[str], device_src_ptr1[str], device_dst_ptr0[str],num_bytes);
//
//	}
//	for (u32 str = 0; str < NUM_STREAMS; str++){
//		nppSetStream(stream[str]);
//		checkCudaErrors(cudaMemcpyAsync(host_dst_ptr0[str], device_dst_ptr0[str], num_bytes,cudaMemcpyDeviceToHost,stream[str]));
//		checkCudaErrors(cudaStreamSynchronize(stream[str]));
//	}
//	checkCudaErrors(cudaEventRecord(stop, 0));
//
//	float elapasedTime = 0.f;
//	checkCudaErrors(cudaEventElapsedTime(&elapasedTime,start,stop));
//	printf("> Execute time:%3.1f\n", elapasedTime);
//	for (u32 str = 0; str < NUM_STREAMS; str++){
//		checkCudaErrors(cudaStreamDestroy(stream[str]));
//		checkCudaErrors(cudaFree(device_src_ptr0[str]));
//		checkCudaErrors(cudaFree(device_src_ptr1[str]));
//		checkCudaErrors(cudaFree(device_dst_ptr0[str]));
//		checkCudaErrors(cudaFreeHost(host_dst_ptr0[str]));
//	}
//	checkCudaErrors(cudaFreeHost(host_src_ptr0));
//	checkCudaErrors(cudaFreeHost(host_src_ptr1));
//
//}
//
//int main(int argc, char **argv){
//	printf("> %s Starting...\n\n", argv[0]);
//	srand((unsigned)time(NULL));
//	_nppTest();
//	checkCudaErrors(cudaDeviceReset());
//	EXIT_SUCCESS;
//};