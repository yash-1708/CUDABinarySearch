#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <stdio.h>
#include <ctime>
#define P 128

//main kernel that runs on GPU
__global__ void BinSearch(int *mainVec, int *toSearch, bool *isPresent, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int key = toSearch[i];
	bool flag = false;
	int min = 0;
	int max = N;
	int mid = (min + max) / 2;
	while (min != mid) {
		if (mainVec[mid] == key) {
			flag = true;
			break;
		}
		else if (key < mainVec[mid]){
			max = mid;
			min = min;
		}
		else {
			min = mid;
			max = max;
		}
		mid = (min + max) / 2;
	}
	isPresent[i] = flag;
}

bool serial_BinSearch(int* mainVec, int* toSearch, int i, int N) {
	int key = toSearch[i];
	bool flag = false;
	int min = 0;
	int max = N;
	int mid = (min + max) / 2;
	while (min != mid) {
		if (mainVec[mid] == key) {
			flag = true;
			break;
		}
		else if (key < mainVec[mid]) {
			max = mid;
			min = min;
		}
		else {
			min = mid;
			max = max;
		}
		mid = (min + max) / 2;
	}
	return flag;
}


//random vector generator
void randomInts(int* a, int N) {
	for (int i = 0; i < N; i++){
		a[i] = rand() % 100;
	}
}

int main() {
	clock_t start, stop;

	int N = 100;//number of elements in main vector
	int M = 123400;//number of elements to be searched
	
	int* mainVec = (int*)malloc(N * sizeof(int));
	int* toSearch = (int*)malloc(M * sizeof(int));
	bool* isPresent = (bool*)malloc(M * sizeof(bool));
	bool* serial_isPresent = (bool*)malloc(M * sizeof(bool));

	//randomInts(mainVec, N);
	for (size_t i = 0; i < N; i++){
		mainVec[i] = i;
	}
	randomInts(toSearch, M);

	/*printf("\n\nMain Vector : \n");
	for (size_t i = 0; i < N; i++)
	{
		printf("%d \n", mainVec[i]);
	}
	
	printf("\n\nTo Search Vector : \n");
	for (size_t i = 0; i < M; i++)
	{
		printf("%d \n", toSearch[i]);
	}*/

	int* d_mainVec;
	int* d_toSearch;
	bool* d_isPresent;

	cudaMalloc(&d_mainVec, N * sizeof(int));
	cudaMalloc(&d_toSearch, M * sizeof(int));
	cudaMalloc(&d_isPresent, M * sizeof(bool));

	cudaMemcpy(d_mainVec, mainVec, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_toSearch, toSearch, M * sizeof(int), cudaMemcpyHostToDevice);

	start = std::clock();
	BinSearch<<<(P+M-1)/P,M>>>(d_mainVec, d_toSearch, d_isPresent, N);
	cudaDeviceSynchronize();
	stop = std::clock();
	long float timeP = stop - start;

	cudaMemcpy(isPresent, d_isPresent, M * sizeof(bool), cudaMemcpyDeviceToHost);

	printf("\n\n\n%3.3f", timeP);
	/*printf("\n************************PARALLEL*****************************\n");
	for (size_t i = 0; i < M; i++){
		if (isPresent[i] == true) {
			printf("\n isPresent[%d] : true", i);
		}
		else {
			printf("\n isPresent[%d] : false", i);
		}
	}
	printf("\n*************************************************************\n");*/

	start = std::clock();
	for (size_t i = 0; i < M; i++){
		serial_isPresent[i] = serial_BinSearch(mainVec,toSearch,i,N);
	}
	stop = std::clock();
	long float timeN = stop - start;
	printf("\n\n\n%3.3f", timeN);
	/*
	printf("\n*****************************SERIES**************************\n");
	for (size_t i = 0; i < M; i++) {
		if (serial_isPresent[i] == true) {
			printf("\n isPresent[%d] : true", i);
		}
		else {
			printf("\n isPresent[%d] : false", i);
		}
	}
	printf("\n*************************************************************\n");*/

	//getting GPU properties and storing in prop
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	int cores = prop.multiProcessorCount * 128;
	float totalCost = cores * timeP;

	//results printing
	printf("\n***********************************************************************************************************************\n");
	printf("M \t\t\t Nor Time \t Par Time \t Cores \t\t Tot Cost \t Speedup \t Efficiency \n");
	printf("%-20d \t %-7.3f \t %-7.3f \t %-10d \t %-7.3f \t %-7.3f \t %-5.5f \n", M, timeN, timeP, cores, totalCost, timeN / timeP, timeN / (timeP * cores));
	printf("\n***********************************************************************************************************************\n");
	
	//free memory
	cudaFree(d_mainVec);
	cudaFree(d_toSearch);
	cudaFree(d_isPresent);
	free(mainVec);
	free(toSearch);
	free(isPresent);
	free(serial_isPresent);

	return  0;
}