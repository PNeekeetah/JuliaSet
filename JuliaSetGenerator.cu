#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/complex.h>
#include <fstream>
#include <chrono>

template <typename T = uint8_t>  
struct Contiguous2DArray 
{
	int size_x;
	int size_y;
	T* linear = NULL;
	T** matrix = NULL;
};

enum ContiguousAllocationStatus{ 
	LINEAR_ALLOCATION_FAIL, 
	MATRIX_ALLOCATION_FAIL, 
	CONTIGUOUS_ALLOCATION_SUCCESS,
	ALREADY_ALLOCATED
};

/*****************************
*ALLOCATE CONTIGUOUS 2D ARRAY*
*****************************/
template <typename T>
ContiguousAllocationStatus allocateContiguous2DArray(Contiguous2DArray<T> &arr2D,int size_x, int size_y) {
	if ((arr2D.linear != NULL)  || (arr2D.matrix != NULL)) {
		return ALREADY_ALLOCATED;
	}
	arr2D.size_x = size_x;
	arr2D.size_y = size_y;
	arr2D.linear = (T*)malloc(size_x * size_y * sizeof(T));
	if (!arr2D.linear) {
		return LINEAR_ALLOCATION_FAIL;
	}
	arr2D.matrix = (T**)malloc(size_x * sizeof(T*));
	if (!arr2D.matrix) {
		return MATRIX_ALLOCATION_FAIL;
	}
	for (int x = 0; x < size_x; x++) {
		arr2D.matrix[x] = &arr2D.linear[x * size_y];
	}
	return CONTIGUOUS_ALLOCATION_SUCCESS;
}

/***********************************
*CHECK CONTIGUOUS ALLOCATION STATUS*
***********************************/
void checkAllocation(ContiguousAllocationStatus status) {
	if (status == ALREADY_ALLOCATED) {
		std::cout << "There are elements into the contiguous allocation. Consider freeing." << "\n";
	}
	else if (status == LINEAR_ALLOCATION_FAIL) {
		std::cout << "Cannot allocate the linear contiguous array. Not enough space." << "\n";
	}
	else if (status == MATRIX_ALLOCATION_FAIL) {
		std::cout << "Cannot allocate the bidimensional array. Not enough space." << "\n";
	}
	else {
		std::cout << "Contiguous allocation succesful." << "\n";
	}
}

/****************************
*DEALLOCATE CONTIGUOUS SPACE*
****************************/
template <typename T>
void deallocateContiguous2DArray(Contiguous2DArray<T> arr2D) {
	free(arr2D.matrix);
	free(arr2D.linear);
	arr2D.matrix = NULL;
	arr2D.linear = NULL;
	arr2D.size_x = 0;
	arr2D.size_y = 0;
}

/*************************
*PRINT 2D MATRIX ELEMENTS*
**************************/
template<typename T>
void printMatrixElements(T** matrix, int size_x, int size_y) {
	for (int i = 0; i < size_x; i++) {
		for (int j = 0; j < size_y; j++) {
			std::cout << matrix[i][j] << "\t";
		}
		std::cout << "\n";
	}
	std::cout << std::endl;
}

/****************************
*CHECK CUDA FUNCTIONS STATUS*
****************************/
void gpuErrCheck(cudaError state, std::string message) {
	if (state != cudaSuccess) {
		std::cout << "Failed at : " << message << "\n";
	}
	else {
		std::cout << message << " - Success" << "\n";
	}
}

/**************************
*GPU MANDELBROT GENERATION*
***************************/
template<typename T>
__global__
void gpuGenerateMandelbrot(T* deviceMatrix, size_t pitch, int size_x, int size_y, thrust::complex<float> j, int power = 2) {
	int dx = blockDim.x * blockIdx.x + threadIdx.x;
	int dy = blockDim.y * blockIdx.y + threadIdx.y;
	if ((dx < size_x) && (dy < size_y)) {
		T* row = (T*)((char*)deviceMatrix + pitch * dy);
		T iterations = 255;
		float x = float(dx) * 4.0f / (float(size_x) - 1.0f) - 2.0f;						// x and y are put in range [-2,2]
		float y = float(dy) * 4.0f / (float(size_y) - 1.0f) - 2.0f;
		thrust::complex<float> m = thrust::complex<float>(x, y);
		thrust::complex<float> z = thrust::complex<float>(0.0f, 0.0f);
		thrust::complex<float> l_z = z;
		if (j != thrust::complex<float>(2.0f, 2.0f)) {									// Asking for a Julia Set
			z = m;
			m = j;
		}
		while ((iterations > 0) && (thrust::abs(z) < 4)) {
			l_z = z;
			z = thrust::pow(z, power) + m;
			iterations -= 1;
			if (fabs(thrust::abs(z - l_z)) < 0.01 * fabs(thrust::abs(l_z))) {			//Unlikely it diverges from 0
				iterations = 0;
			}
		}
		row[dx] = iterations;
	}
}

/**************************
*CPU MANDELBROT GENERATION*
**************************/
template <typename T>
void cpuGenerateMandelbrot(T **hostMatrix,int size_x, int size_y, thrust::complex<float> j, int power = 2) {
	for (int dx = 0; dx < size_x; dx++) {
		for (int dy = 0; dy < size_y; dy++) {
			int iterations = 255;
			float x = float(dx) * 4.0f / (float(size_x) - 1.0f) - 2.0f;						// x and y are put in range [-2,2]
			float y = float(dy) * 4.0f / (float(size_y) - 1.0f) - 2.0f;
			thrust::complex<float> m = thrust::complex<float>(x, y);
			thrust::complex<float> z = thrust::complex<float>(0.0f, 0.0f);
			thrust::complex<float> l_z = z;
			if (j != thrust::complex<float>(2.0f, 2.0f)){									// Asking for a Julia Set
				z = m;
				m = j;
			}

			while ((iterations > 0) && (thrust::abs(z) < 4)) {
				l_z = z;
				z = thrust::pow(z, power) + m;
				iterations -= 1;
				if (fabs(thrust::abs(z - l_z)) < 0.01 * fabs(thrust::abs(l_z))) {			//Unlikely it diverges from 0
					iterations = 0;
				}
			}
			hostMatrix[dy][dx] = iterations;
		}
	}
}

/*********************
*WRITE TO FILE HELPER*
*********************/
template <typename T>
void writeImage(Contiguous2DArray<T>& arr2D, std::string additionalTitle = "") {

	int name_size = 300;
	char* file_name = (char*)malloc(name_size * sizeof(char));
	std::cout << arr2D.size_x << " " << arr2D.size_y << "\n";
	snprintf(file_name, name_size * sizeof(char), "%d_x_%d_%s_image.ppm", arr2D.size_x, arr2D.size_y, additionalTitle.data());

	auto image_file = std::fstream(file_name, std::ios::out | std::ios::binary | std::fstream::trunc);
	image_file << "P5" << "\t" << arr2D.size_x << "\t" << arr2D.size_y << "\t" << "255" << "\t";						// P5 header
	image_file.write((char*)&arr2D.linear[0], arr2D.size_x * arr2D.size_y * sizeof(T));									// Data
	image_file.close();
	std::cout << "Written " << float(arr2D.size_x * arr2D.size_y * sizeof(T)) / pow(2.0f, 20) << " MB of data to " << file_name << "\n";
}

/*****
*MAIN*
*****/
int main() {
	Contiguous2DArray<uint8_t> host;
	uint8_t* deviceMatrix = NULL;
	size_t devicePitch;
	volatile int size = 8192*4;
	volatile bool time = true;
	volatile bool useGPU = true;
	volatile int ts = 8;
	volatile int pwr = 4;
	thrust::complex<float> startingPoint = thrust::complex<float>(0.28f, .008f);
	auto start = std::chrono::high_resolution_clock::now();
	std::string additionalTitle = "";

	checkAllocation(allocateContiguous2DArray<uint8_t>(host, size, size));

	if (useGPU){
		gpuErrCheck(cudaMallocPitch(&deviceMatrix, &devicePitch, size * sizeof(uint8_t), size),"Allocate Pitch");
		gpuErrCheck(cudaMemcpy2D(deviceMatrix, devicePitch, host.linear , size * sizeof(uint8_t), size * sizeof(uint8_t), size, cudaMemcpyHostToDevice),"Host to Device Copy");
	
		dim3 threads = dim3(ts,ts);
		dim3 blocks = dim3(size/threads.x, size/threads.y);
	
		gpuGenerateMandelbrot << <blocks, threads >> > (deviceMatrix, devicePitch, size, size, startingPoint, pwr);
		gpuErrCheck(cudaDeviceSynchronize(),"Synchronization");
		gpuErrCheck(cudaMemcpy2D(host.linear, size * sizeof(uint8_t), deviceMatrix, devicePitch, size * sizeof(uint8_t), size, cudaMemcpyDeviceToHost),"Device to Host Copy");
	}
	else {
		cpuGenerateMandelbrot(host.matrix, size, size, startingPoint ,pwr);
	}
	writeImage(host, additionalTitle);
	
	if (time) {
		auto elapsed = std::chrono::high_resolution_clock::now() - start;
		long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
		int seconds = microseconds / pow(10, 6);
		std::cout << seconds << " seconds have elapsed. \n";
	}
	cudaFree(deviceMatrix);
	deallocateContiguous2DArray(host);
	std::cout << "End.\n";
	return 0;
}

