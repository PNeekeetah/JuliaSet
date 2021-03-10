#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/complex.h>
#include <fstream>

void printMatrixElements(int **matrix,int size_x,int size_y) {
	for (int i = 0; i < size_x; i++) {
		for (int j = 0; j < size_y; j++) {
			std::cout << matrix[i][j] << "\t";
		}
		std::cout << "\n";
	}
	std::cout << std::endl;

}

int** allocateSquareMatrix(int size) {
	int** matrix = (int**)malloc(size * sizeof(int*));
	for (int i = 0; i < size; i++) {
		matrix[i] = (int*)malloc(size * sizeof(int));
	}
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			matrix[i][j] = 1;
		}
	}
	return matrix;
}

void freeSpace(int** mat, int size) {
	for (int i = 0; i < size; i++) {
		free(mat[i]);
	}
	free(mat);
}
// Well, this is pretty much it. All I have to do is include the calculation here in Matrix Assignment and the program to calculate the
// Mandelbrot set is done
__global__
void MatrixAssignment(int* deviceMatrix, size_t pitch, int size_x, int size_y) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	// printf("x = %d, y = %d \n", x, y);
	if ((x < size_x) && (y < size_y)) {
		int* row = (int*)((char*)deviceMatrix + pitch * y);
		row[x] = y * x;
	}
}

__global__
void MandelbrotGenerator(int* deviceMatrix, size_t pitch, int size_x, int size_y) {
	int dx = blockDim.x * blockIdx.x + threadIdx.x;
	int dy = blockDim.y * blockIdx.y + threadIdx.y;
	
	if ((dx < size_x) && (dy < size_y)) {
		int* row = (int*)((char*)deviceMatrix + pitch * dy);
		int iterations = 255;
		float x = float(dx) / float(size_x / 4) - 2.0;
		float y = float(dy) / float(size_y / 4) - 2.0;
		//printf("x = %f, y = %f \n", x, y);
		thrust::complex<float> exp = thrust::complex<float>(2.0, 0.0);
		thrust::complex<float> m = thrust::complex<float>(x,y);
		thrust::complex<float> z = thrust::complex<float>(0.0,0.0);
		while ((iterations > 0) && (thrust::abs(z) < thrust::abs(exp))) {
			z = thrust::pow(z, exp) + m;
			iterations -= 1;
		}
		if (thrust::abs(z) < thrust::abs(exp) ) {
			row[dx] = iterations;
		}
		else {
			row[dx] = iterations;
		}
	}
}

int main() {
	int size = 16000;
	cudaError flag = cudaSuccess;
	
	int* bigHostMatrix = (int*)malloc(size * size * sizeof(int)); // contiguous
	int** hostMatrix = (int**)malloc(size*sizeof(int*));
	for (int i = 0; i < size; i++) {
		hostMatrix[i] = &bigHostMatrix[i * size];
	}
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			hostMatrix[i][j] = 0;
		}
	}

//	printMatrixElements(hostMatrix, size, size);
	// for 4 by 4, I can do 2,2 blocks and 2,2 threads.
	// for 8 by 8, I likely need 4,4 blocks and 4,4 threads
	/*
	thrust::complex<float> z = thrust::complex<float>(3, 4);
	thrust::complex<float> exp = thrust::complex<float>(2, 0);
	std::cout << thrust::abs(z) << std::endl;
	z = thrust::pow(z, exp);
	std::cout << thrust::pow(z, exp) << std::endl;
	std::cout << thrust::abs(z) << std::endl;
	*/
	

	int* deviceMatrix = NULL;
	size_t devicePitch;
	flag = cudaMallocPitch(&deviceMatrix, &devicePitch, size * sizeof(int), size);
	flag = cudaMemcpy2D(deviceMatrix, devicePitch, bigHostMatrix , size * sizeof(int), size * sizeof(int), size, cudaMemcpyHostToDevice);
	dim3 threads = dim3(256 / 256);
	dim3 blocks = dim3(size/threads.x, size/threads.y);
	
	MandelbrotGenerator << <blocks, threads >> > (deviceMatrix, devicePitch, size, size);
	cudaDeviceSynchronize();
	flag = cudaMemcpy2D(bigHostMatrix, size * sizeof(int), deviceMatrix, devicePitch, size * sizeof(int), size, cudaMemcpyDeviceToHost);
	//printMatrixElements(hostMatrix, size, size);
	
	char file_name[100] = "";
	snprintf(file_name, 100 * sizeof(char), "%d_x_%d_image.ppm", size, size);
	std::ofstream image_file(file_name);
	image_file << "P6" << std::endl;
	image_file << size << " " << size << std::endl;
	image_file << "255" << std::endl;

	for (int j = 0; j < size; j++) {
		for (int i = 0; i < size; i++) {
			image_file << uint8_t(hostMatrix[j][i]) << uint8_t(hostMatrix[j][i]) << uint8_t(hostMatrix[j][i]);
		}
	}

	image_file.close();	
	cudaFree(deviceMatrix);
	free(hostMatrix);
	free(bigHostMatrix);
	std::cout << "End";
}