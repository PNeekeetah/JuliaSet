#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/complex.h>
#include <fstream>
#include <chrono>
#include "CImg.h"

#define RADIUS_CONSTANT 0.0078125f*1.5f
#define FONT_CONSTANT 0.015625f*1.5f

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
void deallocateContiguous2DArray(Contiguous2DArray<T> &arr2D) {
	if (arr2D.matrix == NULL && arr2D.linear == NULL) {
		std::cout << "Nothing to deallocate! ";
		return;
	}
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
void gpuGenerateMandelbrot(T* deviceMatrix, size_t pitch, int size_x, int size_y, thrust::complex<float> j, int power = 2, int iter = 255) {
	int dx = blockDim.x * blockIdx.x + threadIdx.x;
	int dy = blockDim.y * blockIdx.y + threadIdx.y;
	if ((dx < size_x) && (dy < size_y)) {
		T* row = (T*)((char*)deviceMatrix + pitch * dy);
		T iterations = T(iter);
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
void cpuGenerateMandelbrot(T **hostMatrix,int size_x, int size_y, thrust::complex<float> j, int power = 2, int iter = 255) {
	for (int dx = 0; dx < size_x; dx++) {
		for (int dy = 0; dy < size_y; dy++) {
			T iterations = T(iter);
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

/***************
*GENERATE IMAGE*
***************/
template <typename T>
void generateImage(Contiguous2DArray<T> &mainBuff, thrust::complex<float> startPoint ,int size, bool time = true, bool useGPU = true, bool saveFile = false, std::string additionalTitle = "", int threadsNo = 8, int power = 2, int iterations = 255 ) {
	uint8_t* deviceMatrix = NULL;
	size_t devicePitch;
	
	auto start = std::chrono::high_resolution_clock::now();
	checkAllocation(allocateContiguous2DArray<uint8_t>(mainBuff, size, size));
	if (useGPU) {
		gpuErrCheck(cudaMallocPitch(&deviceMatrix, &devicePitch, size * sizeof(uint8_t), size), "Allocate Pitch");
		gpuErrCheck(cudaMemcpy2D(deviceMatrix, devicePitch, mainBuff.linear, size * sizeof(uint8_t), size * sizeof(uint8_t), size, cudaMemcpyHostToDevice), "Host to Device Copy");

		dim3 threads = dim3(threadsNo, threadsNo);
		dim3 blocks = dim3(size / threads.x, size / threads.y);

		gpuGenerateMandelbrot << <blocks, threads >> > (deviceMatrix, devicePitch, size, size, startPoint, power,iterations);
		gpuErrCheck(cudaDeviceSynchronize(), "Synchronization");
		gpuErrCheck(cudaMemcpy2D(mainBuff.linear, size * sizeof(uint8_t), deviceMatrix, devicePitch, size * sizeof(uint8_t), size, cudaMemcpyDeviceToHost), "Device to Host Copy");
		gpuErrCheck(cudaFree(deviceMatrix), "Free GPU Memory");
	}
	else {
		cpuGenerateMandelbrot(mainBuff.matrix, size, size, startPoint, power, iterations);
	}
	if (saveFile) {
		writeImage(mainBuff, additionalTitle);
	}
	if (time) {
		auto elapsed = std::chrono::high_resolution_clock::now() - start;
		long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
		int seconds = microseconds / pow(10, 6);
		std::cout << seconds << " seconds have elapsed. \n";
	}
}

/*********************
*RANGE TRANSFORMATION*
*********************/
float changeRange(int point, int old_range_min, int old_range_max, int new_range_min, int new_range_max) {
	if ((old_range_max - old_range_min - 1) == 0) {
		return -9999999.0f;
	}
	return float(point) * float(new_range_max - new_range_min) / float(old_range_max-old_range_min-1) + float(new_range_min);
}

/***************
*BOOLEAN PARSER*
***************/
volatile bool parseBool(std::string message) {
	std::cout << message << "\n";
	int v = 0;
	std::string value;
	bool flag = false;
	while (!flag) {
		std::cin >> value;
		try {
			v = std::stoi(value);
			if (v == 0) {
				return false;
			}
			else if (v == 1) {
				return true;
			}
			else {
				printf("Please enter either 0 for 'false' or 1 for 'true'.\n");
			}
		}
		catch (std::exception e) {
			printf("%s is not a number. You must enter 0 for 'false' or 1 for 'true'! \n",value.data());
		}
	}
}

/***************
*INTEGER PARSER*
***************/
volatile int parseInt(std::string message, int lower_limit, int upper_limit) {
	std::cout << message << "\n";
	int v = 0;
	std::string value;
	bool flag = false;
	while (!flag) {
		std::cin >> value;
		try {
			v = std::stoi(value);
			if (lower_limit <= v && v <= upper_limit){
				flag = true;
			}
			else {
				printf("Upper limit is %d and lower limit is %d. You entered %d.\n", upper_limit, lower_limit,v);
			}
		}
		catch (std::exception e) {
			printf("%s is not a number! \n", value.data());
		}
	}
	return v;
}

/*****
*MAIN*
*****/
int main() {
	Contiguous2DArray<uint8_t> mandelbrot;
	Contiguous2DArray<uint8_t> julia;
	volatile int size = 8192 /16;
	volatile bool time = true;
	volatile bool useGPU = true;
	volatile int ts = 8;
	volatile int pwr = 2;
	volatile bool saveFile = false;
	volatile bool animate = false;
	volatile bool expertMode = false;
	volatile bool defaultGPU = true;
	expertMode = parseBool("Would you like to use expert mode?\n"
						   "Expert mode gives you access to threads, image size, animations and more!\n"
						   "You can also choose to run the program with the defaults (recommended for first run).\n"
						   "Enter 0 for FALSE and 1 for TRUE, then hit ENTER to proceed.");
	if(expertMode){
		size = parseInt("\nEnter the size of the picture.\n" 
						"2 40,960 by 40,960 pictures take up 16 GB of RAM, keep that in mind.\n"
					    "Nonetheless, limit is set to 8192*2 on both the X and Y axis.\n"
						"This determines both the X and Y dimensions.\n"
						"Bigger pictures work better with GPU acceleration." , 1, 8192 * 2);
		pwr = parseInt("\nEnter the power for the recurrence relationship.\n"
					   "For a power of 2, you get the standard Mandelbrot set.\n"
					   "For a power of 3, you get 2-fold symmetry in your guide image and 3-fold symmetry \n"
					   "in the set you generate. Generally, you get (n-1)-fold symmetry for the guide image and \n"
					   "n-fold symmetry in the generated sets.\n"
					   "Lower limit is -20 and upper limit is 20.",-20,20);
		time = parseBool("\nTells you additional infomation about timing.\n"
						 "Based on the time it takes, you can select a size runnable by your computer.\n"
						 "Enter 0 for FALSE and 1 for TRUE, then hit ENTER to proceed.");
		useGPU = parseBool("\nUses GPU acceleration to generate image\n"
						   "Works well with animation, but don't expect miracles.\n"
						   "If GPU isn't CUDA ready or it's not NVIDIA, I doubt it'll work.\n"
						   "If you don't select this, your CPU is used instead.\n"
						   "Enter 0 for FALSE and 1 for TRUE, then hit ENTER to proceed.");
		if (useGPU) {
			ts = parseInt("\nEnter the number of threads per dimension you want to use.\n"
						  "The lower limit is 8, the upper limit is 16.", 8, 16);
		}
		animate = parseBool("\nWhether animations are used or not.\n"
							"If you don't use your GPU, i'd strongly suggest not enabling this\n"
							"Enter 0 for FALSE and 1 for TRUE, then hit ENTER to proceed.");
	}
	else {
		std::cout << "\nYou have chosen default mode. The default image size is 512 by 512 pixels,\n"
				  << "the Mandelbrot set is generated as the guide and the Julia set corresponding to the \n"
				  << "point is generated. Animations are disabled and timing is shown.\n";
		useGPU = parseBool("\nWould you like to use GPU acceleration? 8 threads are used by default.\n"
						   "If you don't use GPU acceleration, your CPU is used to generate the pictures.\n"
						   "Enter 0 for FALSE and 1 for TRUE, then hit ENTER to proceed.");
	}
	volatile int border[4] = { 0,0,size,size };

	int circle_radius = int(size * RADIUS_CONSTANT);
	int font_size = int(size * FONT_CONSTANT);
	int intertext_space = font_size + 2;
	
	int y = size/2;
	int x = size/2;
	thrust::complex<float> startingPoint = thrust::complex<float>(0.5f, -0.01f);
	std::string additionalTitle = "";
	generateImage(mandelbrot, thrust::complex<float>(2.0f, 2.0f), size, time, useGPU, saveFile, additionalTitle, ts, abs(pwr), uint8_t(255));
	generateImage(julia, startingPoint, size, time, useGPU, saveFile, additionalTitle, ts, pwr, uint8_t(255));
	
	
	cimg_library::CImg<uint8_t> mb_img = cimg_library::CImg<uint8_t>(mandelbrot.linear, size, size, 1, 1);
	cimg_library::CImg<uint8_t> jl_img = cimg_library::CImg<uint8_t>(julia.linear, size, size, 1, 1);
	cimg_library::CImgDisplay mb_disp (mb_img, "Mandelbrot Fractal");
	cimg_library::CImgDisplay jl_disp(jl_img, "Corresponding Julia Fractal");
	mb_disp.display(mb_img).resize(1024, 1024);
	jl_disp.display(jl_img).resize(1024, 1024);
	uint8_t circle_color[1] = { 125 };
	uint8_t text_color[1] = { 0 };
	uint8_t line_color[1] = { 0 };
	while (!mb_disp.is_closed() && !jl_disp.is_closed()) {
		if (mb_disp.button() &&
			mb_disp.mouse_x() * size / mb_disp.width() >= border[0] &&
			mb_disp.mouse_y() * size / mb_disp.width() >= border[1] &&
			mb_disp.mouse_x() * size / mb_disp.height() < border[2] &&
			mb_disp.mouse_y() * size / mb_disp.height() < border[3]) {
			x = mb_disp.mouse_x()* size / mb_disp.width();
			y = mb_disp.mouse_y()* size / mb_disp.height();
			//printf("x : %d, y : %d \n", x, y);
		}
		mb_img = cimg_library::CImg<uint8_t>(mandelbrot.linear, size, size, 1, 1);
		mb_disp.display(mb_img.draw_circle(x, y, circle_radius, circle_color).
			draw_text(2, 2, "x : %f | y : %f", text_color, 0, 0.7f, font_size, changeRange(x,0,size,-2,2),changeRange(y,0,size,-2,2)).
			draw_text(2,2 + intertext_space, "Drawing Julia Set for c = %f + %f*i",text_color,0,0.7f, font_size, changeRange(x, 0, size, -2, 2), changeRange(y, 0, size, -2, 2)).
			draw_text(2,2 + 2* intertext_space, "Click any mouse button to change grey dot position.",text_color,0,0.7, font_size).
			draw_text(2, 2 + 3* intertext_space, "When ready, bring into focus the other display and hit ENTER.", text_color, 0, 0.7, font_size).
			draw_line(0,y,size,y, line_color,0.7).
			draw_line(x,0,x,size,line_color,0.7)).
			wait(jl_disp.key() == cimg_library::cimg::keyENTER);
		if (mb_disp.is_resized()) {
			mb_disp.resize().display(mb_img);
			border[2] = mb_disp.width();
			border[3] = mb_disp.height();
		}
		if (jl_disp.is_resized()) {
			jl_disp.resize().display(jl_img);
		}
		startingPoint = thrust::complex<float>((float(x) * 4 / (size - 1) - 2), (float(y) * 4 / (size - 1) - 2));
		if (jl_disp.key() == cimg_library::cimg::keyENTER) {
			//printf("Pressed ENTER \n");
			if (animate) {
				for (int iterations = 0; iterations < 255; iterations++) {
					deallocateContiguous2DArray(julia);
					generateImage(julia, startingPoint, size, time, useGPU, saveFile, additionalTitle, ts, pwr, uint8_t(iterations));
					jl_img = cimg_library::CImg<uint8_t>(julia.linear, size, size, 1, 1);
					jl_disp.display(jl_img).wait(10);
				}
			}
			else {
				deallocateContiguous2DArray(julia);
				generateImage(julia, startingPoint, size, time, useGPU, saveFile, additionalTitle, ts, pwr, 255);
				jl_img = cimg_library::CImg<uint8_t>(julia.linear, size, size, 1, 1);
				jl_disp.display(jl_img).wait(10);
			}
		}
		
	}
	deallocateContiguous2DArray(mandelbrot);

	
	std::cout << "End.\n";
	return 0;
}

