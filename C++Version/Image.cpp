#include "Image.h"

// Empty constructor. Initializes an image of size 0.
Image::Image() {
	this->image_x = 0;
	this->image_y = 0;
}

// Copy constructor. Calls copy image.
Image::Image(const Image& img) {
	if (!img.imageMatrix) {
		printf("Nothing to copy.\n");
	}
	else {
		copyImage(img.imageMatrix, img.image_x, img.image_y);
	}
}

// Initializes an empty X by Y image.
Image::Image(int size_x, int size_y) {
	if ((size_x <= 0) || (size_y <= 0)) {
		printf("x = %d, y = %d \n", size_x, size_y);
		throw std::runtime_error("Cannot have width or height smaller than 1.");
	}
	this->image_x = size_x;
	this->image_y = size_y;
	this->imageMatrix = dynamicallyAllocateImage(size_x, size_y);
}

// Deallocates used space.
Image::~Image() {
	if ((this->image_x > 0) && (this->image_y > 0)) {
		freeImage(this->imageMatrix, this->image_x, this->image_y);
	}
	else {
		printf("Nothing to free ! \n");
	}
	printf("Destructor called, space deallocated.");
}

// Takes a matrix and copies its contents to image.
void Image::copyImage(uint16_t*** matrix, int size_x, int size_y) {
	// If thre is no matrix, allocate one with the given sizes
	if (!imageMatrix) {
		this->imageMatrix = dynamicallyAllocateImage(size_x, size_y);
	}
	// If there is a matrix but the sizes are different, deallocate and then reallocate 
	else if ((imageMatrix) && ((this->image_x != size_x) || (this->image_y != size_y))) {
		freeImage(this->imageMatrix, this->image_x, this->image_y);
		this->image_x = size_x;
		this->image_y = size_y;
		this->imageMatrix = dynamicallyAllocateImage(size_x, size_y);
	}
	for (int i = 0; i < size_x; i++) {
		for (int j = 0; j < size_y; j++) {
			this->imageMatrix[i][j][0] = matrix[i][j][0];
			this->imageMatrix[i][j][1] = matrix[i][j][1];
			this->imageMatrix[i][j][2] = matrix[i][j][2];
		}
	}
}

// Gets image X size
int Image::getImageX() {
	return this->image_x;
}

// Gets image Y size
int Image::getImageY() {
	return this->image_y;
}

// Gets the image matrix.
uint16_t*** Image::getMatrix() {
	if ((this->image_x == 0) && (this->image_y == 0)) {
		printf("Image is not initialized");
		return nullptr;
	}
	return imageMatrix;
}

// Allocates an X by Y by 3 image
uint16_t*** Image::dynamicallyAllocateImage(int size_x, int size_y) {
	uint16_t*** rows = (uint16_t***)malloc(size_x * sizeof(uint16_t**));
	for (int i = 0; i < size_x; i++) {
		rows[i] = (uint16_t**)malloc(size_y * sizeof(uint16_t*));
		for (int j = 0; j < size_y; j++) {
			rows[i][j] = (uint16_t*)malloc(3 * sizeof(uint16_t));
		}
	}
	return rows;
}

// Deallocates space, called in destructor
void Image::freeImage(uint16_t*** matrix, int size_x, int size_y) {
	for (int i = 0; i < size_x; i++) {
		for (int j = 0; j < size_y; j++) {
			free(matrix[i][j]);
		}
		free(matrix[i]);
	}
	free(matrix);
}
