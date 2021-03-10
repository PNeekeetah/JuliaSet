#pragma once
#include <iostream>

class Image {

private:

	uint16_t*** imageMatrix;
	int image_x;
	int image_y;

public:

	Image();
	Image(const Image& img);
	Image(int size_x, int size_y);
	~Image();

	void copyImage(uint16_t*** matrix, int size_x, int size_y);
	int getImageX();
	int getImageY();

	uint16_t*** getMatrix();
	uint16_t*** dynamicallyAllocateImage(int size_x, int size_y);
	void freeImage(uint16_t*** matrix, int size_x, int size_y);

};