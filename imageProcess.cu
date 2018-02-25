#include <stdio.h>
#include <string.h>
#include <time.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>
#include "opencv\cv.h"
#include "opencv2\imgcodecs\imgcodecs_c.h"

#define	BGR	3
#define	BGRA	4

__host__ uchar4* getData(IplImage* image) {
    long index, i, j;
    char *str;
    uchar4* data = (uchar4*)malloc(sizeof(uchar4) * image->height * image->width);

    for (i = 0; i < image->height; ++i) {
	str = image->imageData + i * image->widthStep;
	for (j = 0; j < image->width; ++j) {
	    index = i * image->width + j;
	    data[index].x = str[image->nChannels * j + 0]; //B
	    data[index].y = str[image->nChannels * j + 1]; //G
	    data[index].z = str[image->nChannels * j + 2]; //R
	    if (image->nChannels == BGRA)
		data[index].w = str[image->nChannels * j + 3]; //A
	}
    }
    return data;
}


__host__ void toGrayScale(uchar4* data, IplImage* outImage) {
    long index, i, j;
    char *ptr;
    for (i = 0; i < outImage->height; ++i) {
	ptr = outImage->imageData + i * outImage->widthStep;
	for (j = 0; j < outImage->width; ++j) {
	    index = i * outImage->width + j;
	    ptr[j] = (char)(.299f * data[index].z + .587f * data[index].y + .114f * data[index].x);
	}
    }
}

__host__ double* generateGausianBlur(double sigma, int N) {
    double* result = (double*)malloc(sizeof(double) * N * N);
    double mean = N / 2, sum = 0;
    long i, j, index;
    for (i = 0; i < N; ++i) {
	for (j = 0; j < N; ++j) {
	    index = i * N + j;
	    result[index] = exp( -0.5 * (pow(i - mean, 2.0) + pow(j - mean, 2.0))) / (2 * M_PI * sigma * sigma);
	    sum += result[index];
	}
    }

    for (i = 0; i < N * N; ++i)
	result[i] /= sum;
    return result;
}

__global__ void grayCuda(uchar4 *d_in, char *d_out, int d_width, int d_height){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < d_height && i < d_width) {
	int index = j * d_width + i;
	d_out[index] = (char)(.299f * d_in[index].z + .587f * d_in[index].y + .114f * d_in[index].x);
    }
}


__host__ int cudaGrayScale(cudaDeviceProp deviceProp, IplImage* image, IplImage* outImage, const uchar4* const h_in) {
    int thread = (int)sqrt(deviceProp.maxThreadsPerBlock);

    uchar4 *d_in;
    char *d_out;
    size_t ARRAY_INPUT_BYTE = image->height * image->width * sizeof(uchar4);
    size_t ARRAY_OUT_BYTE = image->height * image->width * sizeof(char);

    cudaMalloc((void**) &d_in, ARRAY_INPUT_BYTE);
    cudaMalloc((void**) &d_out, ARRAY_OUT_BYTE);
    cudaMemcpy(d_in, h_in, ARRAY_INPUT_BYTE, cudaMemcpyHostToDevice);

    dim3 blockSize(thread, thread);
    dim3 gridSize(ceil((float)image->width / thread), ceil((float)image->height / thread));
    grayCuda<<<gridSize, blockSize>>>(d_in, d_out, image->width, image->height);
    cudaDeviceSynchronize();

    int i;
    char* ptr = outImage->imageData, *src = d_out;
    for (i = 0; i < image->height; ++i) {
	cudaMemcpy(ptr, src, image->width, cudaMemcpyDeviceToHost);
	ptr += outImage->widthStep;
	src += image->width;
    }
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}

__host__ uchar4 getValueBlur(uchar4* data, IplImage* outImage, int x, int y, const double* const filter, const int filterWidth) {
    long index, i, j, fltIndex = 0;
    uchar4 value = { 0, 0, 0 };
    double filterValue;

    for (i = x - filterWidth / 2; i <= x + filterWidth / 2; ++i) {
	for (j = y - filterWidth / 2; j <= y + filterWidth / 2; ++j) {
	    index = i * outImage->width + j;
	    if (index < 0 || index >= outImage->width * outImage->height)
		continue;

	    filterValue = filter[fltIndex++];

	    value.x += data[index].x * filterValue;
	    value.y += data[index].y * filterValue;
	    value.z += data[index].z * filterValue;
	}
    }
    return value;
}

__host__ void blurImage(uchar4* data, IplImage* outImage, const double* const filter, const int filterWidth) {
    long i, j;
    char *ptr;
    uchar4 value;

    for (i = 0; i < outImage->height; ++i) {
	ptr = outImage->imageData + i * outImage->widthStep;
	for (j = 0; j < outImage->width; ++j) {
	    value = getValueBlur(data, outImage, i, j, filter, filterWidth);
	    ptr[j] = value.x;		//B
	    ptr[j + 1] = value.y;	//G
	    ptr[j + 2] = value.z;	//R
            ptr += 3;
	}
    }
}

__global__
void gaussian_blur(const unsigned char* const inChannel,
			unsigned char* outChannel,
			int numRows, int numCols,

			const double* const filter, const int filterWidth) 
{
    long x, y, i, j, fltIndex, stencilIndex, index;

    x = blockIdx.x * blockDim.x + threadIdx.x;
    y = blockIdx.y * blockDim.y + threadIdx.y;
    index = y * numCols + x;

    if (x >= numCols || y >= numRows)
	return;

    outChannel[index] = fltIndex = 0;
    for (i = x - filterWidth / 2; i <= x + filterWidth / 2; ++i) {
	for (j = y - filterWidth / 2; j <= y + filterWidth / 2; ++j) {
	    if (x < 0 || x >= numCols || y < 0 || y >= numRows)
		continue;
	    stencilIndex = j * numCols + x;
	    outChannel[index] += inChannel[stencilIndex] * filter[fltIndex++];
	}
    }
}



__global__
void separateChannels(const uchar4* const inputImageRGBA,
				int numRows, int numCols,

				unsigned char* const redChannel,

				unsigned char* const  greenChannel,

				unsigned char* const  blueChannel)
 {

    long x, y, index;
    x = blockIdx.x * blockDim.x + threadIdx.x;
    y = blockIdx.y * blockDim.y + threadIdx.y;
    index = y * numCols + x;

    if (x >= numCols || y >= numRows)
	return;

    redChannel[index] = inputImageRGBA[index].z;
    greenChannel[index] = inputImageRGBA[index].y;
    blueChannel[index] = inputImageRGBA[index].x;
}

__host__ int cudaBlur(cudaDeviceProp deviceProp, IplImage* outImage, const uchar4* const h_in, const int filterWidth, const double* const h_filter) {
    size_t SIZE = outImage->width * outImage->height;
    size_t IN_BYTE = sizeof(uchar4) * SIZE;
    size_t CHANNEL_BYTE = sizeof(unsigned char) * SIZE;
    size_t FILTER_BYTE = sizeof(double) * filterWidth * filterWidth;

    int thread = (int)sqrt(deviceProp.maxThreadsPerBlock);
    dim3 blockSize(thread, thread);
    dim3 gridSize(ceil((float)outImage->width / thread), ceil((float)outImage->height / thread));

    uchar4 *d_in;
    unsigned char *d_red, *d_green, *d_blue, *d_red_out, *d_green_out, *d_blue_out;
    unsigned char *h_red, *h_green, *h_blue;
    double *d_filter;

    h_red = (unsigned char*)malloc(sizeof(unsigned char) * SIZE);
    h_green = (unsigned char*)malloc(sizeof(unsigned char) * SIZE);
    h_blue = (unsigned char*)malloc(sizeof(unsigned char) * SIZE);

    cudaMalloc((void**)&d_in, IN_BYTE);
    cudaMalloc((void**)&d_red, CHANNEL_BYTE);
    cudaMalloc((void**)&d_green, CHANNEL_BYTE);
    cudaMalloc((void**)&d_blue, CHANNEL_BYTE);
    cudaMalloc((void**)&d_red_out, CHANNEL_BYTE);
    cudaMalloc((void**)&d_green_out, CHANNEL_BYTE);
    cudaMalloc((void**)&d_blue_out, CHANNEL_BYTE);
    cudaMalloc((void**)&d_filter, FILTER_BYTE);

    cudaMemcpy(d_in, h_in, IN_BYTE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, FILTER_BYTE, cudaMemcpyHostToDevice);

    separateChannels<<<gridSize, blockSize>>>(d_in, outImage->height, outImage->width,
 d_red,
 d_green,
 d_blue);
    cudaDeviceSynchronize();

    cudaMemcpy(d_red_out, d_red, CHANNEL_BYTE, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_green_out, d_green, CHANNEL_BYTE, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_blue_out, d_blue, CHANNEL_BYTE, cudaMemcpyDeviceToDevice);

    gaussian_blur<<<gridSize, blockSize>>>(d_red, d_red_out, outImage->height, outImage->width,
 d_filter, filterWidth);
    cudaDeviceSynchronize();
    gaussian_blur<<<gridSize, blockSize>>>(d_green, d_green_out, outImage->height, outImage->width,
 d_filter, filterWidth);
    cudaDeviceSynchronize();
    gaussian_blur<<<gridSize, blockSize>>>(d_blue, d_blue_out, outImage->height, outImage->width,
 d_filter, filterWidth);
    cudaDeviceSynchronize();

    cudaMemcpy(h_red, d_red_out, CHANNEL_BYTE, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_green, d_green_out, CHANNEL_BYTE, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_blue, d_blue_out, CHANNEL_BYTE, cudaMemcpyDeviceToHost);

    int i, j, index;
    char *ptr;
    for (i = 0; i < outImage->height; ++i) {
	ptr = outImage->imageData + i * outImage->widthStep;
	for (j = 0; j < outImage->width; ++j) {
	    index = i * outImage->width + j;
	    ptr[j] = h_blue[index];	//B
	    ptr[j + 1] = h_green[index];//G
	    ptr[j + 2] = h_red[index];	//R
            ptr += 3;
	}
    }

    cudaFree(d_in);
    cudaFree(d_red);
    cudaFree(d_green);
    cudaFree(d_blue);
    cudaFree(d_red_out);
    cudaFree(d_green_out);
    cudaFree(d_blue_out);
    cudaFree(d_filter);

    free(h_red);
    free(h_green);
    free(h_blue);
    return 0;
}

int main( int argc, char **argv) {
    int CUDA = 0;
    clock_t start, stop;
    cudaDeviceProp deviceProp;
    IplImage* outImage;

    if (argc < 4) {
	printf("Usage option: command srcImage dstImage -CUDA(optional)\n");
	return -1;
    }
    if (argc == 5) {
	if (strcmp(argv[4], "-CUDA") != 0) {
	    printf("Sorry but fourth options must be -CUDA\n");
	    return -1;
	}
	CUDA = 1;
    }

    IplImage* image = cvLoadImage(argv[2]);
    if (image == NULL || image->imageData == NULL) {
	printf("Could not open or find the image: %s\n", argv[2]);
	return -1;
    }
    if (image->nChannels != BGR && image->nChannels != BGRA) {
	printf("Image is not color\n");
	return -1;
    }

    start = clock();
    uchar4* h_in = getData(image);

    if (CUDA) {
	if (cudaGetDeviceProperties(&deviceProp, 0) != 0) {
	    	printf("Error get device prop\n");
	    	return -1;
	}
    }

    if (strcmp(argv[1], "gray") == 0) {
	outImage = cvCreateImage(CvSize(image->width, image->height), IPL_DEPTH_8U, 1);
    	if (CUDA) {
	    if (cudaGrayScale(deviceProp, image, outImage, h_in) != 0)
		return -1;
	}
    	else {
	    toGrayScale(h_in, outImage);
	}
    }
    else if (strcmp(argv[1], "blur") == 0) {
	const int filterWidth = 3;
        double* filter = generateGausianBlur(1, filterWidth);

	outImage = cvCreateImage(CvSize(image->width, image->height), IPL_DEPTH_8U, 4);

	if (CUDA) {
	    if (cudaBlur(deviceProp, outImage, h_in, filterWidth, filter) != 0)
		return -1;
	}
	else {
	    blurImage(h_in, outImage, filter, filterWidth);
	}
	free(filter);
    }

    stop = clock();

    cvSaveImage(argv[3], outImage);

    free(h_in);
    cvReleaseImage(&image);
    cvReleaseImage(&outImage);

    printf("%f\n", (float)(stop - start) / CLOCKS_PER_SEC);
    return 0;
}