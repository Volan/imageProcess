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

uchar4* getData(IplImage* image) {
    long index, i, j;
    char *str;
    uchar4* data = new uchar4[image->height * image->width];

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


void toGrayScale(uchar4* data, IplImage* outImage) {
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

double* generateGausianBlur(double sigma, int N) {
    double* result = (double*)malloc(sizeof(double) * N);
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


uchar4 getValueBlur(uchar4* data, IplImage* outImage, int x, int y, const double* const filter, const int filterWidth) {
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

void blurImage(uchar4* data, IplImage* outImage, const double* const filter, const int filterWidth) {
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


__global__ void grayCuda(uchar4 *d_in, char *d_out, int d_width, int d_height){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j < d_height && i < d_width) {
	int index = j * d_width + i;
	d_out[index] = (char)(.299f * d_in[index].z + .587f * d_in[index].y + .114f * d_in[index].x);
    }
}

int cudaGrayScale(cudaDeviceProp deviceProp, IplImage* image, IplImage* outImage, uchar4* h_in) {
        int thread = (int)sqrt(deviceProp.maxThreadsPerBlock);

	//Find thread count on block
        while (thread > deviceProp.maxThreadsDim[0] || thread > deviceProp.maxThreadsDim[1])
	    --thread;
	if (thread <= 0) {
	    printf("Don't find block size\n");
	    return -1;
	}

	//Find grid size
	while (image->width / thread > deviceProp.maxGridSize[0] ||  image->height / thread > deviceProp.maxGridSize[1])
	    --thread;
	if (thread <= 0) {
	    printf("Don't find dim size\n");
	    return -1;
	}

	uchar4 *d_in;
	char *d_out;
	size_t ARRAY_INPUT_BYTE = image->height * image->width * sizeof(uchar4);
	size_t ARRAY_OUT_BYTE = image->height * image->width * sizeof(char);

	cudaMalloc((void**) &d_in, ARRAY_INPUT_BYTE);
	cudaMalloc((void**) &d_out, ARRAY_OUT_BYTE);
	cudaMemcpy(d_in, h_in, ARRAY_INPUT_BYTE, cudaMemcpyHostToDevice);


	dim3 threadsPerBlock(thread, thread);
	dim3 numBlocks(ceil((float)image->width / thread), ceil((float)image->height / thread));
	grayCuda<<<numBlocks, threadsPerBlock>>>(d_in, d_out, image->width, image->height);

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
    	if (CUDA && cudaGrayScale(deviceProp, image, outImage, h_in) != 0)
	    return -1;
    	else
	    toGrayScale(h_in, outImage);
    }
    else if (strcmp(argv[1], "blur") == 0) {
	outImage = cvCreateImage(CvSize(image->width, image->height), IPL_DEPTH_8U, 4);

	const int filterWidth = 3;
        double* filter = generateGausianBlur(1, filterWidth);
	blurImage(h_in, outImage, filter, filterWidth);	

	for (int i = 0; i < filterWidth; ++i) {
	    for (int j = 0; j < filterWidth; ++j)
		printf("%f ", filter[i * filterWidth + j]);
	    printf("\n");
	}

	free(filter);
    }

    stop = clock();

    cvSaveImage(argv[3], outImage);

    delete(h_in);
    cvReleaseImage(&image);
    cvReleaseImage(&outImage);

    printf("%f\n", (float)(stop - start) / CLOCKS_PER_SEC);
    return 0;
}