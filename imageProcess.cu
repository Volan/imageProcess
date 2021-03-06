#include <stdio.h>
#include <string.h>
#include <time.h>
#include <float.h>
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

__host__ float4* getHDRData(IplImage* image) {
    long index, i, j;
    float* str;
    float4* data = (float4*)malloc(sizeof(float4) * image->height * image->width);

    for (i = 0; i < image->height; ++i) {
	str = (float*)(image->imageData + i * image->widthStep);
	for (j = 0; j < image->width; ++j) {
	    index = i * image->width + j;
	    data[index].x = str[image->nChannels * j]; //B
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
    extern __shared__ double flt[];

    long x, y, i, j, fltIndex, stencilIndex, index;

    x = blockIdx.x * blockDim.x + threadIdx.x;
    y = blockIdx.y * blockDim.y + threadIdx.y;

    index = blockDim.x * threadIdx.y + threadIdx.x;
    if (index < filterWidth * filterWidth)
	flt[index] = filter[index];
    __syncthreads();

    if (x >= numCols || y >= numRows)
	return;
    index = y * numCols + x;

    outChannel[index] = fltIndex = 0;
    for (i = x - filterWidth / 2; i <= x + filterWidth / 2; ++i) {
	for (j = y - filterWidth / 2; j <= y + filterWidth / 2; ++j) {
	    if (x < 0 || x >= numCols || y < 0 || y >= numRows)
		continue;
	    stencilIndex = j * numCols + x;
	    outChannel[index] += inChannel[stencilIndex] * flt[fltIndex++];
	}
    }
}



__global__
void separateChannelsChar(const uchar4* const inputImageRGBA,
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

    separateChannelsChar<<<gridSize, blockSize>>>(d_in, outImage->height, outImage->width,
 d_red,
 d_green,
 d_blue);
    cudaDeviceSynchronize();

    cudaMemcpy(d_red_out, d_red, CHANNEL_BYTE, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_green_out, d_green, CHANNEL_BYTE, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_blue_out, d_blue, CHANNEL_BYTE, cudaMemcpyDeviceToDevice);

    gaussian_blur<<<gridSize, blockSize, FILTER_BYTE>>>(d_red, d_red_out, outImage->height, outImage->width,
 d_filter, filterWidth);
    cudaDeviceSynchronize();
    gaussian_blur<<<gridSize, blockSize, FILTER_BYTE>>>(d_green, d_green_out, outImage->height, outImage->width,
 d_filter, filterWidth);
    cudaDeviceSynchronize();
    gaussian_blur<<<gridSize, blockSize, FILTER_BYTE>>>(d_blue, d_blue_out, outImage->height, outImage->width,
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

float getValue(int i, int j, const float4* const data, IplImage* outImage, int channel) {
    long index = i * outImage->width + j;
    return (channel == 0 ? data[index].x : (channel == 1 ? data[index].y : data[index].z));
}

__host__ void hdrChannelToImage(const float4* const data, IplImage* outImage, int channel, int numBins) {
    float min = FLT_MAX, max = FLT_MIN, value, range;
    long i, j, vCDF;
    char pixel, *ptr;

    //min max
    for (i = 0; i < outImage->height; ++i) {
	for (j = 0; j < outImage->width; ++j) {
	    value = getValue(i, j, data, outImage, channel);
	    if (value < min)
		min = value;
	    if (value > max)
		max = value;
	}
    }

    //range
    range = max - min;

    int* cdf = (int*)malloc(sizeof(int) * numBins);
    for (i = 0; i < numBins; ++i)
	cdf[i] = 0;

    //histo
    for (i = 0; i < outImage->height; ++i) {
	for (j = 0; j < outImage->width; ++j) {
	    value = getValue(i, j, data, outImage, channel);
	    cdf[(int)fmin(numBins - 1, (int)((numBins * (value - min)) / range))]++;
	}
    }

    //include
    for (i = 1; i < numBins; ++i)
	cdf[i] = cdf[i - 1] + cdf[i];

    for (i = 0; i < outImage->height; ++i) {
	ptr = outImage->imageData + i * outImage->widthStep;
	for (j = 0; j < outImage->width; ++j) {
	    value = getValue(i, j, data, outImage, channel);
	    vCDF = cdf[(int)fmin(numBins - 1, (int)((numBins * (value - min)) / range))];
	    pixel = round(((float)(vCDF - cdf[0]) / (outImage->height * outImage->width - 1)) * 255);
	    ptr[j * outImage->nChannels + channel] = pixel;
	}
    }
    free(cdf);
}

__host__ void hdrToImage(const float4* const data, IplImage* outImage) {
    const int numBins = 1024;
    hdrChannelToImage(data, outImage, 0, numBins); //B
    hdrChannelToImage(data, outImage, 1, numBins); //G
    hdrChannelToImage(data, outImage, 2, numBins); //R
}



__global__ 
void separateChannelsFloat(const float4* const data,
					int numRows, int numCols,

					float* const redChannel,

					float* const greenChannel,

					float* const blueChannel)
 {

    long x, y, index;
    x = blockIdx.x * blockDim.x + threadIdx.x;
    y = blockIdx.y * blockDim.y + threadIdx.y;
    index = y * numCols + x;

    if (x >= numCols || y >= numRows)
	return; 

    redChannel[index] = data[index].z;
    greenChannel[index] = data[index].y;
    blueChannel[index] = data[index].x;
}

__global__ void reduce(float* const channel, int size, float* out, int op) {
    extern __shared__ float data[];

    long limit, index;

    index = blockIdx.x * blockDim.x + threadIdx.x;
    limit = index < size ? blockDim.x : (size % blockDim.x);

    if (index < size)
	data[threadIdx.x] = channel[index];
    __syncthreads();

    for (long s = blockDim.x / 2; s > 0; s >>= 1) {
	if (threadIdx.x < s && (threadIdx.x + s) < limit) {
	    if (op == 0) {
		if (data[threadIdx.x] < data[threadIdx.x + s])
		    data[threadIdx.x] = data[threadIdx.x + s];
	    } else {
		if (data[threadIdx.x] > data[threadIdx.x + s])
		    data[threadIdx.x] = data[threadIdx.x + s];
	    }
	}
    	__syncthreads();
    }
    if (threadIdx.x == 0) {
	out[blockIdx.x] = data[0];
    }
}

__host__ void reduceCompute(float *d_channel, const long size, cudaDeviceProp deviceProp, float *h_min, float *h_max) {
    *h_min = FLT_MAX;
    *h_max = FLT_MIN;

    long thread = deviceProp.maxThreadsPerBlock;
    long gridSize = ceil((float)size / thread), i;

    long REDUCE = sizeof(float) * thread;
    long OUT_BYTE = sizeof(float) * gridSize;
    float* d_out, *h_out = (float*)malloc(OUT_BYTE);

    cudaMalloc((void**)&d_out, OUT_BYTE);
    reduce<<<gridSize, thread, REDUCE>>>(d_channel, size, d_out, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, OUT_BYTE, cudaMemcpyDeviceToHost);

    *h_max = h_out[0];
    for (i = 1; i < gridSize; ++i)
	if (h_out[i] > *h_max)
	    *h_max = h_out[i];

    reduce<<<gridSize, thread, REDUCE>>>(d_channel, size, d_out, 1);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, OUT_BYTE, cudaMemcpyDeviceToHost);

    *h_min = h_out[0];
    for (i = 1; i < gridSize; ++i)
	if (h_out[i] < *h_min)
	    *h_min = h_out[i];

    cudaFree(d_out);
    free(h_out);
}

__global__ void histogram(float *channel, const long size, int *cdf, const long numBins, const float min, const float range) {
    extern __shared__ int hist[];
    if (threadIdx.x < numBins) {
	hist[threadIdx.x] = 0;
    }
    __syncthreads();

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int iCdf;
    if (index < size) {
        iCdf = (int)((numBins * (channel[index] - min)) / range);
	if (numBins - 1 < iCdf)
	    iCdf = numBins - 1;
	atomicAdd(&hist[iCdf], 1);
    }
    __syncthreads();
    if (threadIdx.x < numBins) {
	atomicAdd(&cdf[threadIdx.x], hist[threadIdx.x]);
    }
}

__host__ int* histogramCompute(float *d_channel, const long size, cudaDeviceProp deviceProp, float h_min, float h_max, const int numBins) {
    long SIZE_CDF = sizeof(int) * numBins;
    int* h_cdf = (int*)malloc(SIZE_CDF), *d_cdf;

    cudaMalloc((void**)&d_cdf, SIZE_CDF);
    cudaMemset(&d_cdf, 0, SIZE_CDF);	 

    long thread = deviceProp.maxThreadsPerBlock;
    long gridSize = ceil((float)size / thread);
    histogram<<<gridSize, thread, SIZE_CDF>>>(d_channel, size, d_cdf, numBins, h_min, h_max - h_min);

    cudaMemcpy(h_cdf, d_cdf, SIZE_CDF, cudaMemcpyDeviceToHost);
    cudaFree(d_cdf);
    return h_cdf;
}

__global__ void scan(int *in, const long numBins) {
    int gap = 1;

    while (gap < numBins) {
	if (threadIdx.x + gap < numBins) {
	    atomicAdd(&in[threadIdx.x + gap], in[threadIdx.x]);
	}
	__syncthreads();
	gap *= 2;
    }
}

__host__ void scanCompute(int *h_in, const int numBins, cudaDeviceProp deviceProp) {
    long SIZE_CDF = sizeof(int) * numBins;
    int *d_cdf;

    cudaMalloc((void**)&d_cdf, SIZE_CDF);
    cudaMemcpy(d_cdf, h_in, SIZE_CDF, cudaMemcpyHostToDevice);
    scan<<<1, numBins>>>(d_cdf, numBins);

    cudaMemcpy(h_in, d_cdf, SIZE_CDF, cudaMemcpyDeviceToHost);
    cudaFree(d_cdf);
}

__global__ void map(char *out, float *channel, const long size, int *cdf, int numBins, const long numCols, float min, float range) {
    int vCDF;
    long x, y, index;
    x = blockIdx.x * blockDim.x + threadIdx.x;
    y = blockIdx.y * blockDim.y + threadIdx.y;
    index = y * numCols + x;

    if (index < size) {
	vCDF = cdf[(int)fmin( (float)numBins - 1, (numBins * (channel[index] - min)) / range)];
	out[index] = llround(((float)(vCDF - cdf[0]) / (size - 1)) * 255);
    }
}

__host__ char* mapCompute(cudaDeviceProp deviceProp, float *d_channel, int *h_cdf, int numBins, const long numCols, const long numRows, float h_min, float h_max) {
    long SIZE_CDF = sizeof(int) * numBins;
    long size = numCols * numRows;
    long SIZE_OUT = sizeof(char) * size;
    int *d_cdf;
    char* channel = (char*)malloc(SIZE_OUT), *d_out;

    cudaMalloc((void**)&d_cdf, SIZE_CDF);
    cudaMemcpy(d_cdf, h_cdf, SIZE_CDF, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_out, SIZE_OUT);

    int thread = (int)sqrt(deviceProp.maxThreadsPerBlock);
    dim3 blockSize(thread, thread);
    dim3 gridSize(ceil((float)numCols / thread), ceil((float)numRows / thread));

    map<<<gridSize, blockSize>>>(d_out, d_channel, size, d_cdf, numBins, numCols, h_min, h_max - h_min);

    cudaMemcpy(channel, d_out, SIZE_OUT, cudaMemcpyDeviceToHost);

    cudaFree(d_channel);
    cudaFree(d_cdf);
    cudaFree(d_out);
    return channel;
}

__host__ void computeChannel(cudaDeviceProp deviceProp, IplImage* outImage, float *d_channel, const int numBins, int channel) {
    float h_min, h_max;
    int *cdf;
    long SIZE = outImage->width * outImage->height, i, j;
    char *h_channel;

    reduceCompute(d_channel, SIZE, deviceProp, &h_min, &h_max);
    cdf = histogramCompute(d_channel, SIZE, deviceProp, h_min, h_max, numBins);
    scanCompute(cdf, numBins, deviceProp);
    h_channel = mapCompute(deviceProp, d_channel, cdf, numBins, outImage->width, outImage->height, h_min, h_max);

    char* ptr;
    for (i = 0; i < outImage->height; ++i) {
	ptr = outImage->imageData + i * outImage->widthStep;
	for (j = 0; j < outImage->width; ++j)
	    ptr[j * outImage->nChannels + channel] = h_channel[i * outImage->width + j];
    }

    free(cdf);
    free(h_channel);
}

__host__ int cudaHdrToImage(cudaDeviceProp deviceProp, const float4* const h_data, IplImage* outImage) {
    const int numBins = 1024;
    long SIZE = outImage->width * outImage->height;
    size_t IN_BYTE = sizeof(float4) * SIZE;
    size_t CHANNEL_BYTE = sizeof(float) * SIZE;

    float4* d_in;
    float *d_red, *d_green, *d_blue;


    int thread = (int)sqrt(deviceProp.maxThreadsPerBlock);
    dim3 blockSize(thread, thread);
    dim3 gridSize(ceil((float)outImage->width / thread), ceil((float)outImage->height / thread));

    cudaMalloc((void**)&d_in, IN_BYTE);
    cudaMalloc((void**)&d_red, CHANNEL_BYTE);
    cudaMalloc((void**)&d_green, CHANNEL_BYTE);
    cudaMalloc((void**)&d_blue, CHANNEL_BYTE);

    cudaMemcpy(d_in, h_data, IN_BYTE, cudaMemcpyHostToDevice);

    separateChannelsFloat<<<gridSize, blockSize>>>(d_in, outImage->height, outImage->width,
 d_red,
 d_green,
 d_blue);
    cudaDeviceSynchronize();

    computeChannel(deviceProp, outImage, d_blue, numBins, 0);
    computeChannel(deviceProp, outImage, d_green, numBins, 1);
    computeChannel(deviceProp, outImage, d_red, numBins, 2);

    cudaFree(d_in);
    cudaFree(d_red);
    cudaFree(d_green);
    cudaFree(d_blue);

    return 0;
}

__host__ int main( int argc, char **argv) {
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

    IplImage* image = cvLoadImage(argv[2], CV_LOAD_IMAGE_COLOR | CV_LOAD_IMAGE_ANYDEPTH);
    if (image == NULL || image->imageData == NULL) {
	printf("Could not open or find the image: %s\n", argv[2]);
	return -1;
    }
    if (image->nChannels != BGR && image->nChannels != BGRA) {
	printf("Image is not color. Number channels is %d\n", image->nChannels);
	return -1;
    }

    start = clock();
    uchar4* h_in;
    float4* h_in_hdr;

    if (CUDA) {
	if (cudaGetDeviceProperties(&deviceProp, 0) != 0) {
	    	printf("Error get device prop\n");
	    	return -1;
	}
    }

    if (strcmp(argv[1], "gray") == 0) {
	h_in = getData(image);
	outImage = cvCreateImage(CvSize(image->width, image->height), IPL_DEPTH_8U, 1);
    	if (CUDA) {
	    if (cudaGrayScale(deviceProp, image, outImage, h_in) != 0)
		return -1;
	}
    	else {
	    toGrayScale(h_in, outImage);
	}
    	free(h_in);
    }
    else if (strcmp(argv[1], "blur") == 0) {
	h_in = getData(image);
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
    	free(h_in);
    } else if (strcmp(argv[1], "histoEqual") == 0) {
	h_in_hdr = getHDRData(image);
	outImage = cvCreateImage(CvSize(image->width, image->height), IPL_DEPTH_8U, 4);
	if (CUDA) {
	    if (cudaHdrToImage(deviceProp, h_in_hdr, outImage) != 0)
		return -1;
	}
	else {
	    hdrToImage(h_in_hdr, outImage);
	}
    	free(h_in_hdr);
    }

    stop = clock();

    cvSaveImage(argv[3], outImage);

    cvReleaseImage(&image);
    cvReleaseImage(&outImage);

    printf("%f\n", (float)(stop - start) / CLOCKS_PER_SEC);
    return 0;
}