//============================================================================
// Name        : lab2.cu
// Author      : Daniel RIvera Ruiz
// Version     :
// Copyright   :
// Description : CNN with CUDA and cuDNN
//============================================================================

#include <iostream>
#include <time.h>
#include <cudnn.h>

#define C 3
#define FH 3
#define FW 3
#define P 1
#define K 5
#define H 1024
#define W 1024
#define B 32

#define CUDNN_CALL(x) do { 								\
	cudnnStatus_t status = (x); 						\
	if (status != CUDNN_STATUS_SUCCESS) { 				\
		fprintf(stderr, "%s:%d ERROR: %s\n", __FILE__, 	\
		__LINE__, cudnnGetErrorString(status)); 		\
		exit(-1); 										\
	} 													\
} while (0)

__device__
double dot(int k, int h, int w, double *I0, double *F) {
	double sum = 0.0;
	for (int c = 0; c < C; c++) {
		for (int fh = 0; fh < FH; fh++) {
			for (int fw = 0; fw < FW; fw++) {
				sum += F[k * C * FH * FW + c * FH * FW + (FH - 1 - fh) * FW + (FW - 1 - fw)] *
					   I0[c * (H + 2 * P) * (W + 2 * P) + (h + fh) * (W + 2 * P) + (w + fw)];
			}
		}
	}
	return sum;
}

__global__
void convolution(double *I0, double *F, double *O) {
	O[blockIdx.z * H * W + blockIdx.y * W + threadIdx.x]
		= dot(blockIdx.z, blockIdx.y, threadIdx.x, I0, F);
}

__global__
void convolution_shared(double *I0, double *F, double *O) {
	__shared__ double sI0[C * B * B];
	__shared__ double sF[C * FH * FW];

	int offset = (blockIdx.y * (W + 2 * P) + blockIdx.x) * (B - 2);
	int index =  offset + threadIdx.y * (W + 2 * P) + threadIdx.x;

	int indexI;
	for (int c = 0; c < C; c++) {
		if (index < (blockIdx.y * (B - 2) + threadIdx.y) * (W + 2 * P) + (H + 2 * P)) {
			indexI = c * (H + 2 * P) * (W + 2 * P) + index;
			sI0[c * B * B + threadIdx.y * B + threadIdx.x] = I0[indexI];
		} else {
			sI0[c * B * B + threadIdx.y * B + threadIdx.x] = 0.0;
		}
		if (threadIdx.y * FH + threadIdx.x < FH * FW) {
			sF[c * FH * FW + threadIdx.y * FH + threadIdx.x] =
				F[blockIdx.z * C * FH * FW + c * FH * FW + threadIdx.y * FH + threadIdx.x];
		}
	}
	__syncthreads();

	double sum = 0.0;
	if (threadIdx.y + FH <= B && threadIdx.x + FW <= B) {
		for (int c = 0; c < C; c++) {
			for (int fh = 0; fh < FH; fh++) {
				for (int fw = 0; fw < FW; fw++) {
					sum += sF[c * FH * FW + (FH - 1 - fh) * FW + (FW - 1 - fw)] *
					   sI0[c * B * B + (threadIdx.y + fh) * B + (threadIdx.x + fw)];
				}
			}
		}
		__syncthreads();

		if (index < (blockIdx.y * (B - 2) + threadIdx.y) * (W + 2 * P) + H
					&& P * (blockIdx.y * (B - 2) + threadIdx.y) < H) {
			int indexO = index - 2 * P * (blockIdx.y * (B - 2) + threadIdx.y);
			O[blockIdx.z * H * W + indexO] = sum;
		}
	}
}

int main() {
	struct timespec start, end;
	double time;
	double msec = 1000.0;
	double nsec = 1000000.0;

	size_t sizeI = sizeof(double) * C * H * W;
	size_t sizeI0 = sizeof(double) * C * (H + 2 * P) * (W + 2 * P);
	size_t sizeF = sizeof(double) * K * C * FH * FW;
	size_t sizeO = sizeof(double) * K * H * W;

	double *I, *I0, *F, *O;
	double *dI0, *dF, *dO;
	I = (double*) malloc(sizeI);
	I0 = (double*) malloc(sizeI0);
	F = (double*) malloc(sizeF);
	O = (double*) malloc(sizeO);
	cudaMalloc(&dI0, sizeI0);
	cudaMalloc(&dO, sizeO);
	cudaMalloc(&dF, sizeF);

	for (int c = 0; c < C; c++) {
		for (int h = 0; h < H; h++) {
			for (int w = 0; w < W; w++) {
				I[c * H * W + h * W + w] = c * (h + w);
			}
		}
	}

	for (int c = 0; c < C; c++) {
		for (int h = 0; h < H + 2 * P; h++) {
			for (int w = 0; w < W + 2 * P; w++) {
				if (h < P || w < P || h >= H + P || w >= W + P) {
					I0[c * (H + 2 * P) * (W + 2 * P) + h * (W + 2 * P) + w] = 0.0;
				} else {
					I0[c * (H + 2 * P) * (W + 2 * P) + h * (W + 2 * P) + w] =
							I[c * H * W + (h - P) * W + (w - P)];
				}
			}
		}
	}

	for (int k = 0; k < K; k++) {
		for (int c = 0; c < C; c++) {
			for (int fh = 0; fh < FH; fh++) {
				for (int fw = 0; fw < FW; fw++) {
					F[k * C * FH * FW + c * FH * FW + fh * FW + fw] = (c + k) * (fh + fw);
				}
			}
		}
	}

	cudaMemcpy(dI0, I0, sizeI0, cudaMemcpyHostToDevice);
	cudaMemcpy(dF, F, sizeF, cudaMemcpyHostToDevice);

////////////////////////////////////////////////////////////////////////////////////

	cudaMemset(dO, 0, sizeO);
	dim3 dimGrid1(1, H, K);
	dim3 dimBlock1(W);

	clock_gettime(CLOCK_MONOTONIC, &start);
	convolution<<<dimGrid1, dimBlock1>>>(dI0, dF, dO);
	cudaMemcpy(O, dO, sizeO, cudaMemcpyDeviceToHost);
	clock_gettime(CLOCK_MONOTONIC, &end);
	time = (end.tv_sec * msec + end.tv_nsec / nsec) - (start.tv_sec * msec + start.tv_nsec / nsec);

	double sum = 0.0;
	for (int k = 0; k < K; k++) {
		for (int h = 0; h < H; h++) {
			for (int w = 0; w < W; w++) {
				sum += O[k * H * W + h * W + w];
			}
		}
	}

	printf("%.2lf,%4.3lf\n", sum, time);

////////////////////////////////////////////////////////////////////////////////////

	cudaMemset(dO, 0, sizeO);
	dim3 dimGrid2((W + 2 * P) / (B - 2) + 1, (H + 2 * P) / (B - 2) + 1, K);
	dim3 dimBlock2(B, B);

	clock_gettime(CLOCK_MONOTONIC, &start);
	convolution_shared<<<dimGrid2, dimBlock2>>>(dI0, dF, dO);
	cudaMemcpy(O, dO, sizeO, cudaMemcpyDeviceToHost);
	clock_gettime(CLOCK_MONOTONIC, &end);
	time = (end.tv_sec * msec + end.tv_nsec / nsec) - (start.tv_sec * msec + start.tv_nsec / nsec);

	sum = 0.0;
	for (int k = 0; k < K; k++) {
		for (int h = 0; h < H; h++) {
			for (int w = 0; w < W; w++) {
				sum += O[k * H * W + h * W + w];
			}
		}
	}

	printf("%.2lf,%4.3lf\n", sum, time);

////////////////////////////////////////////////////////////////////////////////////

	cudaMemset(dO, 0, sizeO);

	cudnnHandle_t cudnn;
	CUDNN_CALL(cudnnCreate(&cudnn));

	cudnnTensorDescriptor_t in_desc;
	CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
	CUDNN_CALL(cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, C, H, W));

	cudnnTensorDescriptor_t out_desc;
	CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
	CUDNN_CALL(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, K, H, W));

	cudnnFilterDescriptor_t filter_desc;
	CUDNN_CALL(cudnnCreateFilterDescriptor(&filter_desc));
	CUDNN_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, K, C, FH, FW));

	cudnnConvolutionDescriptor_t conv_desc;
	CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
	CUDNN_CALL(cudnnSetConvolution2dDescriptor(conv_desc, P, P, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_DOUBLE));

	cudnnConvolutionFwdAlgo_t conv_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

	size_t workspace_bytes = 0;
	CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(cudnn, in_desc, filter_desc, conv_desc, out_desc, conv_algo, &workspace_bytes));
	
	double *dI;
	void *dW;
	cudaMalloc(&dI, sizeI);
	cudaMalloc(&dW, workspace_bytes);

	const double alpha = 1.0, beta = 0.0;

	clock_gettime(CLOCK_MONOTONIC, &start);
	CUDNN_CALL(cudnnConvolutionForward(cudnn, &alpha, in_desc, dI, filter_desc, dF, conv_desc, conv_algo, dW, workspace_bytes, &beta, out_desc, dO));
	cudaDeviceSynchronize();
	clock_gettime(CLOCK_MONOTONIC, &end);
	time = (end.tv_sec * msec + end.tv_nsec / nsec) - (start.tv_sec * msec + start.tv_nsec / nsec);

	sum = 0.0;
	for (int k = 0; k < K; k++) {
		for (int h = 0; h < H; h++) {
			for (int w = 0; w < W; w++) {
				sum += O[k * H * W + h * W + w];
			}
		}
	}

	printf("%.2lf,%4.3lf\n", sum, time);                                 	                                         

////////////////////////////////////////////////////////////////////////////////////

	cudnnDestroyTensorDescriptor(in_desc);
	cudnnDestroyTensorDescriptor(out_desc);
	cudnnDestroyFilterDescriptor(filter_desc);
	cudnnDestroyConvolutionDescriptor(conv_desc);   
	cudnnDestroy(cudnn);
	
	cudaFree(dI);
	cudaFree(dI0);
	cudaFree(dF);
	cudaFree(dO);
	cudaFree(dW);

	free(I);
	free(I0);
	free(F);
	free(O);

}
