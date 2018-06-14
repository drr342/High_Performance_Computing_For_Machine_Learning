/*
 ============================================================================
 Name        : lab1.c
 Author      : Daniel RIvera Ruiz
 Version     :
 Copyright   : Your copyright notice
 Description : HPC Lab 1
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include "mkl.h"

#define DIM_X0_Y 50176
#define DIM_W0_Y 4000
#define DIM_W1_Y 1000

double* initialize (int dimX, int dimY) {
	double* m = malloc(dimX * dimY * sizeof(double));
	for (int i = 0; i < dimX; i++) {
		for (int j = 0; j < dimY; j++) {
			m[dimY * i + j] = 0.5 + ((i + j) % 50 - 30) / 50.0;
		}
	}
	return m;
}

double* column (double* A, int Ax, int Ay, int c) {
	double* column = malloc(Ax * sizeof(double));
	for (int j = 0; j < Ax; j++) {
		column[j] = A[Ay * j + c];
	}
	return column;
}

double vectorProduct (double* x, double* y, int size) {
	double product = 0;
	for (int i = 0; i < size; i++) {
		product += x[i] * y[i];
	}
	return product;
}

double* matrixProduct (double* A, int Ax, double* B, int Bx, int By) {
	double* C = malloc(Ax * By * sizeof(double));
	for (int i = 0; i < Ax; i++) {
		for (int j = 0; j < By; j++) {
			C[By * i + j] = vectorProduct(A, column(B, Bx, By, j), Bx);
		}
	}
    return C;
}

double* relu (double* x, int size) {
	double* out = malloc(size * sizeof(double));
	for (int i = 0; i < size; i++) {
		out[i] = (x[i] < 0) ? 0 : x[i];
	}
	return out;
}

double sum (double* x, int size) {
	double out = 0;
	for (int i = 0; i < size; i++) {
		out += x[i];
	}
	return out;
}

int main(void) {
	struct timespec start, end;
	/////// C1 ///////////////////////////////////////////////////////////////////////
	printf("\n============================================\n");
	printf("          C1: Memory Bandwidth\n");
	printf("============================================\n\n");
	srand(time(NULL));
	long length = 367001600;
	double time, min = DBL_MAX;
	for (int i = 0; i < 16; i++) {
		double* array = malloc(length * sizeof(double));
		for (long i = 0; i < length; i++) {
			array[i] = (double) rand() / RAND_MAX;
		}
		clock_gettime(CLOCK_MONOTONIC, &start);
		double sum = 0;
		for (long j = 0; j < length; j++) {
			sum += array[j];
		}
		free(array);
		printf("Sum %d = %.4f\n", i, sum);
		clock_gettime(CLOCK_MONOTONIC, &end);
		time = ((double)end.tv_sec + (double)end.tv_nsec / 1000000000) - ((double)start.tv_sec + (double)start.tv_nsec / 1000000000);
		min = (time < min) ? time : min;
	}
	double data = ((double) length / 1048576) * sizeof(double);
	printf("\nData transferred: %.0lf MB\n", data);
	printf("Min time elapsed: %.4lf s\n", min);
	printf("Max bandwidth reached: %.2lf GB/s\n\n", data / (1024 * min));

	/////// C4 ///////////////////////////////////////////////////////////////////////
	double* x0 = initialize(1, DIM_X0_Y);
	double* w0 = initialize(DIM_X0_Y, DIM_W0_Y);
	double* w1 = initialize(DIM_W0_Y, DIM_W1_Y);

	clock_gettime(CLOCK_MONOTONIC, &start);
	double* z0 = relu(matrixProduct(x0, 1, w0, DIM_X0_Y, DIM_W0_Y), DIM_W0_Y);
	double* z1 = relu(matrixProduct(z0, 1, w1, DIM_W0_Y, DIM_W1_Y), DIM_W1_Y);
	double s = sum(z1, DIM_W1_Y);
	clock_gettime(CLOCK_MONOTONIC, &end);
	time = ((double)end.tv_sec + (double)end.tv_nsec / 1000000000) - ((double)start.tv_sec + (double)start.tv_nsec / 1000000000);

	printf("============================================\n");
	printf("                C4: C Code\n");
	printf("============================================\n\n");
	printf("Checksum S = %.6lf\n", s);
	printf("Execution time = %.6lf s\n\n", time);

	/////// C5 ///////////////////////////////////////////////////////////////////////
	clock_gettime(CLOCK_MONOTONIC, &start);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		1, DIM_W0_Y, DIM_X0_Y, 1.0, x0, DIM_X0_Y, w0, DIM_W0_Y, 0.0, z0, DIM_W0_Y);
	z0 = relu(z0, DIM_W0_Y);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		1, DIM_W1_Y, DIM_W0_Y, 1.0, z0, DIM_W0_Y, w1, DIM_W1_Y, 0.0, z1, DIM_W1_Y);
	z1 = relu(z1, DIM_W1_Y);
	s = sum(z1, DIM_W1_Y);
	clock_gettime(CLOCK_MONOTONIC, &end);
	time = ((double)end.tv_sec + (double)end.tv_nsec / 1000000000) - ((double)start.tv_sec + (double)start.tv_nsec / 1000000000);

	printf("============================================\n");
	printf("         C5: C Code with MKL Library\n");
	printf("============================================\n\n");
	printf("Checksum S = %.6lf\n", s);
	printf("Execution time = %.6lf s\n\n", time);

	free(x0);
	free(w0);
	free(w1);
	return EXIT_SUCCESS;
}
