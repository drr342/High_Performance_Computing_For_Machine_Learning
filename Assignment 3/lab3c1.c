/******************************************************************************
* FILE: lab3.c
* DESCRIPTION:
*   Tensor Arithmetic Mean with MPI
* AUTHOR: Daniel Rivera Ruiz (drr342@nyu.edu)
******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mpi.h"

#define MASTER 0
#define C 3
#define H 1024
#define W 1024
#define MSEC 1000.0
#define NSEC 1000000.0

struct timespec start, end;
double time_eps;
int r, R;

void c1 () {
    double* Ir = malloc(sizeof(double) * C * H * W);
    MPI_Request request;
    MPI_Status status;

    if (r != MASTER) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                	Ir[c * H * W + h * W + w] = r + c * (h + w);
                }
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Isend(Ir, C * H * W, MPI_DOUBLE, MASTER, r, MPI_COMM_WORLD, &request);
		MPI_Wait(&request, &status);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (r == MASTER) {
        int source;
		double* O = malloc(sizeof(double) * C * H * W);
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    O[c * H * W + h * W + w] = 0.0;
                }
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        clock_gettime(CLOCK_MONOTONIC, &start);
        for (source = 1; source < R; source++) {
            MPI_Irecv(Ir, C * H * W, MPI_DOUBLE, source, source, MPI_COMM_WORLD, &request);
			MPI_Wait(&request, &status);
            for (int c = 0; c < C; c++) {
                for (int h = 0; h < H; h++) {
                    for (int w = 0; w < W; w++) {
                        O[c * H * W + h * W + w] += Ir[c * H * W + h * W + w];
                    }
                }
            }
        }
        double checksum = 0.0;
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    O[c * H * W + h * W + w] /= (R - 1);
                    checksum += O[c * H * W + h * W + w];
                }
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        clock_gettime(CLOCK_MONOTONIC, &end);
        time_eps = (end.tv_sec * MSEC + end.tv_nsec / NSEC) - (start.tv_sec * MSEC + start.tv_nsec / NSEC);
        printf("%.2lf, %4.3lf\n", checksum, time_eps);
        free(O);
    }
    free(Ir);
}

int main (int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &R);
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    c1();
    MPI_Finalize();
}
