#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cusparse.h"

#include <stdio.h>
#include <iostream>
#include <fstream>

#include <stdlib.h>
#include <time.h>
#include <algorithm>

#include <thrust/count.h>

using namespace std;

// Reads Matrix in Matrix Market format (.mtx)
// Returns Matrix in Compressed-Row Format (CSR)
void read_mat(char* filename, float*& csrValA, int*& csrRowPtrA, int*& csrColIndA, int& cols, int& rows, int& nnz) {

    cout << "Reading matrix " << filename << endl;

    ifstream readfile(filename);

    while (readfile.peek() == '%') readfile.ignore(2048, '\n');

    // Ignore lines with % (comment)
    // First line: dimension of matrix + non-zero values
    // Read defining parameters:
    readfile >> rows >> cols >> nnz;

    csrValA = new float[nnz];
    csrRowPtrA = new int[rows + 1]; csrRowPtrA[0] = 0;
    csrColIndA = new int[nnz];

    int* row_amount = new int[rows];
    // init with zeros

    for (int row = 0; row < rows; row++) row_amount[row] = 0;

    // Second and beyond lines: row column value

    for (int i = 0; i < nnz; i++)
    {
        int m, n;
        float data;
        readfile >> m >> n >> data;
        row_amount[m - 1]++;
        csrValA[i] = data;
        csrColIndA[i] = n - 1;
    }

    int count = 0;
    // Finding cumulative sum
    /*for (int row = 0; row < rows; row++) {
        csrRowPtrA[row] += count;
        count += row_amount[row];
    }
    csrRowPtrA[rows] += count;*/

    for (int row = 1; row <= rows; row++) {
        csrRowPtrA[row] = csrRowPtrA[row - 1] + row_amount[row - 1];
    }

    readfile.close();

    delete[] row_amount;

    cout << "Matrix stored in CSR format.\n";
}

// graph coloring

__global__ void color_jpl_kernel(int n, int c, const int* Ao,
    const int* Ac, const float* Av,
    const int* randoms, int* colors)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x;
        i < n;
        i += blockDim.x * gridDim.x)
    {
        bool f = true; // true iff you have max random

        // ignore nodes colored earlier
        if ((colors[i] != -1)) continue;

        int ir = randoms[i];

        // look at neighbors to check their random number
        for (int k = Ao[i]; k < Ao[i + 1]; k++) {
            // ignore nodes colored earlier (and yourself)
            int j = Ac[k];
            int jc = colors[j];
            if (((jc != -1) && (jc != c)) || (i == j)) continue;
            int jr = randoms[j];
            if (ir <= jr) f = false;
        }

        // assign color if you have the maximum random number
        if (f) colors[i] = c;
    }
}

int get_rand(int max) {
    srand((unsigned)time(NULL));
    return rand() % max;
}

void init_rand_array(int*& randoms, int size) {
    for (int i = 0; i < size; i++) {
        randoms[i] = get_rand(size << 1);
    }
}

void color_jpl(int n,
    const int* Ao, const int* Ac, const float* Av,
    int* colors)
{
    int* randoms; // allocate and init random array 
    randoms = new int[n];

    init_rand_array(randoms, n);

    thrust::fill(colors, colors + n, -1); // init colors to -1

    cout << "initiallized random numbers and colors\n";

    for (int c = 0; c < n; c++) {
        int nt = 256;
        int nb = min((n + nt - 1) / nt, 1000);
        color_jpl_kernel << <nb, nt >> > (n, c,
            Ao, Ac, Av,
            randoms,
            colors);
        int left = (int)thrust::count(colors, colors + n, -1);
        if (left == 0) break;
    }

    delete[] randoms;
}



int main()
{
    float* csrVal;
    int* csrRowPtr;
    int* csrColInd;

    int nnz, rows, cols;

    read_mat("Matrices/offshore.mtx", csrVal, csrRowPtr, csrColInd, cols, rows, nnz);

    cout << "Rows cols nnz " << rows << " " << cols << " " << nnz << endl;

    int* d_csrRowPtr, * d_csrColInd;
    float* d_csrVal;

    // Separating space on GPU for matrix
    cudaMalloc((void**)&d_csrRowPtr, (rows + 1) * sizeof(int));
    cudaMalloc((void**)&d_csrColInd, nnz * sizeof(int));
    cudaMalloc((void**)&d_csrVal, nnz * sizeof(float));

    // color and reordering info

    int ncolors = 0, * coloring;
    int* d_coloring, * d_reordering;
    float fraction = 1.0;
    coloring = (int*)calloc(rows, sizeof(int));

    // separating space for colors and reordering in gpu

    cudaMalloc((void**)&d_coloring, rows * sizeof(int));
    cudaMalloc((void**)&d_reordering, rows * sizeof(int));
    cudaMemset(d_reordering, 0, rows * sizeof(int));
    cudaDeviceSynchronize();

    // Sending matrix info to GPU

    cudaMemcpy(d_csrRowPtr, csrRowPtr, (rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrColInd, csrColInd, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrVal, csrVal, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    cusparseStatus_t status;
    cusparseHandle_t handle;
    status = cusparseCreate(&handle);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        printf("error!");
        exit(1);
    }
    cusparseMatDescr_t descr;
    status = cusparseCreateMatDescr(&descr);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        printf("error!");
        exit(1);
    }

    cusparseColorInfo_t info;
    status = cusparseCreateColorInfo(&info);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        printf("error!");
        exit(1);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    // color
    status = cusparseScsrcolor(handle, rows, nnz, descr, d_csrVal, d_csrRowPtr, d_csrColInd, &fraction, &ncolors, d_coloring, d_reordering, info);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cout << "Milliseconds for operation (CC): " << milliseconds << endl;
    cout << "Colors: " << ncolors << endl;

    /*
    for (int i = 0; i < rows; i++) {
        printf("coloring[%d]: %d\n", i, coloring[i]);
    }

    for (int i = 0; i < rows; i++) {
        printf("reordering[%d]: %d\n", i, reordering[i]);
    }*/
    //cout << "coloring " << coloring << endl;
    //cout << "reordering " << reordering << endl;

    int* colors = new int[nnz];
    cout << "JPL algorithm time\n";
    /*
    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);

    cudaEventRecord(start2);
    color_jpl(nnz, csrRowPtr, csrColInd, csrVal, colors);
    cudaEventRecord(stop2);

    cudaEventSynchronize(stop2);
    float jpl_milli = 0;
    cudaEventElapsedTime(&jpl_milli, start2, stop2);

    cout << "Milliseconds for operation (JPL): " << jpl_milli << endl;

    int jpl_colors = 0;
    for (int i = 0; i < nnz; i++) {
        if (colors[i] > jpl_colors) {
            jpl_colors = colors[i];
        }
    }

    cout << "Colors: " << jpl_colors << endl;
    */
    delete[] csrVal;
    delete[] csrRowPtr;
    delete[] csrColInd;
    delete[] colors;

    cudaFree(d_csrRowPtr);
    cudaFree(d_csrColInd);
    cudaFree(d_csrVal);

}