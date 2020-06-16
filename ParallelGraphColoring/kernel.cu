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

#include <random>
// random number generator
#include <intrin.h>

#pragma intrinsic(__rdtsc)

using namespace std;

int iDivUp(int a, int b) {
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

// Reads Matrix in Matrix Market format (.mtx)
// Returns Matrix in Compressed-Row Format (CSR)
void read_mat(char* filename, float*& csrValA, int*& csrRowPtrA, int*& csrColIndA, int& cols, int& rows, int& nnz) {

    cout << "Reading matrix " << filename << endl;

    ifstream readfile(filename);

    while (readfile.peek() == '%') readfile.ignore(2048, '\n');
    /*
    std::string str;
    getline(readfile, str);
    char c;
    sscanf(str.c_str(), "%c", &c);
    while (c == '%') {
        getline(readfile, str);
        sscanf(str.c_str(), "%c", &c);
    }
    */
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
    //int i = threadIdx.x + blockIdx.x * blockDim.x;
    //if (i < n){
        bool f = true; // true iff you have max random

        // ignore nodes colored earlier
        if ((colors[i] != -1)) continue;
        //if ((colors[i] != -1)) return;

        int ir = randoms[i];

        // look at neighbors to check their random number
        for (int k = Ao[i]; k < Ao[i + 1]; k++) {
            // ignore nodes colored earlier (and yourself)
            int j = Ac[k];
            int jc = colors[j];
            if (((jc != -1) && (jc != c)) || (i == j)) continue;
            int jr = randoms[j];
            if (ir <= jr) {
                f = false;
                break;
            }
        }

        // assign color if you have the maximum random number
        if (f) colors[i] = c;
    }
}

int get_rand(int max, bool seed_time) {
    if (seed_time) srand((unsigned)time(NULL));
    else srand(__rdtsc());
    return rand() % max;
}

void init_rand_array(int*& randoms, int size, bool seed_time) {
    for (int i = 0; i < size; i++) {
        randoms[i] = get_rand(size << 2, seed_time);
        //cout << "rand " << randoms[i] << endl;
    }
}

void color_jpl(int n,
    const int* Ao, const int* Ac, const float* Av,
    int* colors, int* d_randoms)
{

    thrust::fill(colors, colors + n, -1); // init colors to -1

    int* d_colors;
    cudaMalloc((void**)&d_colors, n * sizeof(int));
    cudaMemcpy(d_colors, colors, n * sizeof(int), cudaMemcpyHostToDevice);

    cout << "initiallized random numbers and colors\n";

    //cout << "nodes left: " << (int)thrust::count(colors, colors + n, -1) << endl;
    for (int c = 0; c < n; c++) {
        int nt = 256;
        //int nb = min((n + nt - 1) / nt, 1000);
        int nb = (ceil(n / nt));
        color_jpl_kernel << <nb, nt >> > (n, c,
            Ao, Ac, Av,
            d_randoms,
            d_colors);
        cudaDeviceSynchronize();
        cudaMemcpy(colors, d_colors, n * sizeof(int), cudaMemcpyDeviceToHost);
        int left = (int)thrust::count(colors, colors + n, -1);
        //cout << "nodes left: " << left << endl;
        if (left == 0) break;
    }

    cudaDeviceSynchronize();

    cudaMemcpy(colors, d_colors, n * sizeof(int), cudaMemcpyDeviceToHost);

    //delete[] randoms;
    //cudaFree(d_randoms);
    cudaFree(d_colors);
}


// graph_coloring

void cusparse_graph_coloring(int rows, int nnz, float*& d_csrVal, int*& d_csrRowPtr, int*& d_csrColInd, int*& d_coloring, int*& d_reordering, ofstream& outfile,
    bool doing_ilu = false) {
    // color and reordering info
    
    int ncolors = 0, * coloring;
    //int* d_coloring, * d_reordering;
    float fraction = 1.0;
    coloring = (int*)calloc(rows, sizeof(int));

    // separating space for colors and reordering in gpu

    cudaMalloc((void**)&d_coloring, rows * sizeof(int));
    cudaMalloc((void**)&d_reordering, rows * sizeof(int));
    cudaMemset(d_reordering, 0, rows * sizeof(int));
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

    if (!doing_ilu) {
        outfile << "Milliseconds for operation (CC): " << milliseconds << endl;
        outfile << "Colors: " << ncolors << endl;
    }

}

// implemented algorithm
void graph_coloring(int rows, int nnz, float*& d_csrVal, int*& d_csrRowPtr, int*& d_csrColInd, ofstream& outfile, bool seed_time=true) {  

    int* colors = new int[rows];
    cout << "JPL algorithm time\n";

    int* randoms; // allocate and init random array 
    randoms = new int[rows];

    cout << "Initializing random values\n";
    init_rand_array(randoms, rows, seed_time);
    cout << "Rand values initialized\n";

    int* d_randoms;
    cudaMalloc((void**)&d_randoms, rows * sizeof(int));

    cudaMemcpy(d_randoms, randoms, rows * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);

    cudaEventRecord(start2);
    color_jpl(rows, d_csrRowPtr, d_csrColInd, d_csrVal, colors, d_randoms);
    cudaEventRecord(stop2);

    cudaEventSynchronize(stop2);
    float jpl_milli = 0;
    cudaEventElapsedTime(&jpl_milli, start2, stop2);

    //cudaMemcpy(colors, d_colors, nnz * sizeof(int), cudaMemcpyDeviceToHost);

    cout << "Milliseconds for operation (JPL): " << jpl_milli << endl;

    int jpl_colors = 0;
    for (int i = 0; i < rows; i++) {
        if (colors[i] > jpl_colors) {
            jpl_colors = colors[i];
        }
    }

    cout << "Colors: " << jpl_colors << endl;

    outfile << "JPL operation " << jpl_milli << endl;
    outfile << "Colors: " << jpl_colors << endl;

    delete[] colors;
}

void regular_ilu(int rows, int nnz, float*& d_csrVal, int*& d_csrRowPtr, int*& d_csrColInd, ofstream& outfile) {
    cudaEvent_t start3, stop3;
    cudaEventCreate(&start3);
    cudaEventCreate(&stop3);

    cudaEventRecord(start3);

    cusparseMatDescr_t descr_M = 0;
    cusparseMatDescr_t descr_L = 0;
    cusparseMatDescr_t descr_U = 0;
    csrilu02Info_t info_M = 0;
    csrsv2Info_t  info_L = 0;
    csrsv2Info_t  info_U = 0;

    cusparseStatus_t status;
    cusparseHandle_t handle;
    status = cusparseCreate(&handle);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        printf("error!");
        exit(1);
    }

    status = cusparseCreateMatDescr(&descr_M);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        printf("error!");
        exit(1);
    }

    // L and U matrix descriptors

    cout << "Creating matrix descriptors\n";

    cusparseCreateMatDescr(&descr_L);
    cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER);
    cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_UNIT);

    cusparseCreateMatDescr(&descr_U);
    cusparseSetMatIndexBase(descr_U, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descr_U, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatFillMode(descr_U, CUSPARSE_FILL_MODE_UPPER);
    cusparseSetMatDiagType(descr_U, CUSPARSE_DIAG_TYPE_NON_UNIT);

    csrilu02Info_t info_C = 0; cusparseCreateCsrilu02Info(&info_C);
    cusparseCreateCsrsv2Info(&info_L);
    cusparseCreateCsrsv2Info(&info_U);

    // buffer size

    //cout << "init buffer sizes\n";

    int pBufferSize_M, pBufferSize_L, pBufferSize_U;
    cusparseScsrilu02_bufferSize(handle, rows, nnz, descr_M, d_csrVal, d_csrRowPtr, d_csrColInd, info_C, &pBufferSize_M);

    cusparseScsrsv2_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, rows, nnz,
        descr_L, d_csrVal, d_csrRowPtr, d_csrColInd, info_L, &pBufferSize_L);
    cusparseScsrsv2_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, rows, nnz,
        descr_U, d_csrVal, d_csrRowPtr, d_csrColInd, info_U, &pBufferSize_U);

    int pBufferSize = max(pBufferSize_M, max(pBufferSize_L, pBufferSize_U));
    void* pBuffer = 0; cudaMalloc((void**)&pBuffer, pBufferSize);

    int structural_zero;

    // problem analysis

    //cout << "Problem analysis\n";

    cusparseScsrilu02_analysis(handle, rows, nnz, descr_M, d_csrVal, d_csrRowPtr, d_csrColInd, info_C, CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer);
    cusparseStatus_t status2 = cusparseXcsrilu02_zeroPivot(handle, info_C, &structural_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status2) { printf("A(%d,%d) is missing\n", structural_zero, structural_zero); }

    cusparseScsrsv2_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, rows, nnz, descr_L,
        d_csrVal, d_csrRowPtr, d_csrColInd,
        info_L, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer);

    cusparseScsrsv2_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, rows, nnz, descr_U,
        d_csrVal, d_csrRowPtr, d_csrColInd,
        info_U, CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer);

    // M = L * U

    //cout << "M = L * U\n";

    int numerical_zero;
    cusparseScsrilu02(handle, rows, nnz, descr_M, d_csrVal, d_csrRowPtr, d_csrColInd, info_C, CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer);
    status2 = cusparseXcsrilu02_zeroPivot(handle, info_C, &numerical_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status2) { printf("U(%d,%d) is zero\n", numerical_zero, numerical_zero); }

    cudaEventRecord(stop3);

    cudaEventSynchronize(stop3);
    float ilu_milli = 0;
    cudaEventElapsedTime(&ilu_milli, start3, stop3);

    cout << "Regular ILU factorization done\n time: " << ilu_milli << endl;
    outfile << "Regular ILU time: " << ilu_milli << endl;
}

// Auxiliar matrices of 1's created by reordering

// elements per row = 1, so csrRowPtr[i] = i
__global__ void setRowIndices(int* d_B_RowIndices, const int N) {

    const int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid == N)       d_B_RowIndices[tid] = N;
    else if (tid < N)   d_B_RowIndices[tid] = tid;

}

// elements = 1.0f, so csrVal[i] = 1.0f;
__global__ void setB(float* d_B, const int N) {

    const int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid < N)    d_B[tid] = 1.0f;

}

// ILU factorization using graph coloring for reordering
void ilu_with_reordering(int rows, int nnz, float*& d_csrVal, int*& d_csrRowPtr, int*& d_csrColInd, ofstream& outfile) {
    cudaEvent_t start3, stop3;
    cudaEventCreate(&start3);
    cudaEventCreate(&stop3);

    cudaEventRecord(start3);

    // Matrix B: matrix of 1's

    int* d_coloring, * d_csrColIndB;
    int* d_csrRowPtrB;
    float* d_csrValB;

    const int BLOCKSIZE = 256;

    // graph coloring
    cusparse_graph_coloring(rows, nnz, d_csrVal, d_csrRowPtr, d_csrColInd, d_coloring, d_csrColIndB, outfile, true);

    // storing coloring array and reordering array at host
    int* h_coloring = (int*)malloc(rows * sizeof(int));
    int* h_csrColIndB = (int*)malloc(rows * sizeof(int));
    cudaMemcpy(h_coloring, d_coloring, rows * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_csrColIndB, d_csrColIndB, rows * sizeof(double), cudaMemcpyDeviceToHost);

    // creating reordering matrix B

    cudaMalloc((void**)&d_csrRowPtrB, (rows + 1) * sizeof(int));
    int* h_csrRowPtrB = (int*)malloc((rows + 1) * sizeof(double));
    setRowIndices << <iDivUp(rows + 1, BLOCKSIZE), BLOCKSIZE >> > (d_csrRowPtrB, rows); //d_csrRowPtrB[i] = i

    cudaMalloc((void**)&d_csrValB, rows * sizeof(float));
    float* h_csrValB = (float*)malloc(rows * sizeof(float));
    setB << <iDivUp(rows, BLOCKSIZE), BLOCKSIZE >> > (d_csrValB, rows); // init csrValB with 1.0f

    cudaMemcpy(h_csrValB, d_csrValB, rows * sizeof(float), cudaMemcpyDeviceToHost);

    // ILU factorization

    int* d_csrColIndC;
    int* d_csrRowPtrC;
    float* d_csrValC;

    cusparseMatDescr_t descrA;
    cusparseCreateMatDescr(&descrA);

    cusparseMatDescr_t descrB; cusparseCreateMatDescr(&descrB);
    /*status = cusparseCreateMatDescr(&descr);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        printf("error!");
        exit(1);
    }*/

    cusparseMatDescr_t descrC = 0; // new matrix

    // --- Descriptor for sparse matrix C
    cusparseCreateMatDescr(&descrC);
    cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ZERO);

    cusparseMatDescr_t descr_M = 0; // non-used
    cusparseMatDescr_t descr_L = 0;
    cusparseMatDescr_t descr_U = 0;
    //csrilu02Info_t info_M = 0;
    csrsv2Info_t  info_L = 0;
    csrsv2Info_t  info_U = 0;

    cusparseStatus_t status;
    cusparseHandle_t handle;
    status = cusparseCreate(&handle);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        printf("error!");
        exit(1);
    }

    status = cusparseCreateMatDescr(&descr_M);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        printf("error!");
        exit(1);
    }

    // L and U matrix descriptors

    cout << "Creating matrix descriptors\n";

    cusparseCreateMatDescr(&descr_L);
    cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER);
    cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_UNIT);

    cusparseCreateMatDescr(&descr_U);
    cusparseSetMatIndexBase(descr_U, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descr_U, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatFillMode(descr_U, CUSPARSE_FILL_MODE_UPPER);
    cusparseSetMatDiagType(descr_U, CUSPARSE_DIAG_TYPE_NON_UNIT);

    csrilu02Info_t info_C = 0; cusparseCreateCsrilu02Info(&info_C);
    cusparseCreateCsrsv2Info(&info_L);
    cusparseCreateCsrsv2Info(&info_U);

    // store at matrix C the reordered matrix

    //cudaMalloc((void**)&d_csrRowPtrC, (rows + 1) * sizeof(int));

    // --- Performing the matrix - matrix multiplication
    int nnzB = nnz;
    int nnzC;
    int baseC;
    int* nnzTotalDevHostPtr = &nnzC;

    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);

    cusparseXcsrgemmNnz(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, rows, rows, rows, descrB, nnzB,
        d_csrRowPtrB, d_csrColIndB, descrA, nnz, d_csrRowPtr, d_csrColInd, descrC, d_csrRowPtrC,
        nnzTotalDevHostPtr); // Find sparsity pattern
    if (NULL != nnzTotalDevHostPtr) nnzC = *nnzTotalDevHostPtr;
    else {
        cudaMemcpy(&nnzC, d_csrRowPtrC + rows, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&baseC, d_csrRowPtrC, sizeof(int), cudaMemcpyDeviceToHost);
        nnzC -= baseC;
    }
    cudaMalloc((void**)&d_csrColIndC, nnzC * sizeof(int));
    cudaMalloc((void**)&d_csrValC, nnzC * sizeof(float));
    double* h_C = (double*)malloc(nnzC * sizeof(float));
    int* h_C_ColIndices = (int*)malloc(nnzC * sizeof(int));
    cusparseScsrgemm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, rows, rows, rows, descrB, nnzB,
        d_csrValB, d_csrRowPtrB, d_csrColIndB, descrA, nnz, d_csrVal, d_csrRowPtr, d_csrColInd, descrC,
        d_csrValC, d_csrRowPtrC, d_csrColIndC); // Multiplication involving transposed matrices

    // buffer size (START OF ILU)

    int pBufferSize_M, pBufferSize_L, pBufferSize_U;
    cusparseScsrilu02_bufferSize(handle, rows, nnzC, descrC, d_csrValC, d_csrRowPtrC, d_csrColIndC, info_C, &pBufferSize_M);

    cusparseScsrsv2_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, rows, nnzC,
        descr_L, d_csrValC, d_csrRowPtrC, d_csrColIndC, info_L, &pBufferSize_L);
    cusparseScsrsv2_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, rows, nnzC,
        descr_U, d_csrValC, d_csrRowPtrC, d_csrColIndC, info_U, &pBufferSize_U);

    int pBufferSize = max(pBufferSize_M, max(pBufferSize_L, pBufferSize_U));
    void* pBuffer = 0; cudaMalloc((void**)&pBuffer, pBufferSize);

    int structural_zero;

    // problem analysis

    cusparseScsrilu02_analysis(handle, rows, nnzC, descrC, d_csrValC, d_csrRowPtrC, d_csrColIndC, info_C, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer);
    cusparseStatus_t status2 = cusparseXcsrilu02_zeroPivot(handle, info_C, &structural_zero);
    //if (CUSPARSE_STATUS_ZERO_PIVOT == status2) { printf("A(%d,%d) is missing\n", structural_zero, structural_zero); }
    
    cusparseScsrsv2_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, rows, nnzC, descr_L,
        d_csrValC, d_csrRowPtrC, d_csrColIndC,
        info_L, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer);

    cusparseScsrsv2_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, rows, nnzC, descr_U,
        d_csrValC, d_csrRowPtrC, d_csrColIndC,
        info_U, CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer);

    // M = L * U

    //cout << "M = L * U\n";

    int numerical_zero;
    cusparseScsrilu02(handle, rows, nnzC, descrC, d_csrValC, d_csrRowPtrC, d_csrColIndC, info_C, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer);
    status2 = cusparseXcsrilu02_zeroPivot(handle, info_C, &numerical_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status2) { printf("U(%d,%d) is zero\n", numerical_zero, numerical_zero); }

    cudaEventRecord(stop3);

    cudaEventSynchronize(stop3);
    float ilu_milli = 0;
    cudaEventElapsedTime(&ilu_milli, start3, stop3);

    cout << "Reordered ILU factorization done\n time: " << ilu_milli << endl;
    outfile << "Reordered ILU time: " << ilu_milli << endl;
}

void do_operation(char* mat, int operation, ofstream& outfile, bool seed_for_gc = true) { // seed_for_gc -> false, use cpu cycles as seed

    // operation
    // 0 -> graph coloring with randoms
    // 1 -> csrcolor graph coloring
    // 2 -> ilu factorization using levels
    // 3 -> ilu factorization using graph coloring

    float* csrVal;
    int* csrRowPtr;
    int* csrColInd;

    int nnz, rows, cols;
    read_mat(mat, csrVal, csrRowPtr, csrColInd, cols, rows, nnz);
    cout << "Rows cols nnz " << rows << " " << cols << " " << nnz << endl;

    outfile << "Matrix " << mat << endl;

    int* d_csrRowPtr, * d_csrColInd;
    float* d_csrVal;

    // Separating space on GPU for matrix
    cudaMalloc((void**)&d_csrRowPtr, (rows + 1) * sizeof(int));
    cudaMalloc((void**)&d_csrColInd, nnz * sizeof(int));
    cudaMalloc((void**)&d_csrVal, nnz * sizeof(float));

    // Sending matrix info to GPU
    cudaMemcpy(d_csrRowPtr, csrRowPtr, (rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrColInd, csrColInd, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrVal, csrVal, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    int* d_coloring, * d_reordering;
    
    if (operation == 0) graph_coloring(rows, nnz, d_csrVal, d_csrRowPtr, d_csrColInd, outfile, seed_for_gc);
    else if (operation == 1) cusparse_graph_coloring(rows, nnz, d_csrVal, d_csrRowPtr, d_csrColInd, d_coloring, d_reordering, outfile);
    else if (operation == 2) regular_ilu(rows, nnz, d_csrVal, d_csrRowPtr, d_csrColInd, outfile);
    else if (operation == 3) ilu_with_reordering(rows, nnz, d_csrVal, d_csrRowPtr, d_csrColInd, outfile);

    delete[] csrVal;
    delete[] csrRowPtr;
    delete[] csrColInd;

    cudaFree(d_csrRowPtr);
    cudaFree(d_csrColInd);
    cudaFree(d_csrVal);
}

int main()
{

    ofstream outfile;

    outfile.open("results.txt", std::ios_base::app);


    for (int i = 0; i <= 3; i++) {
        do_operation("Matrices/offshore.mtx", i, outfile);
        do_operation("Matrices/af_shell3.mtx", i, outfile);
        do_operation("Matrices/parabolic_fem.mtx", i, outfile);
        do_operation("Matrices/apache2.mtx", i, outfile);
        do_operation("Matrices/ecology2.mtx", i, outfile);
        do_operation("Matrices/thermal2.mtx", i, outfile);
        do_operation("Matrices/G3_circuit.mtx", i, outfile);
    }

    outfile << "Using CPU cycles as seed (JPL)\n";

    do_operation("Matrices/offshore.mtx", 0, outfile, false);
    do_operation("Matrices/af_shell3.mtx", 0, outfile, false);
    do_operation("Matrices/parabolic_fem.mtx", 0, outfile, false);
    do_operation("Matrices/apache2.mtx", 0, outfile, false);
    do_operation("Matrices/ecology2.mtx", 0, outfile, false);
    do_operation("Matrices/thermal2.mtx", 0, outfile, false);
    do_operation("Matrices/G3_circuit.mtx", 0, outfile, false);

    outfile.close();
}