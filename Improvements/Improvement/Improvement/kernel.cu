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

#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>

//#include <curand.h>

#include <random>
// random number generator
#include <intrin.h>

#pragma intrinsic(__rdtsc)

using namespace std;


// https://stackoverflow.com/questions/24069524/error-using-ldg-in-cuda-kernel-at-compile-time/24073775
template<typename T>
__device__ __forceinline__ T ldg(const T* ptr) {
#if __CUDA_ARCH__ >= 350
    return __ldg(ptr);
#else
    return *ptr;
#endif
}

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

__global__ void color_jpl_kernel_improved(int n, int c, const int* Ao,
    const int* Ac, const float* Av,
    const int* randoms, int* colors)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x;
        i < n;
        i += blockDim.x * gridDim.x)
    {
        bool f = true; // true iff you have max random

        // ignore nodes colored earlier

        int cur_color = ldg(&colors[i]);

        if ((cur_color != -1)) continue;
        int ir = ldg(&randoms[i]);

        // look at neighbors to check their random number
        int start = ldg(&Ao[i]);
        int end = ldg(&Ao[i + 1]);
        for (int k = start; k < end; k++) {
            // ignore nodes colored earlier (and yourself)
            int j = ldg(&Ac[k]);
            int jc = ldg(&colors[j]);
            if (((jc != -1) && (jc != c)) || (i == j)) continue;
            int jr = ldg(&randoms[j]);
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

void color_jpl_improved(int n,
    const int* Ao, const int* Ac, const float* Av,
    int* colors, int* d_randoms)
{

    int* d_colors;
    cudaMalloc((void**)&d_colors, n * sizeof(int));
    //cudaMemcpy(d_colors, colors, n * sizeof(int), cudaMemcpyHostToDevice);

    thrust::fill(thrust::device, d_colors, d_colors + n, -1); // init colors to -1

    cout << "initiallized random numbers and colors\n";

    //cout << "nodes left: " << (int)thrust::count(colors, colors + n, -1) << endl;
    for (int c = 0; c < n; c++) {
        int nt = 256;
        int nb = (ceil(n / nt));
        color_jpl_kernel_improved << <nb, nt >> > (n, c,
            Ao, Ac, Av,
            d_randoms,
            d_colors);
        cudaDeviceSynchronize();
        int left = (int)thrust::count(thrust::device, d_colors, d_colors+n, -1);
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
void graph_coloring(int rows, int nnz, float*& d_csrVal, int*& d_csrRowPtr, int*& d_csrColInd, ofstream& outfile, bool seed_time = true, bool improved = false) {

    int* colors = new int[rows];
    cout << "JPL algorithm time\n";

    int* randoms; // allocate and init random array 
    randoms = new int[rows];
    
    int* d_randoms;

    cudaMalloc((void**)&d_randoms, rows * sizeof(int));

    cout << "Initializing random values\n";
    init_rand_array(randoms, rows, seed_time);
    /*
    * Usage of curand
    * Didn't give better results
    else {
        curandGenerator_t gen;
        unsigned int* d_randoms2;

        // Allocate n floats on device
        cudaMalloc((void**)&d_randoms2, rows * sizeof(int));

        // Create pseudo-random number generator 
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);

        // Set seed
        curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

        // Generate n floats on device
        curandGenerate(gen, d_randoms2, rows);

        //  Cleanup
        //curandDestroyGenerator(gen);
        //cudaMemcpy(randoms, d_randoms2, rows * sizeof(int), cudaMemcpyDeviceToHost);
    }
    */
    cout << "Rand values initialized\n";

    cudaMemcpy(d_randoms, randoms, rows * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);

    cudaEventRecord(start2);
    if (!improved) color_jpl(rows, d_csrRowPtr, d_csrColInd, d_csrVal, colors, d_randoms);
    else color_jpl_improved(rows, d_csrRowPtr, d_csrColInd, d_csrVal, colors, d_randoms);
    //color_jpl(rows, d_csrRowPtr, d_csrColInd, d_csrVal, colors, d_randoms);
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

    if (improved) outfile << "IMPROVED JPL" << endl;
    outfile << "JPL operation " << jpl_milli << endl;
    outfile << "Colors: " << jpl_colors << endl;

    delete[] colors;
}

#define	MAXCOLOR 128

__global__ void initialize(int* coloring, bool* colored, int m) {
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < m) {
        coloring[id] = MAXCOLOR;
        colored[id] = false;
    }
}

__global__ void firstFit(int rows, int* csrRowPtr, int* csrColInd, int* coloring, bool* changed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    bool forbiddenColors[MAXCOLOR + 1];
    if (coloring[id] == MAXCOLOR) {
        for (int i = 0; i < MAXCOLOR; i++)
            forbiddenColors[i] = false;
        int row_begin = csrRowPtr[id];
        int row_end = csrRowPtr[id + 1];
        for (int offset = row_begin; offset < row_end; offset++) {
            int neighbor = csrColInd[offset];
            int color = coloring[neighbor];
            forbiddenColors[color] = true;
        }
        int vertex_color;
        for (vertex_color = 0; vertex_color < MAXCOLOR; vertex_color++) {
            if (!forbiddenColors[vertex_color]) {
                coloring[id] = vertex_color;
                break;
            }
        }
        *changed = true;
    }
}

__global__ void conflictResolve(int rows, int* csrRowPtr, int* csrColInd, int* coloring, bool* colored) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (!colored[id]) {
        int row_begin = csrRowPtr[id];
        int row_end = csrRowPtr[id + 1];
        int offset;
        for (offset = row_begin; offset < row_end; offset++) {
            int neighbor = csrColInd[offset];
            if (coloring[id] == coloring[neighbor] && id < neighbor) {
                coloring[id] = MAXCOLOR;
                break;
            }
        }
        if (offset == row_end)
            colored[id] = true;
    }
}

void ffit_color(int rows, int nnz, int* d_csrRowPtr, int* d_csrColInd, int* coloring, int blksz, ofstream& outfile) {
    int* colors;
    int* d_coloring;
    bool* changed, hchanged;
    bool* d_colored;

    cudaMalloc((void**)&d_coloring, rows * sizeof(int));
    cudaMalloc((void**)&d_colored, rows * sizeof(int));

    cudaMalloc((void**)&changed, sizeof(bool));
    initialize << <((rows - 1) / blksz + 1), blksz >> > (d_coloring, d_colored, rows);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    do {
        hchanged = false;
        cudaMemcpy(changed, &hchanged, sizeof(hchanged), cudaMemcpyHostToDevice);
        int nblocks = (rows - 1) / blksz + 1;
        firstFit << <nblocks, blksz >> > (rows, d_csrRowPtr, d_csrColInd, d_coloring, changed);
        conflictResolve << <nblocks, blksz >> > (rows, d_csrRowPtr, d_csrColInd, d_coloring, d_colored);
        cudaDeviceSynchronize();
        cudaMemcpy(&hchanged, changed, sizeof(hchanged), cudaMemcpyDeviceToHost);
    } while (hchanged);
    cudaDeviceSynchronize();
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ffit_milli = 0;
    cudaEventElapsedTime(&ffit_milli, start, stop);

    cout << "Milliseconds for operation (ffit): " << ffit_milli << endl;

    colors = new int[rows];

    cudaMemcpy(colors, d_coloring, rows * sizeof(int), cudaMemcpyDeviceToHost);

    int ffit_colors = 0;
    for (int i = 0; i < rows; i++) {
        if (colors[i] > ffit_colors) {
            ffit_colors = colors[i];
        }
    }

    cout << "Colors: " << ffit_colors << endl;

    outfile << "Ffit operation " << ffit_milli << endl;
    outfile << "Colors: " << ffit_colors << endl;

    delete[] colors;
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
    else if (operation == 2) graph_coloring(rows, nnz, d_csrVal, d_csrRowPtr, d_csrColInd, outfile, seed_for_gc, true);
    else if (operation == 3) ffit_color(rows, nnz, d_csrRowPtr, d_csrColInd, d_coloring, 256, outfile);

    delete[] csrVal;
    delete[] csrRowPtr;
    delete[] csrColInd;

    cudaFree(d_csrRowPtr);
    cudaFree(d_csrColInd);
    cudaFree(d_csrVal);
}

int main()
{

    char* matrices[] = { "Matrices/offshore.mtx", "Matrices/af_shell3.mtx" ,
        "Matrices/parabolic_fem.mtx", "Matrices/apache2.mtx", "Matrices/ecology2.mtx",
        "Matrices/thermal2.mtx", "Matrices/G3_circuit.mtx"};

    ofstream outfile;

    outfile.open("results.txt", std::ios_base::app);

    for (int matrix = 0; matrix < 7; matrix++) {
        for (int i = 0; i <= 3; i++) {
            /*do_operation("Matrices/offshore.mtx", i, outfile, false);
            do_operation("Matrices/af_shell3.mtx", i, outfile, false);
            do_operation("Matrices/parabolic_fem.mtx", i, outfile, false);
            do_operation("Matrices/apache2.mtx", i, outfile, false);
            do_operation("Matrices/ecology2.mtx", i, outfile, false);
            do_operation("Matrices/thermal2.mtx", i, outfile, false);
            do_operation("Matrices/G3_circuit.mtx", i, outfile, false);*/
            do_operation(matrices[matrix], i, outfile, false);
        }
    }

    /*do_operation("Matrices/offshore.mtx", 0, outfile, false);
    do_operation("Matrices/af_shell3.mtx", 0, outfile, false);
    do_operation("Matrices/parabolic_fem.mtx", 0, outfile, false);
    do_operation("Matrices/apache2.mtx", 0, outfile, false);
    do_operation("Matrices/ecology2.mtx", 0, outfile, false);
    do_operation("Matrices/thermal2.mtx", 0, outfile, false);
    do_operation("Matrices/G3_circuit.mtx", 0, outfile, false);

    do_operation("Matrices/offshore.mtx", 2, outfile, false);
    do_operation("Matrices/af_shell3.mtx", 2, outfile, false);
    do_operation("Matrices/parabolic_fem.mtx", 2, outfile, false);
    do_operation("Matrices/apache2.mtx", 2, outfile, false);
    do_operation("Matrices/ecology2.mtx", 2, outfile, false);
    do_operation("Matrices/thermal2.mtx", 2, outfile, false);
    do_operation("Matrices/G3_circuit.mtx", 2, outfile, false);

    do_operation("Matrices/offshore.mtx", 3, outfile, false);
    do_operation("Matrices/af_shell3.mtx", 3, outfile, false);
    do_operation("Matrices/parabolic_fem.mtx", 3, outfile, false);
    do_operation("Matrices/apache2.mtx", 3, outfile, false);
    do_operation("Matrices/ecology2.mtx", 3, outfile, false);
    do_operation("Matrices/thermal2.mtx", 3, outfile, false);
    do_operation("Matrices/G3_circuit.mtx", 3, outfile, false);*/

    outfile.close();
}