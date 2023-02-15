#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <assert.h>

#define M_PI 3.14159265358979323846
#define MAX_BLOCKS_PER_DIM 65535
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

struct vector
{
    double x,
        y,
        z;
};
typedef struct vector vector;

__device__ void lcg_random_double(uint64_t *seed, double *res)
{
    const uint64_t m = 9223372036854775808ull;
    const uint64_t a = 2806196910506780709ull;
    const uint64_t c = 1ull;
    *seed = (a * (*seed) + c) % m;
    *res = (double)(*seed) / (double)m;
}

__device__ void fast_forward_lcg(uint64_t seed, uint64_t n, uint64_t *res)
{
    const uint64_t m = 9223372036854775808ull; // 2ˆ63
    uint64_t a = 2806196910506780709ull;
    uint64_t c = 1ull;
    n = n % m;
    uint64_t a_new = 1;
    uint64_t c_new = 0;
    while (n > 0)
    {
        if (n & 1)
        {
            a_new *= a;
            c_new = c_new * a + c;
        }
        c *= (a + 1);
        a *= a;
        n >>= 1;
    }
    *res = (a_new * seed + c_new) % m;
}

__device__ void generate_random_numbers(double *res, uint64_t *seed)
{
    lcg_random_double(seed, res);
}

__device__ void random_cos_values(double *res, uint64_t *seed)
{
    generate_random_numbers(res, seed);
    *res = (*res) * 2.0 - 1.0;
}

__device__ void random_phi_values(double *res, uint64_t *seed)
{
    generate_random_numbers(res, seed);
    *res = (*res) * 2.0 * M_PI;
}

__device__ void vector_addition(vector const *A, vector const *B, vector *res)
{
    res->x = A->x + B->x;
    res->y = A->y + B->y;
    res->z = A->z + B->z;
}

__device__ void vector_subtraction(vector const *A, vector const *B, vector *res)
{
    res->x = A->x - B->x;
    res->y = A->y - B->y;
    res->z = A->z - B->z;
}

__device__ void scalar_multiplication(vector const *A, double *scalar, vector *res)
{
    res->x = A->x * (*scalar);
    res->y = A->y * (*scalar);
    res->z = A->z * (*scalar);
}

__device__ void scalar_division(vector const *A, double *scalar, vector *res)
{
    res->x = A->x / (*scalar);
    res->y = A->y / (*scalar);
    res->z = A->z / (*scalar);
}

__device__ void vector_dot_product(vector const *A, vector const *B, double *res)
{
    *res = (A->x * B->x + A->y * B->y + A->z * B->z);
}

__device__ void vector_magnitude(vector const *A, double *res)
{
    *res = sqrt(
        A->x * A->x +
        A->y * A->y +
        A->z * A->z);
}

__device__ void unit_normal_vector(vector const *A, double *magnitude, vector *res)
{
    res->x = A->x / (*magnitude);
    res->y = A->y / (*magnitude);
    res->z = A->z / (*magnitude);
}

__global__ void ray_tracing(double *G, int grid_size, int N_rays, vector *L,
                            vector *C, double W_y, double W_max, double R)
{

    vector W, V, I,
        IminusC, N, LminusI, S;

    double C_C, V_C, t, mag, S_N, b, W_V,
        phi, cos_theta, sin_theta, cos_phi, sin_phi;

    int i, j;

    uint64_t N_fast_forward, seed;

    int tid0 = blockIdx.x * blockDim.x + threadIdx.x;

    for (int n = tid0; n < N_rays; n += blockDim.x * gridDim.x)
    {
        vector_dot_product(C, C, &C_C);

        N_fast_forward = 200 * n;
        fast_forward_lcg(NULL, N_fast_forward, &seed);
        do
        {
            // SAMPLING V
            random_phi_values(&phi, &seed);
            random_cos_values(&cos_theta, &seed);
            sin_theta = sqrt(1 - cos_theta * cos_theta);
            cos_phi = cos(phi);
            sin_phi = sin(phi);

            V.x = sin_theta * cos_phi,
            V.y = sin_theta * sin_phi,
            V.z = cos_theta;

            W_V = W_y / V.y;

            scalar_multiplication(&V, &W_V, &W);
            vector_dot_product(&V, C, &V_C);

            seed++;
        } while (!(fabs(W.x) < W_max &&
                   fabs(W.z) < W_max &&
                   V_C * V_C + R * R - C_C > 0));

        vector_dot_product(&V, C, &V_C);

        t = V_C - sqrt(V_C * V_C + R * R - C_C);

        // INTERSECTION OF VIEW RAY AND SPHERE
        scalar_multiplication(&V, &t, &I);

        // UNIT NORMAL VECTOR AT I
        vector_subtraction(&I, C, &IminusC);
        vector_magnitude(&IminusC, &mag);
        unit_normal_vector(&IminusC, &mag, &N);

        // DIRECTION OF LIGHT SOURCE
        vector_subtraction(L, &I, &LminusI);
        vector_magnitude(&LminusI, &mag);
        unit_normal_vector(&LminusI, &mag, &S);

        // BRIGHTNESS OBSERVED AT I
        vector_dot_product(&S, &N, &S_N);
        b = fmax(0.0, S_N);

        // FIND ij
        i = grid_size - (grid_size) * (W.x + W_max) / (2 * W_max);
        j = (grid_size) * (W.z + W_max) / (2 * W_max);

        // ADD BRIGHTNESS TO GRID POINT
        // G[i]"[j] += b;
        atomicAdd(&G[i * grid_size + j], b);
    }

    return;
}

__host__ void write_to_file(FILE *fp, int N, double *arr)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            fprintf(fp, "%f, ", arr[i * N + j]);
        }
        fprintf(fp, "\n");
    }
}

int main(int argc, char **argv)
{
    int grid_size = atoi(argv[1]);
    long int N_rays = atoi(argv[2]);

    int nthreads_per_block = atoi(argv[3]), nblocks;
    nblocks = MIN(N_rays / nthreads_per_block + 1, MAX_BLOCKS_PER_DIM);
    printf("BLOCKS: %d THREADS PER BLOCK: %d\n", nblocks, nthreads_per_block);

    vector L_h = {4, 4, -1},
           C_h = {0, 12, 0};

    double W_y = 10,
           W_max = 10,
           R = 6;

    vector *L, *C;
    assert(cudaMalloc((void **)&L, sizeof(vector)) == cudaSuccess);
    assert(cudaMalloc((void **)&C, sizeof(vector)) == cudaSuccess);
    assert(cudaMemcpy(L, &L_h, sizeof(vector), cudaMemcpyHostToDevice) == cudaSuccess);
    assert(cudaMemcpy(C, &C_h, sizeof(vector), cudaMemcpyHostToDevice) == cudaSuccess);

    double *G, *G_h;
    // ALLOCATE AND INIT MATRIX ON DEVICE
    assert(cudaMalloc((void **)&G, sizeof(double) * grid_size * grid_size) == cudaSuccess);
    assert(cudaMemset(G, 0, sizeof(double) * grid_size * grid_size) == cudaSuccess);

    {
        cudaEvent_t start, stop; /* timers */
        float time;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, 0);
        ray_tracing<<<nblocks, nthreads_per_block>>>(G, grid_size, N_rays, L, C, W_y, W_max, R);
        cudaDeviceSynchronize();

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        printf("time elapsed: %lf(s)\n", time / 1000.);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // COPY MATRIX FROM DEVICE TO HOST
    G_h = (double *)malloc(sizeof(double) * grid_size * grid_size);
    assert(cudaMemcpy(G_h, G, sizeof(double) * grid_size * grid_size, cudaMemcpyDeviceToHost) == cudaSuccess);

    // WRITE MATRIX TO FILE
    FILE *fp = fopen("output.txt", "w");
    write_to_file(fp, grid_size, G_h);

    // FREE GRIDS
    cudaFree(G);
    cudaFree(L);
    cudaFree(C);
    free(G_h);

    return 0;
}