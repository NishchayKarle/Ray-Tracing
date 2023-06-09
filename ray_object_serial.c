#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

// struct to represent a 3d Vector
struct vector
{
    double x,
        y,
        z;
};
typedef struct vector vector;

double **create_2d_array(int grid_size)
{
    double **grid = malloc(sizeof(double *) * grid_size);
    for (int i = 0; i < grid_size; i++)
        grid[i] = malloc(sizeof(double) * grid_size);

    for (int i = 0; i < grid_size; i++)
        for (int j = 0; j < grid_size; j++)
            grid[i][j] = 0.0;

    return grid;
}

double random_cos_values()
{
    return ((double)rand() / (double)RAND_MAX) * 2.0 - 1.0;
}

double random_phi_values()
{
    return ((double)rand() / (double)RAND_MAX) * 2.0 * M_PI;
}

void vector_addition(vector const *A, vector const *B, vector *res)
{
    res->x = A->x + B->x;
    res->y = A->y + B->y;
    res->z = A->z + B->z;
}

void vector_subtraction(vector const *A, vector const *B, vector *res)
{
    res->x = A->x - B->x;
    res->y = A->y - B->y;
    res->z = A->z - B->z;
}

void scalar_multiplication(vector const *A, double *scalar, vector *res)
{
    res->x = A->x * (*scalar);
    res->y = A->y * (*scalar);
    res->z = A->z * (*scalar);
}

void scalar_division(vector const *A, double *scalar, vector *res)
{
    res->x = A->x / (*scalar);
    res->y = A->y / (*scalar);
    res->z = A->z / (*scalar);
}

void vector_dot_product(vector const *A, vector const *B, double *res)
{
    *res = (A->x * B->x + A->y * B->y + A->z * B->z);
}

void vector_magnitude(vector const *A, double *res)
{
    *res = sqrt(
        A->x * A->x +
        A->y * A->y +
        A->z * A->z);
}

void unit_normal_vector(vector const *A, double *magnitude, vector *res)
{
    scalar_division(A, magnitude, res);
}

void destroy_grid(double **G, int grid_size)
{
    for (int i = 0; i < grid_size; i++)
    {
        free(G[i]);
        G[i] = NULL;
    }
    free(G);
    G = NULL;
    return;
}

void write_to_file(FILE *fp, int N, double **arr)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            fprintf(fp, "%f, ", arr[i][j]);
        }
        fprintf(fp, "\n");
    }
}

void ray_tracing(double **G, int grid_size, long int N_rays,
                 vector *L, vector *C, double W_y,
                 double W_max, double R)
{
    vector *W = malloc(sizeof(vector)),
           *V = malloc(sizeof(vector)),
           *I = malloc(sizeof(vector)),
           *IminusC = malloc(sizeof(vector)),
           *N = malloc(sizeof(vector)),
           *LminusI = malloc(sizeof(vector)),
           *S = malloc(sizeof(vector));

    for (long int n = 0; n < N_rays; n++)
    {
        double C_C, V_C, t, mag, S_N, b, W_V;

        int i, j;
        vector_dot_product(C, C, &C_C);

        do
        {
            // SAMPLING V
            double phi = random_phi_values(),
                   cos_theta = random_cos_values(),
                   sin_theta = sqrt(1 - cos_theta * cos_theta),
                   cos_phi = cos(phi),
                   sin_phi = sin(phi);

            V->x = sin_theta * cos_phi,
            V->y = sin_theta * sin_phi,
            V->z = cos_theta;

            W_V = W_y / V->y;

            scalar_multiplication(V, &W_V, W);
            vector_dot_product(V, C, &V_C);
        } while (!(fabs(W->x) < W_max &&
                   fabs(W->z) < W_max &&
                   V_C * V_C + R * R - C_C > 0));

        vector_dot_product(V, C, &V_C);

        t = V_C - sqrt(V_C * V_C + R * R - C_C);

        // INTERSECTION OF VIEW RAY AND SPHERE
        scalar_multiplication(V, &t, I);

        vector_subtraction(I, C, IminusC);
        vector_magnitude(IminusC, &mag);

        // UNIT NORMAL VECTOR AT I
        unit_normal_vector(IminusC, &mag, N);

        vector_subtraction(L, I, LminusI);
        vector_magnitude(LminusI, &mag);

        // DIRECTION OF LIGHT SOURCE
        unit_normal_vector(LminusI, &mag, S);

        // BRIGHTNESS OBSERVED AT I
        vector_dot_product(S, N, &S_N);
        b = fmax(0.0, S_N);

        // FIND ij
        i = grid_size - (grid_size) * (W->x + W_max) / (2 * W_max);
        j = (grid_size) * (W->z + W_max) / (2 * W_max);

        // ADD BRIGHTNESS TO GRID POINT
        G[i][j] += b;
    }

    free(W);
    free(V);
    free(I);
    free(IminusC);
    free(N);
    free(LminusI);
    free(S);

    return;
}

int main(int argc, char **argv)
{
    srand(time(NULL));

    int grid_size = atoi(argv[1]);
    long int N_rays = atoi(argv[2]);

    vector L = {4, 4, -1},
           C = {0, 12, 0};

    double W_y = 10,
           W_max = 10,
           R = 6;

    double **G = create_2d_array(grid_size);

    double start = omp_get_wtime();
    ray_tracing(G, grid_size, N_rays, &L, &C, W_y, W_max, R);
    double end = omp_get_wtime();
    printf("time elapsed: %lf(s)\n", end - start);

    // WRITE GRID TO A FILE
    FILE *fp1 = fopen("output.txt", "w");
    write_to_file(fp1, grid_size, G);

    destroy_grid(G, grid_size);

    return EXIT_SUCCESS;
}