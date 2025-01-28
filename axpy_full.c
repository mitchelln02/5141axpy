#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/timeb.h>
#include <pthread.h>

#define REAL float
#define VECTOR_LENGTH 102400

typedef struct {
    int start;       // Starting index of the chunk
    int end;         // Ending index of the chunk
    REAL *Y;         // Pointer to the Y array
    REAL *X;         // Pointer to the X array
    REAL a;          // Scalar value 'a'
} ThreadData;

/* read timer in second */
double read_timer() {
    struct timeb tm;
    ftime(&tm);
    return (double) tm.time + (double) tm.millitm / 1000.0;
}

/* read timer in ms */
double read_timer_ms() {
    struct timeb tm;
    ftime(&tm);
    return (double) tm.time * 1000.0 + (double) tm.millitm;
}

/* initialize a vector with random floating point numbers */
void init(REAL A[], int N) {
    int i;
    for (i = 0; i < N; i++) {
        A[i] = (double) drand48();
    }
}

/* Thread function to compute a portion of AXPY */
void *axpy_thread(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    for (int i = data->start; i < data->end; ++i) {
        data->Y[i] += data->a * data->X[i];
    }
    pthread_exit(NULL);
}

/* PThread-based parallel AXPY kernel */
void axpy_kernel_threading(int N, REAL *Y, REAL *X, REAL a, int num_threads) {
    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];

    // Determine chunk size
    int chunk_size = (N + num_threads - 1) / num_threads;  // Ensure all elements are covered

    for (int t = 0; t < num_threads; ++t) {
        // Calculate the start and end indices for each thread
        thread_data[t].start = t * chunk_size;
        thread_data[t].end = (t + 1) * chunk_size;
        if (thread_data[t].end > N) {
            thread_data[t].end = N;  // Ensure no overflow
        }
        thread_data[t].Y = Y;
        thread_data[t].X = X;
        thread_data[t].a = a;

        // Create thread
        if (pthread_create(&threads[t], NULL, axpy_thread, &thread_data[t]) != 0) {
            fprintf(stderr, "Error creating thread %d\n", t);
            exit(1);
        }
    }

    // Wait for all threads to finish
    for (int t = 0; t < num_threads; ++t) {
        if (pthread_join(threads[t], NULL) != 0) {
            fprintf(stderr, "Error joining thread %d\n", t);
            exit(1);
        }
    }
}

/* Serial AXPY kernel */
void axpy_kernel(int N, REAL *Y, REAL *X, REAL a) {
    int i;
    for (i = 0; i < N; ++i)
        Y[i] += a * X[i];
}

int main(int argc, char *argv[]) {
    int N = VECTOR_LENGTH;
    int num_threads = 4;

    if (argc < 2) {
        fprintf(stderr, "Usage: axpy <n> [<num_threads>] (The default of n is %d, the default num_threads is %d)\n", N, num_threads);
    } else if (argc == 2) {
        N = atoi(argv[1]);
    } else {
        N = atoi(argv[1]);
        num_threads = atoi(argv[2]);
    }

    REAL *X = (REAL *)malloc(sizeof(REAL) * N);
    REAL *Y = (REAL *)malloc(sizeof(REAL) * N);

    srand48((1 << 12));
    init(X, N);
    init(Y, N);

    int num_runs = 10;
    int i;
    REAL a = 0.1234;

    /* Example run */
    double elapsed; /* for timing */
    elapsed = read_timer();
    for (i = 0; i < num_runs; i++) axpy_kernel(N, Y, X, a);
    elapsed = (read_timer() - elapsed) / num_runs;

    double elapsed2; /* for timing */
    elapsed2 = read_timer();
    for (i = 0; i < num_runs; i++) axpy_kernel_threading(N, Y, X, a, num_threads);
    elapsed2 = (read_timer() - elapsed2) / num_runs;

    printf("====================================================================================\n");
    printf("\tAXPY %d numbers, serial and threading\n", N);
    printf("------------------------------------------------------------------------------------\n");
    printf("Performance:\t\tRuntime (ms)\t MFLOPS \n");
    printf("------------------------------------------------------------------------------------\n");
    printf("AXPY-serial:\t\t%4f\t%4f\n", elapsed * 1.0e3, 2 * N / (1.0e6 * elapsed));
    printf("AXPY-%d threads:\t\t%4f\t%4f\n", num_threads, elapsed2 * 1.0e3, 2 * N / (1.0e6 * elapsed2));

    free(X);
    free(Y);

    return 0;
}
