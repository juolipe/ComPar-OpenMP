#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <omp.h>

#define STRONG_N ((size_t)1 << 28)
#define MAX_N ((size_t)1 << 30)
#define REPS_DEFAULT 10

static uint32_t *allocate_vector(size_t n) {
    if (n > SIZE_MAX / sizeof(uint32_t))
        return NULL;

    return (uint32_t *)malloc(n * sizeof(uint32_t));
}

static void initialize_vector(uint32_t *v, size_t n) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) {
        v[i] = (uint32_t)(i % 100) + 1;
    }
}

static uint64_t sum_critical(const uint32_t *v, size_t n, int threads) {
    uint64_t total = 0;

    #pragma omp parallel num_threads(threads) shared(total, v, n)
    {
        uint64_t partial = 0;

        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; i++) {
            partial += v[i];
        }

        #pragma omp critical
        {
            total += partial;
        }
    }

    return total;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr,
                "Uso: %s [strong|weak] [threads 1..6] [repeticoes=10]\n",
                argv[0]);
        return 1;
    }

    int threads = atoi(argv[2]);
    int reps = (argc >= 4) ? atoi(argv[3]) : REPS_DEFAULT;

    if (threads < 1 || threads > 6 || reps < 1) {
        fprintf(stderr,
                "Threads deve estar entre 1 e 6 e repeticoes >= 1.\n");
        return 1;
    }

    size_t n;

    if (strcmp(argv[1], "strong") == 0) {
        n = STRONG_N;
    } else if (strcmp(argv[1], "weak") == 0) {
        n = STRONG_N * (size_t)threads;

        if (n > MAX_N)
            n = MAX_N;
    } else {
        fprintf(stderr, "Modo invalido: use strong ou weak.\n");
        return 1;
    }

    printf("Modo: %s | threads: %d | elementos: %zu | repeticoes: %d\n",
           argv[1], threads, n, reps);

    printf("Memoria aproximada do vetor: %.2f GiB\n",
           ((double)n * sizeof(uint32_t)) /
           (1024.0 * 1024.0 * 1024.0));

    uint32_t *v = allocate_vector(n);

    if (!v) {
        fprintf(stderr,
                "Falha ao alocar vetor. Verifique a RAM disponivel.\n");
        return 2;
    }

    initialize_vector(v, n);

    uint64_t expected = 0;

    for (size_t i = 0; i < n; i++) {
        expected += v[i];
    }

    double total_time = 0.0;

    for (int r = 0; r < reps; r++) {
        double start = omp_get_wtime();

        uint64_t result = sum_critical(v, n, threads);

        double elapsed = omp_get_wtime() - start;

        if (result != expected) {
            fprintf(stderr,
                    "ERRO: soma incorreta na repeticao %d: %" PRIu64
                    " (esperado %" PRIu64 ")\n",
                    r + 1, result, expected);

            free(v);
            return 3;
        }

        total_time += elapsed;

        printf("execucao=%d soma=%" PRIu64 " tempo_s=%.9f\n",
               r + 1, result, elapsed);
    }

    printf("MEDIA modo=%s threads=%d n=%zu media_s=%.9f soma=%" PRIu64 "\n",
           argv[1], threads, n, total_time / reps, expected);

    free(v);

    return 0;
}