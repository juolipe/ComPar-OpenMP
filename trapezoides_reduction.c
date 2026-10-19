/*
 * Nome: Júlia Pestana
 * Exercício 1 
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

double f(double x);
double Local_trap(double a, double b, int n);

int main(int argc, char* argv[]) {
    double global_result = 0.0;
    double a, b;
    int n;
    int thread_count;

    if (argc < 2) {
        fprintf(stderr, "Uso: %s <numero_de_threads>\n", argv[0]);
        return 1;
    }

    thread_count = (int)strtol(argv[1], NULL, 10);
    if (thread_count < 1) {
        fprintf(stderr, "Erro: o numero de threads deve ser maior que zero.\n");
        return 1;
    }

    printf("Digite a, b e n (limite esquerdo, limite direito e numero de trapezios):\n");
    if (scanf("%lf %lf %d", &a, &b, &n) != 3 || n <= 0) {
        fprintf(stderr, "Erro: entrada invalida. n deve ser maior que zero.\n");
        return 1;
    }

    if (n % thread_count != 0) {
        fprintf(stderr,
                "Erro: nesta versao, n (%d) deve ser divisivel pelo numero de threads (%d).\\n",
                n, thread_count);
        return 1;
    }

    #pragma omp parallel num_threads(thread_count) reduction(+:global_result)
    {
        int my_rank = omp_get_thread_num();
        int team_size = omp_get_num_threads();

        int local_n = n / team_size;
        double local_a = a + my_rank * local_n * ((b - a) / n);
        double local_b = local_a + local_n * ((b - a) / n);

        global_result += Local_trap(local_a, local_b, local_n);
    }

    printf("Com n = %d trapezios, a estimativa\n", n);
    printf("da integral de %f ate %f = %.14e\n", a, b, global_result);

    return 0;
}

double f(double x) {
    return x * x;
}

double Local_trap(double a, double b, int n) {
    double h, x, approx;
    int i;

    if (n <= 0) {
        return 0.0;
    }

    h = (b - a) / n;
    approx = (f(a) + f(b)) / 2.0;

    for (i = 1; i < n; i++) {
        x = a + i * h;
        approx += f(x);
    }

    return approx * h;
}
