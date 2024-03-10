#include "graph_builder.h"

#include <iostream>
#include <fstream>

void fill_bipartite_graph_with_random_weights(int *graph, int m, int n, double density, int fill_in_empty, int empty, int weight_cap){

    int index = 0;
    double cond;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {

            cond = ((double) rand()) / RAND_MAX ;

            if (cond <= density) {

                graph[index] = rand() % weight_cap + 1;

            } else if (fill_in_empty) {

                graph[index] = empty;

            }

            index++;

        }
    }

}


void swap(int *array, int i, int j){
    int temp = array[i];
    array[i] = array[j];
    array[j] = temp;
}


int *get_permuation_array(int n){

    int *permuation = (int *) malloc(n*sizeof(int));

    // fill in the array
    for (int i = 0; i < n; i++) {
        permuation[i] = i;
    }

    // random scrambling
    for (int i = 0; i < n; i++) {
        int j = rand() % n;
        swap(permuation, i, j);
    }

    return permuation;
}

int *build_int_bipartite_graph(int m, int n, double density, int include_matching, int weight_cap, int empty){

    int *graph = (int *) calloc(m*n,sizeof(int));
    
    if (empty != 0) {
        
        fill_in_matrix_with(graph, m, n, empty);
        
    }

    if (include_matching) {

        int *match;

        if (m >= n) {

            match = get_permuation_array(m);

            for (int j = 0; j < n; j++) {
                graph[match[j]*n + j] = rand() % weight_cap + 1;
            }

        } else {

            match = get_permuation_array(n);

            for (int i = 0; i < m; i++) {
                graph[i*n+match[i]] = rand() % weight_cap + 1;
            }
        }

        free(match);
    }

    fill_bipartite_graph_with_random_weights(graph, m, n, density, !include_matching, empty, weight_cap);

    return graph;
}

void fill_in_matrix_with(int *matrix, int rows, int cols, int value){

    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
        {
            matrix[row*cols + col] = value;
        }
        
    }

}

void print_matrix(int * matrix, int m, int n){
    int index = 0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%d ", matrix[index++]);
        }
        printf("\n");
    }
}

int *get_adjacency_matrix(int *graph, int m, int n, int null_value){
    int side_len = m+n;
    // i prima delle j
    int * adj = (int *) calloc(side_len*side_len, sizeof(int));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++){
            adj[(m+j)*side_len+i] = adj[i*side_len+(m+j)] = (int) graph[i*n + j];
        }
    }
    return adj;
}

void store_bipartite_test_case(int *graph, int m, int n, const char* file_name) {
    
    // Open the file for writing
    FILE* file = fopen(file_name, "w");
    if (file == NULL) {
        printf("Error: unable to open file for writing\n");
        return;
    }

    // Write the number of people and objects to the file
    fprintf(file, "%d %d\n", m, n);

    // Write the graph array to the file
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            fprintf(file, "%d ", graph[i * n + j]);
        }
        fprintf(file, "\n");
    }

    // Close the file
    fclose(file);
}

void store_bipartite_test_case_binary(int *graph, int m, int n, const char* file_name) {
 
    // Open the file for writing binary data
    FILE* file = fopen(file_name, "wb");
    if (file == NULL) {
        printf("Error: unable to open file for writing\n");
        return;
    }

    // Write the number of people and objects to the file
    fwrite(&m, sizeof(int), 1, file);
    fwrite(&n, sizeof(int), 1, file);

    // Write the graph array to the file
    fwrite(graph, sizeof(int), m*n, file);

    // Close the file
    fclose(file);
}

loaded_graph load_bipartite_test_case(const char* file_name) {
    // Open the file for reading
    FILE* file = fopen(file_name, "r");
    if (file == NULL) {
        printf("Error: unable to open file for reading\n");
        exit(-1);
    }

    loaded_graph res;

    // Read the number of people and objects from the file
    fscanf(file, "%d %d", &res.m, &res.n);


    // Allocate memory for the graph
    res.graph = (int*) malloc(res.m * res.n * sizeof(int));

    if (res.graph == NULL) {
        printf("Error: unable to allocate memory for the graph\n");
        fclose(file);
        exit(-1);
    }

    // Read the graph from the file
    for (int i = 0; i < res.m; i++) {
        for (int j = 0; j < res.n; j++) {
            fscanf(file, "%d", &(res.graph[i * res.n + j]));
        }
    }

    // Close the file
    fclose(file);

    return res;
}

loaded_graph load_bipartite_test_case_binary(const char* file_name) {
    // Open the file for reading binary data
    FILE* file = fopen(file_name, "rb");
    if (file == NULL) {
        printf("Error: unable to open file for reading\n");
        exit(-1);
    }

    loaded_graph res;

    // Read the number of workers and jobs from the file
    fread(&res.m, sizeof(int), 1, file);
    fread(&res.n, sizeof(int), 1, file);

    // Allocate memory for the graph
    res.graph = (int*) malloc(res.m * res.n * sizeof(int));

    if (res.graph == NULL) {
        printf("Error: unable to allocate memory for the graph\n");
        fclose(file);
        exit(-1);
    }

    // Read the graph from the file
    fread(res.graph, sizeof(int), res.m * res.n, file);

    // Close the file
    fclose(file);

    return res;
}


