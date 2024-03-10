#ifndef GRAPH_BUILDER_H
#define GRAPH_BUILDER_H

#include "common.h"
#include <time.h>

typedef struct{
    int m,n;
    int *graph;
} loaded_graph;

// fills the graph m*n matrix with some non negative weights between 1 a and weight_cap (included) according to denisty probability
// i.e. there will be a weighted arc between 2 of the two set nodes with probability equal to density
// if fill_empty is true, unchoosen connections will be set to empty value
void fill_bipartite_graph_with_random_weights(int *graph, int m, int n, double density, int fill_in_empty, int empty, int weight_cap);

void swap(int *array, int i, int j);

// allocates an n-sized array containing a random permutation of the numbers between 0 and n
int *get_permuation_array(int n);

// creates an m*n natural matrix representing the weights of the connections between two sets of m and n nodes
// in a bipartite graph, where each connection exists (i.e. has positive weight) with a probaility equal to density
// if include_matching is set then the results will containfor sure a matching between the two sets, with all the elemnts
// in the smallest set matched
// weights are capped by weight_cap (default RAND_MAX)
// non existing weights will have "empty" value
int *build_int_bipartite_graph(int m, int n, double density, int include_matching, int weight_cap, int empty);

// fill each position of an integer matrix with the provided value
// matrix should be memorized in row major order
void fill_in_matrix_with(int *matrix, int rows, int cols, int value);

// print a matrix by rows
void print_matrix(int * matrix, int m, int n);

// given a bipartite graph weight matrix, derives the corresponding adjacency matrix contatenating
// the row indices before the column ones
int *get_adjacency_matrix(int *graph, int m, int n, int null_value);

// writes benefits graph to file by rows with the dimensions beforehand
void store_bipartite_test_case(int *graph, int m, int n, const char* file_name);

// writes benefits graph to file by rows with the dimensions beforehand
void store_bipartite_test_case_binary(int *graph, int m, int n, const char* file_name);

// loads benefits graph from given file
// .graph component of the result will need to be freed
loaded_graph load_bipartite_test_case(const char* file_name);

// loads benefits graph from given file
// .graph component of the result will need to be freed
loaded_graph load_bipartite_test_case_binary(const char* file_name);



#endif