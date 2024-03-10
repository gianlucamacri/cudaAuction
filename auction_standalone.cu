#include "common.h"
#include "auction_cpu.h"
#include "auction_gpu.h"
#include "graph_builder.h"
#include <time.h>

#define LIMIT 10000 // or RAND_MAX

#define GRAPH_OUTPUT_BIN_FN "graph.bin"
#define GRAPH_OUTPUT_FN "graph.txt"


#define AUCTION_OUTPUT_FN "auction_result.txt"

const char *usage = "\
Usage: %s filename eps\n\
   or  %s -generate m n d eps [seed]\n\
\n\
in the first case a binary file where content has the following structure is expected:\n\
m n benefit matrix of associating i person with object j, row majour order\n\
\n\
with the -generate flag a new graph is generated (using an optional seed) and saved to %s\n\
m is the number of people to be matched, n the nuber of objects and d the probability of having a connection between 2 elements with positive weight\n\
the existence of a matching is guaranteed by construction\n\
eps is the epsilon value used to run the auction\n\
\n\
in both cases the resulting matching is saved to %s\n\0";

int compute_matching_score(int *matching, int *benefits, int person_num, int obj_num){
    int score = 0;
    for (int person = 0; person < person_num; person++) {

        score += benefits[person*obj_num + matching[person]];
    }

    return score;

}

void write_matching_to_file(int * matching, int people_num, int obj_num, const char *output_fn, const char *input_fn, const char *mode, double eps, int score){
    // Open the file for writing
    FILE* file = fopen(output_fn, "w");
    if (file == NULL) {
        printf("Error: unable to open file for writing\n");
        return;
    }

    // Write the number of people and objects to the file
    fprintf(file, "executed on %s with epsilon %lf\ninput file: %s\nmatch score: %d\npeople number: %d\nobject number: %d\nmatching:\n",mode,eps, input_fn ,score, people_num, obj_num);

    // Write the graph array to the file
    for (int i = 0; i < people_num; i++) {
        fprintf(file, "%d %d\n", i, matching[i]);
    }

    // Close the file
    fclose(file);
}



int main (int argc, char** argv) {

    if (argc < 3) {
        printf(usage,
                argv[0],argv[0],GRAPH_OUTPUT_FN,AUCTION_OUTPUT_FN);
        exit(0);
    }

    int *graph;
    int person_num;
    int obj_num;
    double epsilon;
    const char *input_filename = argv[1];

    if (argc == 3) {

        if (sscanf(argv[2], "%lf", &epsilon) != 1){
            fprintf(stderr, "Error reading epsilon\n");
            exit(1);
        }

        if (epsilon <=0 ){
            fprintf(stderr, "Error epsilon must be grater than 0\n");
            exit(1);
        }

        printf("rading file %s\n", input_filename);

        loaded_graph lg = load_bipartite_test_case_binary(input_filename);
        
        graph = lg.graph;
        person_num = lg.m;
        obj_num = lg.n;

        printf("graph with %d people and %d objects loaded\n", person_num, obj_num);
        
    } else if (argc <= 7) {

        input_filename = GRAPH_OUTPUT_FN;

        if (strncmp(argv[1],"-generate\0",9) != 0){
            fprintf(stderr, "Error: invalid flag %s\n", argv[1]);
            exit(1);
        }

        if (argc < 6 ) {
            fprintf(stderr, "Error: missing one or more parameters\n");
            exit(1);
        }

        unsigned int seed = time(NULL);
        double density;

        person_num = atoi(argv[2]);
        obj_num = atoi(argv[3]);

        if (person_num < 2 || obj_num < 2 || obj_num < person_num){
            fprintf(stderr, "Error m and n should be both grater than 1 and n should be less than or equal to m\n");
            exit(1);
        }

        if (sscanf(argv[4], "%lf", &density) != 1){
            fprintf(stderr, "Error reading density\n");
            exit(1);
        }

        if (density <=0 || density > 1){
            fprintf(stderr, "Error density must be between 0 excluded and 1\n");
            exit(1);
        }

        if (sscanf(argv[5], "%lf", &epsilon) != 1){
            fprintf(stderr, "Error reading epsilon\n");
            exit(1);
        }

        if (epsilon <=0 ){
            fprintf(stderr, "Error epsilon must be grater than 0\n");
            exit(1);
        }

        if (argc == 7) {
            seed = atoi(argv[6]);
        }
        
        srand(seed);

        printf("generating new graph using seed %u\n", seed);

        graph = build_int_bipartite_graph(person_num, obj_num, density, 1, LIMIT, INT_MIN);

        printf("storing graph to %s and %s\n", GRAPH_OUTPUT_BIN_FN, GRAPH_OUTPUT_FN);

        store_bipartite_test_case_binary(graph, person_num,obj_num, GRAPH_OUTPUT_BIN_FN);
        store_bipartite_test_case(graph, person_num,obj_num, GRAPH_OUTPUT_FN);

    }

    const char* mode;

    #ifdef CPU
    mode = "cpu";
    #else
        #ifdef DETERMINISM
            #ifdef USE_CUDA_GRAPH
            mode = "gpu in deterministic mode using cuda graphs";
            #else
            mode = "gpu in deterministic mode";
            #endif
        #else 
            #ifdef USE_CUDA_GRAPH
            mode = "gpu using cuda graphs";
            #else
            mode = "gpu";
            #endif
        #endif
    #endif
    printf("running auction algorithm on %s\n", mode);

    #ifdef CPU
    int *matching = auction_cpu(graph, person_num, obj_num, epsilon);
    #else
    int *matching = auction_gpu(graph, person_num, obj_num, epsilon);
    #endif

    int score = compute_matching_score(matching, graph, person_num, obj_num);

    write_matching_to_file(matching, person_num, obj_num, AUCTION_OUTPUT_FN, input_filename, mode, epsilon, score);

    free(matching);
    free(graph);

    return 0;

}



