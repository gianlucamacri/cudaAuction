#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include "graph_builder.h"


#define FILENAME_MAX_LEN 256

#define ROW_BUFFER 256

//#define USE_TXT_FORMAT // slower

const char *config_file_format =
"for k examples\n\
SEED [LIMIT]\n\
m_0 n_0 density_0\n\
...\n\
m_k n_k density_k";


int main(int argc, char *argv[]) {

    if (argc < 2) {
        printf("Usage: %s config_file [save_dir_path]\nconfig file format:\n%s\n", argv[0], config_file_format);
        exit(0);
    }

    const char* config_filename = argv[1];
    const char *dir_path = ".";
    if (argc > 2) {
        dir_path = argv[2];
    }

    FILE *config_file = fopen(config_filename, "r");
    if (config_file == NULL) {
        printf("Error opening config file.\n");
        return 1;
    }

    unsigned int seed;
    int limit, m, n, case_num = 0;
    float density;
    
    char buf[ROW_BUFFER];

    if (fgets(buf, ROW_BUFFER, config_file) == NULL) {
        printf("Error reading config file.\n");
        fclose(config_file);
        return 1;
    }

    int res = sscanf(buf, "%u %d", &seed, &limit);
    if (res < 1) {
        printf("Error reading seed from config file.\n");
        fclose(config_file);
        return 1;
    } else if (res < 2) {
        limit = RAND_MAX;
    }

    struct stat st = {0};

    if (stat(dir_path, &st) == -1) {
        mkdir(dir_path, 0700);
    }

    srand(seed);

    int *graph;
    char file_name[FILENAME_MAX_LEN];

    while(fgets(buf, ROW_BUFFER, config_file) != NULL) {
        res = sscanf(buf, "%d %d %f", &m, &n, &density);
        if (res == 0 ){
            break;
        } else if (res < 3 ) {
            printf("Error reading seed from config file.\n");
            fclose(config_file);
            return 1;
        } 
        case_num++;

        graph = build_int_bipartite_graph(m, n, density, 1, limit, INT_MIN);

        #ifndef USE_TXT_FORMAT
        sprintf(file_name, "%s/testcase_%03d_%d_%d_%2.0f.bin", dir_path, case_num, m, n, 100*density);
        store_bipartite_test_case_binary(graph, m,n, file_name);
        #else
        sprintf(file_name, "%s/testcase_%03d_%d_%d_%2.0f.txt", dir_path, case_num, m, n, 100*density);
        store_bipartite_test_case(graph, m,n, file_name);
        #endif


        free(graph);
    }

    fclose(config_file);
    return 0;
}
