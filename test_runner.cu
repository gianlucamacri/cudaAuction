#include "common.h"
#include "auction_cpu.h"
#include "auction_gpu.h"
#include "graph_builder.h"
#include <time.h>
#include <unistd.h>
#include <assert.h>

#define DELAY 10000 // avoid printing issues

int compute_matching_score(int *matching, int *benefits, int person_num, int obj_num){
    int score = 0;
    for (int person = 0; person < person_num; person++) {

        assert (matching[person] >= 0 && matching[person] < obj_num);

        score += benefits[person*obj_num + matching[person]];
    }

    return score;

}

int are_array_equal(int *array1, int *array2, int array_length){

    for (int i = 0; i < array_length; i++)
    {
        if (array1[i] != array2[i]) {
            return 0;
        }
    }

    return 1;

}

double2 get_mean_and_sd(float *values, int values_num, int correct=0){
    double mean, sd, tmp;
    double sum = 0;

    // mean
    for (int i = 0; i < values_num; i++) {
        sum += (double) values[i];
    }

    mean = sum / values_num;

    // standard dev
    for (int i = 0; i < values_num; i++) {   
        tmp = mean - values[i];
        sd += tmp*tmp;
    }

    sd = sqrt(sd / (correct ? values_num - 1 : values_num));
    
    return (double2) {.x = mean, .y = sd};
    
}


int main (int argc, char** argv) {

    if (argc < 4) {
        printf("Usage: %s eps repetition_number fn_1 [fn2_ ... fn_n]\nexecutes both the cpu and gpu version of the auction algorithm (or gpu only if GPUONLY flag is specified) for each of the provided file names, executing each repetition_number times using the choosen epsilon value (double)\n",
                argv[0]);
        exit(0);
    }

    int repetition_number, file_number, person_num, obj_num,
        *graph, *gpu_matching;

    double epsilon;
    char *filename;
    loaded_graph lg;
    double2 time_results_gpu;

    #ifndef GPUONLY
    int cpu_matching_score, gpu_matching_score,
        *cpu_matching;

    double2 time_results_cpu;

    #endif


    if (sscanf(argv[1],"%lf", &epsilon)!=1) {
        fprintf(stderr,"Error parsing eps\n");
        exit(1);
    }

    if (epsilon <= 0) {
        fprintf(stderr,"Error eps must be a positive value\n");
        exit(1);
    }

    if (sscanf(argv[2],"%d", &repetition_number)!=1) {
        fprintf(stderr,"Error parsing repetition number\n");
        exit(1);
    }

    if (repetition_number <= 0) {
        fprintf(stderr,"Error d must be a positive value\n");
        exit(1);
    }

    file_number = argc - 3;

    for (int i = 3; i < argc; i++){
        if (access(argv[i], F_OK) != 0) {
            fprintf(stderr,"Error: file %s does not exist or cannot be opened\n", argv[i]);
            exit(1);
        }
    }

    float *gpu_execution_times = (float *) calloc(file_number*repetition_number,sizeof(float)); 
    #ifndef GPUONLY
    float *cpu_execution_times = (float *) calloc(file_number*repetition_number,sizeof(float));
    #endif 

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    int iteration_counter = 0;
    int file_counter = 0;

    for (int i = 3; i < argc; i++){
        filename = argv[i];
        printf("reading %s\n", filename);
        lg = load_bipartite_test_case_binary(filename);
        printf("graph loaded\n");
        graph = lg.graph;
        person_num = lg.m;
        obj_num = lg.n;

        #ifndef DETERMINISM
        double epsilon_optimality_threshold = 1.0/person_num;
        #endif

        for(int rep = 0; rep < repetition_number; rep++){
            
            // gpu
            cudaEventRecord(start);

            gpu_matching = auction_gpu(graph, person_num, obj_num, epsilon);

            cudaEventRecord(end);
            cudaEventSynchronize(end);

            cudaEventElapsedTime(&gpu_execution_times[iteration_counter], start, end);

            printf("gpu exec time: %f ms\n\n", gpu_execution_times[iteration_counter] );

            fflush(stdout); // force stdout write

            usleep(DELAY);
            
            #ifndef GPUONLY
            // cpu
            cudaEventRecord(start);
            
            cpu_matching = auction_cpu(graph, person_num, obj_num, epsilon);

            cudaEventRecord(end);
            cudaEventSynchronize(end);

            cudaEventElapsedTime(&cpu_execution_times[iteration_counter], start, end);

            printf("cpu exec time: %f ms\n\n", cpu_execution_times[iteration_counter] );

            fflush(stdout); // force stdout write

            usleep(DELAY);
            
            cpu_matching_score = compute_matching_score(cpu_matching, graph, person_num, obj_num);

            gpu_matching_score = compute_matching_score(gpu_matching, graph, person_num, obj_num);

            if (rep == 0) { // correctness check

                #ifdef DETERMINISM
                if(!are_array_equal(gpu_matching, cpu_matching, person_num)){
                    fprintf(stderr,"warning: DETERMINISM flag is set but the resulting matching differ!\ngpu matching score: %d\ncpu matching score: %d\n",
                            gpu_matching_score,
                            cpu_matching_score);
                }
                #else
                // with small enough eps the algorithm should reach optimal value, despite the determinism flag
                if ( epsilon < epsilon_optimality_threshold && gpu_matching_score != cpu_matching_score) {
                    fprintf(stderr,"warning: optimality not reached despite epsilon being small enough.!\ngpu matching score: %d\ncpu matching score: %d\n",
                            gpu_matching_score,
                            cpu_matching_score);
                }
                #endif

                fflush(stderr);
                
            }

            free(cpu_matching);
            #endif

            free(gpu_matching);


            fflush(stdout); // force stdout write

            iteration_counter+=1;
        }

        // compute times for the current file
        time_results_gpu = get_mean_and_sd(&gpu_execution_times[file_counter*repetition_number], repetition_number);
        #ifndef GPUONLY
        time_results_cpu = get_mean_and_sd(&cpu_execution_times[file_counter*repetition_number], repetition_number);
        #endif

        printf("file results:\ngpu mean time: %lf (sd %lf)\n",time_results_gpu.x,time_results_gpu.y);
        #ifndef GPUONLY
        printf("cpu mean time: %lf (sd %lf)\n",time_results_cpu.x, time_results_cpu.y);
        #endif

        printf("\n");

        file_counter += 1;
        
        free(graph);
    }

    // final results
    time_results_gpu = get_mean_and_sd(gpu_execution_times, file_number*repetition_number);
    #ifndef GPUONLY
    time_results_cpu = get_mean_and_sd(cpu_execution_times, file_number*repetition_number);
    #endif

    printf("global results:\ngpu mean time: %lf (sd %lf)\n",time_results_gpu.x,time_results_gpu.y);
    #ifndef GPUONLY
    printf("cpu mean time: %lf (sd %lf)\n",time_results_cpu.x, time_results_cpu.y);

    free(cpu_execution_times);
    #endif
    free(gpu_execution_times);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    return 0;

}



