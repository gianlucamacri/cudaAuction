# tool macros
CC := nvcc
CCFLAGS := -lineinfo
DEBUGFLAG := -D DEBUG
CPUFLAG := -D CPU
TESTCASEGENNAME := test_case_generator
PYTHON := python3

# default rule
default: all

.PHONY: all
all: gpu gpu_deterministic gpu_graph cpu

# gpu
.PHONY: gpu
gpu:
	$(CC) $(CCFLAGS) -D VERBOSE -o gpu_auction auction_standalone.cu auction_gpu.cu auction_cpu.cu graph_builder.cu  

.PHONY: gpu_deterministic
gpu_deterministic:
	$(CC) $(CCFLAGS) -D VERBOSE -D DETERMINISM -o gpu_auction_deterministic auction_standalone.cu auction_gpu.cu auction_cpu.cu graph_builder.cu  

.PHONY: gpu_graph
gpu_graph:
	$(CC) $(CCFLAGS) -D VERBOSE -D USE_CUDA_GRAPH -o gpu_auction_graph auction_standalone.cu auction_gpu.cu auction_cpu.cu graph_builder.cu  


.PHONY: debug_gpu
debug_gpu:
	$(CC) $(CCFLAGS) $(DEBUGFLAG) -o gpu_auction auction_standalone.cu auction_gpu.cu auction_cpu.cu graph_builder.cu  

.PHONY: debug_gpu_deterministic
debug_gpu_deterministic:
	$(CC) $(CCFLAGS) $(DEBUGFLAG) -D DETERMINISM -o gpu_auction_deterministic auction_standalone.cu auction_gpu.cu auction_cpu.cu graph_builder.cu  

.PHONY: debug_gpu_graph
debug_gpu_graph:
	$(CC) $(CCFLAGS) $(DEBUGFLAG) -D USE_CUDA_GRAPH -o gpu_auction_graph auction_standalone.cu auction_gpu.cu auction_cpu.cu graph_builder.cu  

# cpu
.PHONY: cpu
cpu:
	$(CC) $(CPUFLAG) -D VERBOSE  $(CCFLAGS) -o cpu_auction auction_standalone.cu auction_gpu.cu auction_cpu.cu graph_builder.cu 


.PHONY: debug_cpu
debug_cpu:
	$(CC) $(CCFLAGS) $(CPUFLAG) $(DEBUGFLAG) -o cpu_auction auction_standalone.cu auction_gpu.cu auction_cpu.cu graph_builder.cu 


# testing realted commands
.PHONY: testcases_generator
testcases_generator:
	$(CC) $(CCFLAGS) -o $(TESTCASEGENNAME) test_case_generator.cu graph_builder.cu

# WARNING will use about 16 GB of disk space
.PHONY: generate_testcases
generate_testcases: testcases_generator
	mkdir -p testcases
	./$(TESTCASEGENNAME) configFiles/config_dense.txt testcases/dense
	./$(TESTCASEGENNAME) configFiles/config_sparse.txt testcases/sparse
# used for auxiliary tests
#	./$(TESTCASEGENNAME) configFiles/config_semi_sparse_reduced.txt testcases/semisparse

.PHONY: clean_testcases
clean_testcases:
	rm -r testcases/dense testcases/sprse testcases/smeisparse

.PHONY: deterministic_test_runner
deterministic_test_runner:
	$(CC) -D DETERMINISM -o deterministic_test_runner test_runner.cu graph_builder.cu auction_cpu.cu auction_gpu.cu

.PHONY: test_runner
test_runner:
	$(CC) -o test_runner test_runner.cu graph_builder.cu auction_cpu.cu auction_gpu.cu

.PHONY: graph_test_runner
graph_test_runner:
	$(CC) -D USE_CUDA_GRAPH -o graph_test_runner test_runner.cu graph_builder.cu auction_cpu.cu auction_gpu.cu

.PHONY: run_tests
run_tests: deterministic_test_runner test_runner graph_test_runner #generate_testcases
	$(PYTHON) perform_tests.py --eps 1.0 --output_fn results_deterministic.json ./deterministic_test_runner 3 5 testcases/dense/ testcases/sparse/
	$(PYTHON) perform_tests.py --eps 1.0 --output_fn results.json ./test_runner 3 5 testcases/dense/ testcases/sparse/
	$(PYTHON) perform_tests.py --eps 1.0 --output_fn results_graph.json ./graph_test_runner 3 5 testcases/dense/ testcases/sparse/
	$(PYTHON) perform_tests.py --output_fn new_results_small_eps.json ./test_runner 3 5 testcases/dense/ testcases/sparse/

## preliminary tests: to run them modify the parameters in the source code and complile accodingly named executables
#	$(PYTHON) perform_tests.py --eps 1.0 --gpu_only --output_fn results_semisparse_64.json ./det_test_runner_64 3 2 testcases/semisparse/
#	$(PYTHON) perform_tests.py --eps 1.0 --gpu_only --output_fn results_semisparse_256.json ./det_test_runner_256 3 3 testcases/semisparse/
#	$(PYTHON) perform_tests.py --eps 1.0 --gpu_only --output_fn results_semisparse_1024.json ./det_test_runner_1024 3 3 testcases/semisparse/

#	$(PYTHON) perform_tests.py --eps 1.0 --gpu_only --output_fn results_dense_128_128_1.json ./graph_test_runner_128_128_1 3 2 testcases/semisparse/
#	$(PYTHON) perform_tests.py --eps 1.0 --gpu_only --output_fn results_dense_128_256_1.json ./graph_test_runner_128_256_1 3 2 testcases/semisparse/
#	$(PYTHON) perform_tests.py --eps 1.0 --gpu_only --output_fn results_dense_256_128_1.json ./graph_test_runner_256_128_1 3 2 testcases/semisparse/
#	$(PYTHON) perform_tests.py --eps 1.0 --gpu_only --output_fn results_dense_256_256_1.json ./graph_test_runner_256_256_1 3 2 testcases/semisparse/
#	$(PYTHON) perform_tests.py --eps 1.0 --gpu_only --output_fn results_dense_128_128_2.json ./graph_test_runner_128_128_2 3 2 testcases/semisparse/
#	$(PYTHON) perform_tests.py --eps 1.0 --gpu_only --output_fn results_dense_128_256_2.json ./graph_test_runner_128_256_2 3 2 testcases/semisparse/
#	$(PYTHON) perform_tests.py --eps 1.0 --gpu_only --output_fn results_dense_256_128_2.json ./graph_test_runner_256_128_2 3 2 testcases/semisparse/
#	$(PYTHON) perform_tests.py --eps 1.0 --gpu_only --output_fn results_dense_256_256_2.json ./graph_test_runner_256_256_2 3 2 testcases/semisparse/







