#ifndef AUCTION_GPU_H
#define AUCTION_GPU_H

#include "common.h"
#include <math.h>


// #define USE_CUDA_GRAPH

//#define DETERMINISM // flag to match the serial computation: each person will bid on the lowest index object among the most profitable one
//                    // for the current round, similarly each object will be assigned to the lowest index bidder among the highest ones

#define MIN_TWOMAXPAIR (twoMaxPair) { .els = (double2) {.x=-__FLT_MAX__, .y = -__FLT_MAX__} , .max_idx = -1 }

#define MIN_MAXPAIR (maxPair) { .max = -__DBL_MAX__, .max_idx = -1 }

#define MAX_THREADS_PER_BLOCK 1024

#define WARPSIZEEXP 5 // WARPSIZE should be 2^WARPSIZEEXP
#define WARPSIZE 32

#define WARP_REDUCTION_SHARED (MAX_THREADS_PER_BLOCK / WARPSIZE)

#define FULLMASK 0xffffffff // need to be coherent with WARPSIZE


// kernel to queue missing unmatched bidders for the next round
// it will add to unmatched_next_round all and only the bidders of current_unmatched that did not get assigned to any object in the current round
// previously matched people left unassigned by the assignment phase have already been added with assignmentKernel
__global__ void updateUnmatchedKernel(int *person2obj_matching, int *current_unmatched, int *unmatched_next_round, int people_number);


typedef struct twoMaxPair {
  double2 els; // els.x will be the max and els.y the second max
  int max_idx;
} twoMaxPair;


// performs max and second max reduction considering also the index of the max on a warp, is affected by DETERMINISM flag
__inline__ __device__ twoMaxPair warpReduceTwoMaxBestIndex(twoMaxPair p);


// performs max and second max reduction considering also the index of the max on a block, is affected by DETERMINISM flag
__inline__ __device__ twoMaxPair blockReduceTwoMaxBestIndex(twoMaxPair val);


typedef struct maxPair {
  double max;
  int max_idx;
} maxPair;

// performs max reduction with relative index on a warp, is affected by DETERMINISM flag
__inline__ __device__ maxPair warpReduceMaxBestIndex(maxPair p);


// performs max reduction with relative index on a block, is affected by DETERMINISM flag
__inline__ __device__ maxPair blockReduceMaxBestIndex(maxPair val);


// kernel for the bidding phase: for each person in current_unmatched_people_idx_dev we look for the 2 highest profit objects according to current prices and benefits with a parallel reduction strategy
// each person will suggest an increment on the highest profit object equal to the difference between the two highest profits plus epsilon
// modifies accordingly bid_increment_dev and sets a flag indicating the target in obj_received_bid_from_person_dev (this matrix is intended to be used with noramle object and person indices)
// increments the number of bids for the target object in bids_per_object
// objects_with_updated_bid will contain in the first objects_with_updated_bid[obj_number] positions the objects that received at least a bid in the current round
__global__ void biddingKernel(int *benefits, double* prices, double2 *aux, int *max_idx, int aux_row_dim, int people_number, int obj_number, int* current_unmatched_people_idx_dev, double epsilon, double *bid_increment_dev, unsigned int *first_reduction_done, int8_t *obj_received_bid_from_person_dev, int*bids_per_object, int *objects_with_updated_bid);


// kernel for the assignment phase: given the suggested bid increments of the bidding phase, the relative bidding targets obj_received_bid_from_person_dev
// loops over all and only the objects that received at leas a bid in the current round (the first objects_with_updated_bid[obj_number] elements of objects_with_updated_bid)
// and determines the highest bid and bidder that object 
// modifies prces, person2obj_matching and obj2person_matching according to the updated assignment, possibly incrementing the total counter of people matched
// enqueues previously matched people that will be left unassigned by the new matching in next_round_current_unmatched_people_idx_dev
// resets obj_received_bid_from_person_dev and bids_per_object
__global__ void assignmentKernel(double *bid_increment_dev , double *prices, int *person2obj_matching, int* obj2person_matching, int*people_matched, int people_number, int obj_number, int8_t *obj_received_bid_from_person_dev, int *bids_per_object, int *current_unmatched_people_idx_dev, int *objects_with_updated_bid, int *next_round_unmatched_people_idx_dev);


// debug function to verify bidding phase, prints the bid of each person for the current round
// people that have not bidded in the current round will report a bid to the UNASSIGNED object
__host__ __device__ void print_bidding_matrix_debug(int8_t *assignment_matrix, int people_number, int obj_number, double * bid_increment );


// utility to print an assignment, possibly with the final prices of the auctions
void print_assignment(int *person2obj_matching, int people_number, double *prices=NULL);


// initializes host data, specifically the 2 matching arrays, setting each element to UNASSIGNED,
// zeros the prices and the matched people number, and inserts each people in the unmatched queue
void host_initData(int people_number, int obj_number, int *person2obj_matching, int *obj2person_matching, double *prices, int *people_matched, int* unmatched_people_idx);


// cuda initialization and basic assertion verification (in debug mode)
// returns the numebr of streaming multiprocessors of the device
int cuda_init();


// merges multiple cudaMalloc calls into a single one and returns the array with the addresses (allocated with malloc)
// WARNING: use data types all of the same type or with types that have dimension one multiple of the other and assign the 
// resulting addresses in  decreasing order to avoid memory alignment issues (e.g. all the int addresses before the double ones and after the char ones)
void **cuda_merge_mumtiple_malloc(const int n,  unsigned long* bytesizes);


// interface to the main code
// takes the matrix describiding the value each person associated to each object (people_number x obj_number)
// and its dimensions, plus the epsilon value to be used for the algorithm, higher values correspond to quicker but sub-optimal assignments
// return an arry with the matching
int *auction_gpu(int *benefits, int people_number, int obj_number, double epsilon);

#endif