#include "auction_gpu.h"


#ifdef DEBUG

static void HandleError(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d.\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}
#define HANDLE_ERROR(err) HandleError(err, __FILE__, __LINE__)

void checkCUDAError(const char *msg, const char *file, int line) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "ERRORE CUDA in %s at line %d: >%s<: >%s<. Exiting.\n", file, line, msg,
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

#ifndef USE_CUDA_GRAPH
#define HANDLE_KERNEL_EXEC(msg, ...) \
  __VA_ARGS__;                       \
  cudaDeviceSynchronize();           \
  checkCUDAError(msg, __FILE__, __LINE__)
#else
#define HANDLE_KERNEL_EXEC(msg, ...) \
  __VA_ARGS__;                       \
  checkCUDAError(msg, __FILE__, __LINE__)
#endif


#define HANDLE_KERNEL_EXEC_NO_SYNCH(msg, ...) \
  __VA_ARGS__;                       \
  checkCUDAError(msg, __FILE__, __LINE__)


#define DEBUG_EXEC(...) __VA_ARGS__


#else

#define HANDLE_ERROR(err) err
#define HANDLE_KERNEL_EXEC(msg, ...) __VA_ARGS__
#define HANDLE_KERNEL_EXEC_NO_SYNCH(msg, ...) __VA_ARGS__
#define DEBUG_EXEC(...)

// disable assertions
#define NDEBUG

#endif

#include<assert.h>


__global__ void updateUnmatchedKernel(int *person2obj_matching, int *current_unmatched, int *unmatched_next_round, int people_number){
  int person;
  int limit = current_unmatched[people_number];
  int *counter = unmatched_next_round+people_number;

  for (int person_pos = blockIdx.x * blockDim.x + threadIdx.x; person_pos < limit; person_pos+= gridDim.x * blockDim.x){ // grid stride
    
    person = current_unmatched[person_pos];
    
    if (person2obj_matching[person] == UNASSIGNED){
      unmatched_next_round[atomicAdd(counter, 1)] = person;
    }
  }
}


__inline__ __device__ twoMaxPair warpReduceTwoMaxBestIndex(twoMaxPair p) {
  double max_candidate, snd_max_candidate;
  max_candidate = -__DBL_MAX__;
  snd_max_candidate = -__DBL_MAX__;
  int max_candidate_idx = -1;

  #pragma unroll
  for (int offset =1; offset< WARPSIZE; offset <<= 1){

    max_candidate =  __shfl_xor_sync(FULLMASK, p.els.x, offset); // implicit sync
    snd_max_candidate =  __shfl_xor_sync(FULLMASK, p.els.y, offset);
    max_candidate_idx =  __shfl_xor_sync(FULLMASK, p.max_idx, offset);
    
    if (max_candidate > p.els.x) {
      p.els.y = max(p.els.x, snd_max_candidate);
      p.els.x = max_candidate;
      p.max_idx = max_candidate_idx;
    }
    #ifdef DETERMINISM
      else if (max_candidate >= p.els.y) {
      p.els.y = max_candidate;
      
      if(max_candidate == p.els.x) { // for determinism porpuses, bid on the object with maximum profit and lowest id
        p.max_idx = min(p.max_idx, max_candidate_idx);
      }
    }
    #else
      else if (max_candidate > p.els.y) {
      p.els.y = max_candidate;
    }
    #endif
    
  }

  return p; // only first thread of each warp will have the correct value
}


__inline__ __device__ twoMaxPair blockReduceTwoMaxBestIndex(twoMaxPair val) {
  //static __shared__ twoMaxPair shared[WARP_REDUCTION_SHARED]; // WARP_REDUCTION_SHARED is block dim / warpSize (static here is unnecessay)
  
  static __shared__ int shared_int[WARP_REDUCTION_SHARED]; // should reduce bank conflicts even though they don't seem a major factor
  static __shared__ int fst_shared_lo[WARP_REDUCTION_SHARED];
  static __shared__ int fst_shared_hi[WARP_REDUCTION_SHARED];
  static __shared__ int snd_shared_lo[WARP_REDUCTION_SHARED];
  static __shared__ int snd_shared_hi[WARP_REDUCTION_SHARED];

  // using 2 array may reduce ever so slightly the number of bank conflicts (to be checked)

  int lane = threadIdx.x % WARPSIZE;
  int wid = threadIdx.x >> WARPSIZEEXP;

  val = warpReduceTwoMaxBestIndex(val);

  //if (lane == 0) shared[wid] = val;
  
  if (lane == 0){
    fst_shared_lo[wid]= __double2loint(val.els.x);
    fst_shared_hi[wid]= __double2hiint(val.els.x);
    snd_shared_lo[wid]= __double2loint(val.els.y);
    snd_shared_hi[wid]= __double2hiint(val.els.y);
    shared_int[wid] = val.max_idx;
  }

  __syncthreads();

  //val = (threadIdx.x < (blockDim.x >> WARPSIZEEXP)) ? shared[lane] : MIN_TWOMAXPAIR; // consider only legit shared values

  if (threadIdx.x < (blockDim.x >> WARPSIZEEXP)){ // consider only legit shared values
    val.els.x = __hiloint2double( fst_shared_hi[lane], fst_shared_lo[lane] );
    val.els.y = __hiloint2double( snd_shared_hi[lane], snd_shared_lo[lane] );
    val.max_idx = shared_int[lane];
  } else {
    val = MIN_TWOMAXPAIR;
  }

  if (wid == 0) val = warpReduceTwoMaxBestIndex(val);

  return val;
}



__inline__ __device__ maxPair warpReduceMaxBestIndex(maxPair p) {
  double max_candidate;
  max_candidate = -__DBL_MAX__;
  int max_candidate_idx = -1;

  #pragma unroll
  for (int offset =1; offset<WARPSIZE; offset <<= 1){

    max_candidate =  __shfl_xor_sync(FULLMASK, p.max, offset);
    max_candidate_idx =  __shfl_xor_sync(FULLMASK, p.max_idx, offset);
    
    if (max_candidate > p.max) {
      p.max = max_candidate;
      p.max_idx = max_candidate_idx;
    }
    #ifdef DETERMINISM
      else if (max_candidate == p.max) { // for determinism porpuses, bid on the object with maximum profit and lowest id
      p.max_idx = min(p.max_idx, max_candidate_idx);
    }
    #endif
    

  }

  return p; // only first thread of each warp will have the correct value
}


__inline__ __device__ maxPair blockReduceMaxBestIndex(maxPair val) {
  //static __shared__ maxPair shared[WARP_REDUCTION_SHARED]; // this is not warpSize but rather block dim / warpSize, may be set dynamically (static here is unnecessay)
  
  static __shared__ int shared_int[WARP_REDUCTION_SHARED];    // should reduce bank conflicts even though they don't seem a major factor
  static __shared__ int max_shared_lo[WARP_REDUCTION_SHARED];
  static __shared__ int max_shared_hi[WARP_REDUCTION_SHARED];

  int lane = threadIdx.x % WARPSIZE;
  int wid = threadIdx.x >> WARPSIZEEXP;

  val = warpReduceMaxBestIndex(val);

  //if (lane == 0) shared[wid] = val;
  
  if (lane == 0){
    max_shared_lo[wid]= __double2loint(val.max);
    max_shared_hi[wid]= __double2hiint(val.max);
    shared_int[wid] = val.max_idx;
  }

  __syncthreads();

  //val = (threadIdx.x < (blockDim.x >> WARPSIZEEXP)) ? shared[lane] : MIN_MAXPAIR; // conosider only legit shared values

  if (threadIdx.x < (blockDim.x >> WARPSIZEEXP)){ // consider only legit shared values
    val.max = __hiloint2double( max_shared_hi[lane], max_shared_lo[lane] );
    val.max_idx = shared_int[lane];
  } else {
    val = MIN_MAXPAIR;
  }
  

  if (wid == 0) val = warpReduceMaxBestIndex(val);

  return val;
}


__global__ void biddingKernel(int *benefits, double* prices, double2 *aux, int *max_idx, int aux_row_dim, int people_number, int obj_number, int* current_unmatched_people_idx_dev, double epsilon, double *bid_increment_dev, unsigned int *first_reduction_done, int8_t *obj_received_bid_from_person_dev, int*bids_per_object, int *objects_with_updated_bid) {

  // shared for prices doesn't seem to help much (not many reads form the same block)
  __shared__ int prevatomic;
  twoMaxPair max_p;
  double profit;
  int benefit;
  int person_row_offset_benefit, person, aux_offset, prev_bids;
  int lastblockDone = gridDim.x-1;
  int unmatched_limit = current_unmatched_people_idx_dev[people_number];
  
  // each y-block-index will be responsible for a different unmatched person
  for (int person_pos = blockIdx.y; person_pos < unmatched_limit; person_pos+= gridDim.y) {

    person = current_unmatched_people_idx_dev[person_pos];
    person_row_offset_benefit = person * obj_number;
    max_p = MIN_TWOMAXPAIR;

    // first parallel reduction stage
    for (int obj = blockIdx.x * blockDim.x + threadIdx.x; obj < obj_number; obj += blockDim.x * gridDim.x) { // grid stride loop over objects
      
      benefit = benefits[person_row_offset_benefit + obj];

      if (benefit != NEGINFTY){
        profit = benefit - prices[obj];
        
        if (profit > max_p.els.x) { 
          max_p.els.y = max_p.els.x;
          max_p.els.x = profit;
          max_p.max_idx = obj;
        }
        #ifdef DETERMINISM 
        else if (profit >= max_p.els.y) { // bid on the object with maximum profit and lowest id
          max_p.els.y = profit;
          if (profit == max_p.els.x){
            max_p.max_idx = min(max_p.max_idx, obj);
          }
        }
        #else
        else if (profit > max_p.els.y) { // bid on an object with maximum profit (order may chaneg due to concurrency of blocks)
          max_p.els.y = profit;
        }
        #endif
      }
    }

    max_p = blockReduceTwoMaxBestIndex(max_p);

    if (threadIdx.x == 0) { // write block results to shared

      aux_offset = person_pos * aux_row_dim + blockIdx.x;

      aux[aux_offset] = max_p.els;
      max_idx[aux_offset] = max_p.max_idx;
      prevatomic = atomicInc(&first_reduction_done[person_pos], lastblockDone);

    }    
  
    __syncthreads();
    
    // second parallel reduction stage
    if (prevatomic == lastblockDone) { // each block has finished the first reduction step for person (single block execution)

      aux_offset = person_pos * aux_row_dim + threadIdx.x;

      assert (blockDim.x >= aux_row_dim); // to avoid block stride over shared

      max_p = (threadIdx.x < gridDim.x) ? (twoMaxPair) {.els =  aux[aux_offset], .max_idx = max_idx[aux_offset]}: MIN_TWOMAXPAIR; // look only at valid values in the array
     
      max_p = blockReduceTwoMaxBestIndex(max_p);

      if (threadIdx.x == 0) { // thread 0 writes final result
        
        bid_increment_dev[person] = max_p.els.x - max_p.els.y + epsilon;                // bid increment for person
        obj_received_bid_from_person_dev[max_p.max_idx*people_number + person] = 1;     // set binary flag between bidder and target (person_pos may be used but the performance improvement is really marginal, while it becomes more challenging to perform the debugging)
        prev_bids = atomicAdd(&bids_per_object[max_p.max_idx],1);                       // increase the number of bids for the target object
        
        if (prev_bids == 0){
          objects_with_updated_bid[atomicAdd(&objects_with_updated_bid[obj_number],1)] = max_p.max_idx; // insert object in array of objects with bids
        }

      }
    }
  }
}


__global__ void assignmentKernel(double *bid_increment_dev , double *prices, int *person2obj_matching, int* obj2person_matching, int*people_matched, int people_number, int obj_number, int8_t *obj_received_bid_from_person_dev, int *bids_per_object, int *current_unmatched_people_idx_dev, int *objects_with_updated_bid, int *next_round_unmatched_people_idx_dev){

  double bid;
  maxPair max_p;
  int obj_row_offset,previous_owner, highest_bidder, bidder, obj;
  int bidding_people_limit = current_unmatched_people_idx_dev[people_number];
  int obj_limit = objects_with_updated_bid[obj_number];

  for (int obj_pos = blockIdx.y; obj_pos < obj_limit; obj_pos+= gridDim.y) { // each y-dim-block will be in charge of a single object
    
    obj = objects_with_updated_bid[obj_pos];

    obj_row_offset = obj*people_number; //in obj received bid from

    max_p = MIN_MAXPAIR;
    
    for (int bidder_pos = blockIdx.x * blockDim.x + threadIdx.x; bidder_pos < bidding_people_limit; bidder_pos += blockDim.x ) {
      bidder = current_unmatched_people_idx_dev[bidder_pos]; 
    
      if (obj_received_bid_from_person_dev[obj_row_offset + bidder]) {

        obj_received_bid_from_person_dev[obj_row_offset + bidder] = 0; // "consume the info" to avoid memset

        bid = bid_increment_dev[bidder]; // we know it is relative to obj
        
        if (bid > max_p.max) { 
          max_p.max = bid;
          max_p.max_idx = bidder;
        
        }
        #ifdef DETERMINISM
          else if (bid == max_p.max) { // assign object to highest bidder with lowest id
          max_p.max_idx = min(max_p.max_idx, bidder);
        }
        #endif
      }

    }

    max_p = blockReduceMaxBestIndex(max_p);

    assert(gridDim.x == 1);
    
    if (threadIdx.x == 0) { // single thread per object

      highest_bidder = max_p.max_idx;
      
      prices[obj] += max_p.max;
      previous_owner = obj2person_matching[obj];
      
      person2obj_matching[highest_bidder] = obj; // update matching
      obj2person_matching[obj] = highest_bidder;    

      bids_per_object[obj] = 0; // reset bid number

      if (previous_owner != UNASSIGNED) { // number of total matches stays the same
        person2obj_matching[previous_owner] = UNASSIGNED;
        next_round_unmatched_people_idx_dev[atomicAdd(&next_round_unmatched_people_idx_dev[people_number],1)] = previous_owner; // eunqueue previous owner for next round
      } else {
        atomicAdd(people_matched, 1);     // increase mumber of total matches
      }

    }      

  }

}


__host__ __device__ void print_bidding_matrix_debug2(int8_t * assignment_matrix, int people_number, int obj_number, double * bid_increment, int *current_unmatched_people_idx ){
  
  int limit = current_unmatched_people_idx[people_number];
  printf("round bidder number : %d\n", limit);

  for (int i = 0; i < limit; i++){
    
    int person = current_unmatched_people_idx[i];

    for (int j = 0; j < obj_number; j++) {
      if (assignment_matrix[j*people_number + i] > 0) {
        printf("person %d increments bids on obj %d by %lf\n", person, j, bid_increment[person]);
        
      }
    }

  }
}

__host__ __device__ void print_bidding_matrix_debug(int8_t * assignment_matrix, int people_number, int obj_number, double * bid_increment, int *current_unmatched_people_idx ){
  
  for (int i = 0; i < people_number; i++){
  
    for (int j = 0; j < obj_number; j++) {
      if (assignment_matrix[j*people_number + i] > 0) {
        printf("person %d increments bids on obj %d by %lf\n", i, j, bid_increment[i]);
        
      }
    }

  }
}


void print_assignment(int *person2obj_matching, int people_number, double *prices){
  for (int i = 0; i < people_number; i++){
    if (prices == NULL) {
      printf("person %d won object %d\n", i, person2obj_matching[i]);
    } else {
      printf("person %d won object %d offering %lf\n", i, person2obj_matching[i], prices[person2obj_matching[i]]);
    }
  }
}


void host_initData(int people_number, int obj_number, int *person2obj_matching, int *obj2person_matching, int *people_matched, int* unmatched_people_idx){
  
  for (int person = 0; person < people_number; person++) {
    person2obj_matching[person] = UNASSIGNED;
    unmatched_people_idx[person] = person;
  }
  unmatched_people_idx[people_number] = people_number;
  
  for (int obj = 0; obj < obj_number; obj++) {
    obj2person_matching[obj] = UNASSIGNED;
  }
  
  *people_matched = 0;

}


int cuda_init(){

  cudaDeviceProp prop;
  HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));

  //HANDLE_ERROR(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1)); // may be usefull on lower cc since shared usage is rather limited
  
  const int sm_count = prop.multiProcessorCount;
  
  assert (prop.warpSize == WARPSIZE);  // device warp size should match WARPSIZE

  assert (WARPSIZE == (2 << (WARPSIZEEXP-1)));  

  return sm_count;

}


void **cuda_merge_mumtiple_malloc(const int n,  unsigned long* bytesizes){
  
  void **addresses = (void **) malloc(sizeof(void *)*n);

  unsigned long total_size = 0;
  void* base_address;
  for (int i = 0; i < n; i++) {
    total_size += bytesizes[i];
  }

  HANDLE_ERROR(cudaMalloc((void **)&base_address, sizeof(char)* total_size));

  char* address = (char *) base_address;
  for (int i = 0; i < n; i++) {
    addresses[i] = (void *) address;
    address += bytesizes[i];
  }

  return addresses;
  
}


int *auction_gpu(int *benefits, int people_number, int obj_number, double epsilon){ //, double epsilon){
  DEBUG_EXEC(printf("====== compiled with debug ======\n"));  

  const int sm_count = cuda_init();

  if (people_number > obj_number) {
    fprintf(stderr, "error: cannot find a complete match when the number of people is lower than the numebr of objects.\n");
    exit(-1);
  }

  if (people_number < 2) {
    fprintf(stderr, "error: both people number and object number must be greater than or equal to 2.\n");
    exit(-1);
  }
  const int aux_row_dim = 32;

  const int sm_multiplier = 1;

  const int block_x_num_bidding = min(sm_count, aux_row_dim); // to avoid strided loop in block reduce
  const int block_y_num_bidding = sm_multiplier*sm_count;
  const int bidding_thread_num = 128;

  const int block_y_num_assignemnt = sm_multiplier*sm_count;
  const int assignment_thread_num = 256;

  const int block_x_update_unmatched = sm_multiplier*sm_count;
  const int upddate_unmatched_thread_num = 32;


  double *prices,                     *prices_dev,                          // price of each object
                                      *bid_increment_dev;                   // bid incremets of each person per round
  
  double2                             *two_max_aux_dev;                     // auxiliary array used in parallel reduction for max and second max

  int                                 *benefits_dev,                        // benefits that each person associated to each object
      *person2obj_matching,           *person2obj_matching_dev,             // matching between person and object (person_num length)
      *obj2person_matching,           *obj2person_matching_dev,             // matching between object and person (obj_num length)
      *people_matched,                *people_matched_dev,                  // number of people matched
                                      *max_idx_aux_dev,                     // auxiliary array used in parallel reduction for index of the max
      *current_unmatched_people_idx,  *current_unmatched_people_idx_dev,    // unmatched people in a round, last element (people_number) contains the number of relevant positions
                                      *next_round_unmatched_people_idx_dev, // unmatched people in the following round
                                      *bid_per_object_dev,                  // number of bids received in a round for each object
                                      *objects_with_updated_bid_dev;        // objects with bids updates in a round, last element (obj_number) contains the number of relevant positions
  

  unsigned int                        *first_reduction_done_dev;            // auxiliary array used in parallel reduction to keep track of block work (unsigned due to atomicInc)

  int8_t                              *obj_received_bid_from_person_dev;    // binary array obj_number * people_number, 1 in i,j if person j bidded for object j in the current
  

  #ifdef DEBUG

  // host copies for debugging popruses
  double *bid_increment,
         *two_max_aux;

  int *bid_target,
      *max_idx_aux,
      *bid_per_object,
      *objects_with_updated_bid,
      *next_round_unmatched_people_idx;

  unsigned int *first_reduction_done;

  int8_t *obj_received_bid_from_person;
      
  bid_increment = (double *) malloc(sizeof(double)*people_number);
  two_max_aux = (double *) malloc(sizeof(double2)*aux_row_dim * people_number);

  bid_target = (int *) malloc(sizeof(int)*people_number);
  max_idx_aux = (int *) malloc(sizeof(int)*aux_row_dim  * people_number);
  bid_per_object = (int *) malloc(sizeof(int)*obj_number);
  objects_with_updated_bid = (int *) malloc(sizeof(int)*(obj_number+1));
  next_round_unmatched_people_idx = (int *)malloc(sizeof(int) *(people_number +1) );


  first_reduction_done = (unsigned int *) malloc(sizeof(int)*people_number);

  obj_received_bid_from_person = (int8_t *)malloc(sizeof(int8_t)*people_number*obj_number);
  #endif
  
  // memory mapping seemed to have worse performances, making more memory tranfers than necessary

  const int address_num = 13;
  unsigned long sizes[address_num] =
    {
       sizeof(double2) * aux_row_dim * people_number ,  // two_max_aux_dev
       obj_number * sizeof(double),                     // prices_dev
       sizeof(double) * people_number ,                 // bid_increment_dev
       people_number*obj_number*sizeof(int) ,           // benefits_dev
       (people_number +1)*sizeof(int),                  // person2obj_matching_dev
       obj_number*sizeof(int) ,                         // obj2person_matching_dev
       sizeof(int) * aux_row_dim  * people_number,      // max_idx_aux_dev
       sizeof(int)  * (people_number +1) ,              // current_unmatched_people_idx_dev
       sizeof(int)  * (people_number +1) ,              // next_round_unmatched_people_idx_dev
       sizeof(int)  * obj_number ,                      // bid_per_object_dev
       sizeof(int)  * (obj_number+1) ,                  // objects_with_updated_bid_dev
       sizeof(unsigned int)  * people_number ,          // first_reduction_done_dev
       sizeof(int8_t)  * obj_number * people_number ,   // obj_received_bid_from_person_dev
    };

  // single cuda malloc, possibly more difficult to find a huge block, but improved efficiency
  void** addresses = cuda_merge_mumtiple_malloc(address_num, sizes);
  
  two_max_aux_dev = (double2 *) addresses[0];
  prices_dev = (double *) addresses[1];
  bid_increment_dev = (double *) addresses[2];
  benefits_dev = (int *) addresses[3];
  person2obj_matching_dev = (int *) addresses[4];
  obj2person_matching_dev = (int *) addresses[5];
  max_idx_aux_dev = (int *) addresses[6];
  current_unmatched_people_idx_dev = (int *) addresses[7];
  next_round_unmatched_people_idx_dev = (int *) addresses[8];
  bid_per_object_dev = (int *) addresses[9];
  objects_with_updated_bid_dev = (int *) addresses[10];
  first_reduction_done_dev = (unsigned int *) addresses[11];
  obj_received_bid_from_person_dev = (int8_t *) addresses[12];


  HANDLE_ERROR(cudaHostAlloc((void **)&person2obj_matching, (people_number +1)*sizeof(int),0)); // pagelocked memory to make transfer faster
  people_matched = &person2obj_matching[people_number];
  
  people_matched_dev = &person2obj_matching_dev[people_number];
  
  #ifdef DEBUG
  int previous_round_matched = *people_matched;
  #endif
  
  obj2person_matching = (int*) malloc(obj_number*sizeof(int));
  current_unmatched_people_idx = (int *)malloc(sizeof(int)  * (people_number +1) );

  prices = (double *) malloc(obj_number*sizeof(double)); // unused

  host_initData(people_number, obj_number, person2obj_matching, obj2person_matching, people_matched, current_unmatched_people_idx);
  
  const int stream_num = 8;
  cudaStream_t stream[stream_num];
  for (int i = 0; i < stream_num; i++) {
    HANDLE_ERROR(cudaStreamCreate(&stream[i]));
  }

  HANDLE_ERROR(cudaMemcpyAsync(benefits_dev,benefits, people_number*obj_number*sizeof(int) , cudaMemcpyHostToDevice , stream[0])); 
  HANDLE_ERROR(cudaMemcpyAsync(obj2person_matching_dev,obj2person_matching, obj_number*sizeof(int) , cudaMemcpyHostToDevice , stream[1])); 
  HANDLE_ERROR(cudaMemcpyAsync(person2obj_matching_dev,person2obj_matching, (people_number+1)*sizeof(int) , cudaMemcpyHostToDevice , stream[2])); 
  HANDLE_ERROR(cudaMemcpyAsync(current_unmatched_people_idx_dev, current_unmatched_people_idx, sizeof(int) * (people_number+1), cudaMemcpyHostToDevice, stream[3]));

  HANDLE_ERROR(cudaMemsetAsync(bid_per_object_dev,0, sizeof(int) * obj_number, stream[4]));
  HANDLE_ERROR(cudaMemsetAsync(objects_with_updated_bid_dev,0, sizeof(int) * (obj_number+1), stream[4]));

  HANDLE_ERROR(cudaMemsetAsync(first_reduction_done_dev, 0,sizeof(unsigned int) * people_number, stream[4]));

  HANDLE_ERROR(cudaMemsetAsync(obj_received_bid_from_person_dev,0, sizeof(int8_t) * obj_number * people_number , stream[4]));

  HANDLE_ERROR(cudaDeviceSynchronize()); // wait termination of all operations
  
  // on recent gpus cc>=8.0 you may consider setting cudaAccessPropertyPersisting for the prices

  // unsing dynamic parallelism would require to synchronize with the children which apparently is considered deprecated (+ seemed slower)
  
  int counter=0;

  const int event_num = 6;
  cudaEvent_t event[event_num];
  for (int i = 0; i < event_num; i++) {
    HANDLE_ERROR(cudaEventCreate(&event[i]));
  }

  cudaEvent_t next_round_unmatched_set = event[0];
  cudaEvent_t assignment_done = event[1];
  cudaEvent_t objects_with_updated_bid_reset = event[2];
  cudaEvent_t unmatched_updated = event[3];
  cudaEvent_t people_matched_copied = event[4];


  #ifdef USE_CUDA_GRAPH
  cudaEvent_t init = event[5];
  const int graph_num = 2;
  cudaGraph_t graphs[graph_num];
  cudaGraphExec_t instances[graph_num];
  int cuda_graph_idx = 0;

  do {
    HANDLE_ERROR(cudaStreamBeginCapture(stream[0],cudaStreamCaptureModeGlobal));

    HANDLE_ERROR(cudaEventRecord(init, stream[0]));
    HANDLE_ERROR(cudaStreamWaitEvent(stream[1], init));

  #else
  
  do { // inner part is the "standard execution", while the external one exploits cuda graphs, it is like this due to the swapping not considered by the record (there may be a more elegant way)

  #endif

    HANDLE_ERROR(cudaMemsetAsync(next_round_unmatched_people_idx_dev+people_number,0, sizeof(int), stream[1]));
    HANDLE_ERROR(cudaEventRecord(next_round_unmatched_set, stream[1]));

    HANDLE_KERNEL_EXEC("bidding",biddingKernel<<<dim3(block_x_num_bidding,block_y_num_bidding,1),dim3(bidding_thread_num,1,1),0, stream[0]>>>(benefits_dev, prices_dev, two_max_aux_dev, max_idx_aux_dev, aux_row_dim, people_number, obj_number, current_unmatched_people_idx_dev, epsilon, bid_increment_dev,first_reduction_done_dev, obj_received_bid_from_person_dev, bid_per_object_dev, objects_with_updated_bid_dev ));

    #ifdef DEBUG
    //HANDLE_ERROR(cudaMemcpy(obj_received_bid_from_person, obj_received_bid_from_person_dev, sizeof(int8_t) * (people_number) * obj_number, cudaMemcpyDeviceToHost));
    //HANDLE_ERROR(cudaMemcpy(bid_increment, bid_increment_dev, sizeof(double) * (people_number), cudaMemcpyDeviceToHost));
    //HANDLE_ERROR(cudaMemcpy(current_unmatched_people_idx, current_unmatched_people_idx_dev, sizeof(int) * (people_number +1), cudaMemcpyDeviceToHost));
    //HANDLE_ERROR(cudaDeviceSynchronize());
    //print_bidding_matrix_debug(obj_received_bid_from_person, people_number,  obj_number, bid_increment, current_unmatched_people_idx );
    #endif


    HANDLE_ERROR(cudaStreamWaitEvent(stream[0], next_round_unmatched_set));
    HANDLE_KERNEL_EXEC("assignment",assignmentKernel<<<dim3(1,block_y_num_assignemnt,1),dim3(assignment_thread_num,1,1),0, stream[0]>>>(bid_increment_dev , prices_dev, person2obj_matching_dev, obj2person_matching_dev, people_matched_dev, people_number, obj_number, obj_received_bid_from_person_dev, bid_per_object_dev, current_unmatched_people_idx_dev, objects_with_updated_bid_dev, next_round_unmatched_people_idx_dev));
    HANDLE_ERROR(cudaEventRecord(assignment_done, stream[0]));

    #ifdef DEBUG
    //HANDLE_ERROR(cudaMemcpy(person2obj_matching,person2obj_matching_dev, (people_number)*sizeof(int) , cudaMemcpyDeviceToHost )); 
    //HANDLE_ERROR(cudaMemcpy(prices, prices_dev, sizeof(double)*obj_number, cudaMemcpyDeviceToHost));
    //HANDLE_ERROR(cudaDeviceSynchronize());
    //print_assignment(person2obj_matching, people_number, prices); // use NULL to avoid printing prices
    #endif
    
    HANDLE_ERROR(cudaStreamWaitEvent(stream[2], assignment_done));
    HANDLE_ERROR(cudaMemsetAsync(objects_with_updated_bid_dev+obj_number,0, sizeof(int), stream[2]));
    HANDLE_ERROR(cudaEventRecord(objects_with_updated_bid_reset, stream[2]));

    HANDLE_ERROR(cudaStreamWaitEvent(stream[3], assignment_done));
    HANDLE_ERROR(cudaMemcpyAsync(people_matched,people_matched_dev, sizeof(int) , cudaMemcpyDeviceToHost, stream[3]));
    HANDLE_ERROR(cudaEventRecord(people_matched_copied, stream[3]));

    HANDLE_KERNEL_EXEC("update person2obj_matching",updateUnmatchedKernel<<<block_x_update_unmatched,upddate_unmatched_thread_num, 0, stream[0]>>>(person2obj_matching_dev,current_unmatched_people_idx_dev,next_round_unmatched_people_idx_dev,people_number));
    HANDLE_ERROR(cudaEventRecord(unmatched_updated, stream[0]));

    #ifdef DEBUG
    //HANDLE_ERROR(cudaMemcpy(next_round_unmatched_people_idx,next_round_unmatched_people_idx_dev, (people_number+1)*sizeof(int) , cudaMemcpyDeviceToHost )); 
    //HANDLE_ERROR(cudaDeviceSynchronize());
    //int unmatched = next_round_unmatched_people_idx[people_number];
    //for (int i = 0; i < unmatched; i++){
    //  printf("person %d is unmatched and will bid in the following round\n", next_round_unmatched_people_idx[i]);
    //}
    //print_assignment(person2obj_matching, people_number, prices); // use NULL to avoid printing prices
    #endif
    
    // swap current_unmatched_people_idx_dev and next_round_unmatched_people_idx_dev (not captured by cuda record)
    int *tmp =  current_unmatched_people_idx_dev;
    current_unmatched_people_idx_dev = next_round_unmatched_people_idx_dev;
    next_round_unmatched_people_idx_dev = tmp;

  #ifndef USE_CUDA_GRAPH

    HANDLE_ERROR(cudaStreamSynchronize(stream[0]));
    HANDLE_ERROR(cudaStreamSynchronize(stream[2]));
    HANDLE_ERROR(cudaStreamSynchronize(stream[3]));

    counter+=1;

    #ifdef DEBUG
    assert (previous_round_matched <= *people_matched);
    previous_round_matched = *people_matched;
    //printf("people matched: %d\n", *people_matched);
    #endif


  } while (*people_matched < people_number);

  #else

    HANDLE_ERROR(cudaStreamWaitEvent(stream[0], unmatched_updated));
    HANDLE_ERROR(cudaStreamWaitEvent(stream[0], objects_with_updated_bid_reset));
    HANDLE_ERROR(cudaStreamWaitEvent(stream[0], people_matched_copied));

    HANDLE_ERROR(cudaStreamEndCapture(stream[0], &graphs[cuda_graph_idx]));

    HANDLE_ERROR(cudaGraphInstantiate(&instances[cuda_graph_idx], graphs[cuda_graph_idx], NULL, NULL, 0));

    cuda_graph_idx +=1;
  
  } while (cuda_graph_idx < graph_num);

  do {

    HANDLE_ERROR(cudaGraphLaunch(instances[counter%2], stream[0])); //a lternatively use the first or the second instance of the graph
    
    counter+=1;

    HANDLE_ERROR(cudaStreamSynchronize(stream[0]));

    #ifdef DEBUG
    assert(previous_round_matched <= people_matched);
    previous_round_matched = *people_matched;
    #endif

  } while (*people_matched < people_number);

  for (int i = 0; i < graph_num; i++){
    HANDLE_ERROR(cudaGraphExecDestroy(instances[i]));
    HANDLE_ERROR(cudaGraphDestroy(graphs[i]));
  }

  #endif

  HANDLE_ERROR(cudaMemcpy(person2obj_matching,person2obj_matching_dev, (people_number)*sizeof(int) , cudaMemcpyDeviceToHost )); 

  double total_cost = 0.0;
  
  #ifdef VERBOSE
  HANDLE_ERROR(cudaMemcpy(prices, prices_dev, sizeof(double)*obj_number, cudaMemcpyDeviceToHost));
  print_assignment(person2obj_matching, people_number, prices); // use NULL to avoid printing prices
  #endif

  for ( int i = 0; i < people_number; i++){
    total_cost += benefits[i*obj_number + person2obj_matching[i]];
  }
  printf("auction rounds: %d\n", counter);
  
  printf("total_cost: %lf\n", total_cost);

  for (int i = 0; i < event_num; i++) {
    HANDLE_ERROR(cudaEventDestroy(event[i]));
  }

   for (int i = 0; i < stream_num; i++) {
    HANDLE_ERROR(cudaStreamDestroy(stream[i]));
  }

  HANDLE_ERROR(cudaFree(addresses[0])); // free only main pointer
  free(addresses);
  free(obj2person_matching);
  free(current_unmatched_people_idx);
  free(prices);

  #ifdef DEBUG
  free(bid_increment);
  free(two_max_aux);
  free(bid_target);
  free(max_idx_aux);
  free(bid_per_object);
  free(objects_with_updated_bid);
  free(next_round_unmatched_people_idx);
  free(first_reduction_done);
  free(obj_received_bid_from_person);

  #endif

  
  DEBUG_EXEC(printf("====== compiled with debug ======\n"));  

  // copy matching in non pagelocked memory and return the results
  int *matching = (int *) malloc(sizeof(int)*people_number);
  for (int person = 0; person < people_number; person++) {
    matching[person] = person2obj_matching[person];
  }

  HANDLE_ERROR(cudaFreeHost(person2obj_matching));

  return matching;
}
