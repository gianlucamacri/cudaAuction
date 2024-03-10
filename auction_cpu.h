#ifndef AUCTION_CPU_H

#define AUCTION_CPU_H

#include "common.h"
#include <float.h>

// struct to represent the bid increment along with the targetted object for it
typedef struct bid_increment {
    double increment;
    int target;
} bid_increment;

// computes the best bid for the given person, according to the current prices, the person benefits and the epsilon
// returns the bid increment calculated as the positive difference between the 2 highet profits (benefit - price) plus epsilon and the targetted object
bid_increment get_new_bid_increment(int person, double *prices, int *person_benefits, int obj_num, int person_num, double epsilon);

// realizes the bidding phase of the auction algorithm, collectiong the bid increments by each person according to the given prices, the benefits and epsilon at the current round
// highest bid increments for each object are collected into bid_increments with the corresponding bidder saved into highest_bidder
// only positions with increment > 0 are considered relevant
void update_bids(double *bid_increments, int *highest_bidder, double *prices, int *matching, int *benefits, int obj_num, int person_num, double epsilon);

// compute the new bid for the given person according to current prices, the person benefits and epsilon
// the increment will be equal to the positive difference between the 2 highest profits (benefit - price) plus epsilon
// returns the new bid increment with the associated target object id  
void assign_objects_and_update_prices(double *bid_increments, int *highest_bidder, double *prices, int *matching, int* obj_assignment, int obj_num, int *assigned_people);

// given a matrix of integers (may be generalized to floats) of dimensions person_num*obj_num
// returns when it exists a an array of integer representing an (almost if float benefits are used) optimal matching
// this is represented by and integer array that contains the matched object for each person
int *auction_cpu(int *benefits, int person_num, int obj_num, double epsilon);

#endif
