#include "auction_cpu.h"
#include <omp.h>

bid_increment get_new_bid_increment(int person, double *prices, int *person_benefits, int obj_num, int person_num, double epsilon){
    
    double max_profit, second_max_profit, profit;
    int selected_object, benefit;
    bid_increment new_bid;

    max_profit = second_max_profit = -FLT_MAX; // FLT to avoid overflow


    for (int obj = 0; obj < obj_num; obj++) {
        benefit = person_benefits[obj];

        if(benefit != NEGINFTY){
            profit = benefit - prices[obj];

                if (profit > max_profit) {
                    second_max_profit = max_profit;
                    max_profit = profit;
                    selected_object = obj;
                } else if (profit > second_max_profit){
                    second_max_profit = profit;
                }
        }
    }

    new_bid.increment = max_profit - second_max_profit + epsilon;
    new_bid.target = selected_object;

    //printf("person: %d\nobject number is: %d\nmax is: %lf\nsecond max is: %lf\nmax index is: %d\n\n", person, obj_num, max_profit , second_max_profit, selected_object);
    //printf("person %d increments bids on obj %d by %lf\n", person, new_bid.target, new_bid.increment);

    return new_bid;
}


void update_bids(double *bid_increments, int *highest_bidder, double *prices, int *matching, int *benefits, int obj_num, int person_num, double epsilon){

    bid_increment new_bid;

    memset(bid_increments, 0, obj_num*sizeof(double)); // init increments

    #pragma parallel for schedule (dynamic,1) private (new_bid)
    for (int person = 0; person < person_num; person++) {

        if (matching[person] == UNASSIGNED) { // a person bets only if it has no object assigned

            new_bid = get_new_bid_increment(person, prices, &benefits[person*obj_num], obj_num, person_num, epsilon);

            // register only highest bid increment and bidder for each object
            #pragma critical
            if (new_bid.increment > bid_increments[new_bid.target]) {
                bid_increments[new_bid.target] = new_bid.increment;
                highest_bidder[new_bid.target] = person;
            }
            //printf("person %d increments bids on obj %d by %lf\n", person, new_bid.target, new_bid.increment);
        }
    }
}

void assign_objects_and_update_prices(double *bid_increments, int *highest_bidder, double *prices, int *matching, int* obj_assignment, int obj_num, int *assigned_people){
    
    int previous_highest_bidder;

    //#pragma omp parallel for private(previous_highest_bidder)
    for (int obj = 0; obj < obj_num; obj++) {
        if (bid_increments[obj] > 0) {

            prices[obj] += bid_increments[obj];

            previous_highest_bidder = obj_assignment[obj];

            if (previous_highest_bidder == UNASSIGNED) {
                //#pragma omp atomic
                (*assigned_people) += 1;
            } else {
                matching[previous_highest_bidder] = UNASSIGNED;
            }

            matching[highest_bidder[obj]] = obj;
            obj_assignment[obj] = highest_bidder[obj];
        }
        //printf("object %d received highest bid increment of %lf by %d\n", obj, bid_increments[obj], highest_bidder[obj]);
    }
}

int count_unmatched(int *matching, int len){
    int count = 0;
    for (int i = 0; i < len; i++)
    {
        if (matching[i] == UNASSIGNED) {
            count +=1;
        }
    }
    return count;
    
}

int *auction_cpu(int *benefits, int person_num, int obj_num, double epsilon){

    if (person_num > obj_num) {
        fprintf(stderr, "error: cannot find a complete match when the number of people is lower than the numebr of objects.\n");
        exit(-1);
    }

    if (person_num < 2) {
        fprintf(stderr, "error: both people number and object number must be greater than or equal to 2.\n");
        exit(-1);
    }


    int *assigned_people,       // matched people counter (monothonic)
        *obj_assignments,       // array object -> assigned person
        *matching,              // matching person -> object
        *highest_bidder;        // bidders correspoinding to bid_increments
    double  *prices,            // current object prices
            *bid_increments;    // round highest bid increments for object

    // init
    assigned_people = (int *) calloc(1,sizeof(int));
    prices = (double*) calloc(obj_num, sizeof(double));
    bid_increments = (double*) malloc(obj_num*sizeof(double));
    obj_assignments = (int*) malloc(obj_num*sizeof(int));
    highest_bidder = (int*) malloc(obj_num*sizeof(int));
    matching = (int *) malloc(person_num*sizeof(int));

    for (int i = 0; i < obj_num; i++) obj_assignments[i] = UNASSIGNED;
    for (int i = 0; i < person_num; i++) matching[i] = UNASSIGNED;

    int counter = 0;
    do {

        // bidding phase
        update_bids(bid_increments, highest_bidder, prices, matching, benefits, obj_num, person_num, epsilon);

        // assignment phase
        assign_objects_and_update_prices(bid_increments, highest_bidder, prices, matching, obj_assignments, obj_num, assigned_people);

        #ifdef DEBUG
        //for ( int i = 0; i < person_num; i++){
        //    printf("person %d won object %d offering %lf\n", i, matching[i], prices[matching[i]]);
        //}
        #endif

        counter+=1;

        //printf("unmatched: %d\n", count_unmatched(matching, person_num));
        //printf("matched: %d\n", *assigned_people);
    } while (*assigned_people < person_num );
    
    double total_cost = 0.0;
    for ( int i = 0; i < person_num; i++){
        #ifdef VERBOSE
        printf("person %d won object %d offering %lf\n", i, matching[i], prices[matching[i]]);
        #endif
        total_cost += benefits[i*obj_num + matching[i]]; // - prices[matching[i]];
    }


    printf("auction rounds: %d\n", counter);
    printf("total_cost: %lf\n", total_cost);

    free(assigned_people);
    free(prices);
    free(bid_increments);
    free(obj_assignments);
    free(highest_bidder);

    return matching;
}
