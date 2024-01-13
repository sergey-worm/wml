//#############################################################################
//
//  Collection of utilites for WML.
//
//#############################################################################

#pragma once
#include <stdint.h>
#include "wml_cfg.h"

#define MAX(a, b) (a > b ? a : b)

//-----------------------------------------------------------------------------
// Linear congruential generator (pseudo-random numbers)
//-----------------------------------------------------------------------------
unsigned wml_rand();

static inline T wml_zero()       { return 0; }
static inline T wml_one()        { return 1; }
static inline T wml_rand_10()    { int l = 10;      return (int) wml_rand() % (2*l) - l; }
static inline T wml_rand_100()   { int l = 100;     return (int) wml_rand() % (2*l) - l; }
static inline T wml_rand_1000()  { int l = 1000;    return (int) wml_rand() % (2*l) - l; }
static inline T wml_rand_10000() { int l = 10000;   return (int) wml_rand() % (2*l) - l; }
static inline T wml_randi()      { return (int) wml_rand(); }

//-----------------------------------------------------------------------------
// Dummy memory stack allocator of byte-arrays.
// It warrks as stack allocator:
//    allocating and deallocating always on the top of memory area (stack).
//-----------------------------------------------------------------------------
void  wml_allocator_init();
void* wml_alloc(unsigned num);
T*    wml_alloct(unsigned num);
int   wml_free(void* p);
int   wml_get_mem_consum();
