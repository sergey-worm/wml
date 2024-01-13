//#############################################################################
//
//  Static data loader - an abstraction to work with dataset that represents as
//                       X and Y matrixes.
//
//#############################################################################

// TODO:  "static" - is not sutible word for that, need to think about this.

#pragma once

#include "wml_mat.h"

//-----------------------------------------------------------------------------
// Data loader iterator:
//    contains:
//      - ptr to 2d array of X
//      - ptr to 1d arr Y
//      - batch_sz
//      - counter of returned batchs
//    get:
//      - fill Mat_X and Mat_Y by values with size = batch_sz
//-----------------------------------------------------------------------------
typedef struct
{
	Wml_mat_t* dataset_x;  // iteratible dataset x
	Wml_mat_t* dataset_y;  // iteratable dataset y
	unsigned batch_sz;     // batch size in rows
	unsigned cnt;          // count of iteration
} Wml_dl_static_t;

// init static data loader
int wml_dl_static_init(Wml_dl_static_t* dl,
                       Wml_mat_t* dataset_x, Wml_mat_t* dataset_y,
                       unsigned batch_sz, const char* label);

// fill batch_x and batch_y by next portion of data
int wml_dl_static_next(Wml_dl_static_t* dl,
                       Wml_mat_t* batch_x, Wml_mat_t* batch_y);

