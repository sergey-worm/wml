//#############################################################################
//
//  Collaction of data loaders.
//
//#############################################################################

#pragma once
#include "wml_mat.h"
#include "wml_utils.h"
#include <stdio.h>

//-----------------------------------------------------------------------------
// GENERIC DATA LOADER
//-----------------------------------------------------------------------------

enum
{
	Wml_dl_array = 1,
	Wml_dl_file  = 2,
};

typedef struct
{
	int type;
} Wml_dl_t;

//-----------------------------------------------------------------------------
// STAITIC ARRAY DATA LOADER
//-----------------------------------------------------------------------------

typedef struct
{
	int type;
	// specific params
	// ...
} Wml_dl_array_t;

// TODO:  generic DL can just have pointer to funcs read(pos, sz) and use it
// TODO:  or read_train() and read_test()

//-----------------------------------------------------------------------------
// UNIX FILE DATA LOADER
//
// This data loader reads data from file. Since WML is designed for
// microcontrollers, low memory consumption is important.
//-----------------------------------------------------------------------------
typedef struct
{
	int      type;            // type of data loader
	// specific params
	FILE*    file_train_x;    // file with train sampels
	FILE*    file_train_y;    // file with train targets
	FILE*    file_test_x;     // file with test samples
	FILE*    file_test_y;     // file with test targets
	unsigned skip_x_bytes;    // header size of X file
	unsigned skip_y_bytes;    // header size of Y file
	unsigned x_dim;           // bytes in one sample in X set
	unsigned y_dim;           // bytes in one sample in Y set
	unsigned y_file_dim;      // bytes in one sample in Y file
	unsigned batch_sz;        // size of one batch in samples
	unsigned batch_cnt;       // counter of current batch
	unsigned train_samples;   // amount of samples in train dataset
	unsigned test_samples;    // amount of samples in test dataset
} Wml_dl_file_t;


int wml_dl_file_init(Wml_dl_file_t* dl,
                     const char* fname_train_x, const char* fname_train_y,
                     const char* fname_test_x,  const char* fname_test_y,
                     unsigned skip_x_bytes, unsigned skip_y_bytes,
                     unsigned x_dim, unsigned y_dim, unsigned y_file_dim,
                     unsigned batch_sz);

int wml_dl_file_split_test_train(Wml_dl_file_t* dl);

// clear batch counter to start batch iteration from beginning
void wml_dl_file_clear_batch_counter(Wml_dl_file_t* dl);

// read random data and lables and put it to X and Y matrix
int wml_dl_file_get_next_train_batch(Wml_dl_file_t* dl,
                                     Wml_mat_t* x, Wml_mat_t* y, int rand);
// get test data as matrixes
int wml_dl_file_get_next_test_batch(Wml_dl_file_t* dl,
                                    Wml_mat_t* x, Wml_mat_t* y, int rand);
void wml_dl_file_test();
