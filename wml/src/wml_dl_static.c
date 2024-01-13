//#############################################################################
//
//  Static data loader - an abstraction to work with dataset that represents as
//                       X and Y matrixes.
//
//#############################################################################

#include "wml_dl_static.h"
#include <assert.h>
#include <stdio.h>

// init static data loader
int wml_dl_static_init(Wml_dl_static_t* dl,
                       Wml_mat_t* dataset_x, Wml_mat_t* dataset_y,
                       unsigned batch_sz, const char* label)
{
	if (label)
		printf("dl_init:  %s:  mat_nrow(dataset_x)=%u, mat_nrow(dataset_y)=%u.\n",
			label, mat_nrow(dataset_x), mat_nrow(dataset_y));

	assert(mat_nrow(dataset_x) == mat_nrow(dataset_y));

	dl->dataset_x = dataset_x;
	dl->dataset_y = dataset_y;
	dl->batch_sz  = batch_sz;
	dl->cnt       = 0;
	return 0;
}

// fill batch_x and batch_y by next portion of data
int wml_dl_static_next(Wml_dl_static_t* dl,
                       Wml_mat_t* batch_x, Wml_mat_t* batch_y)
{
	if (dl->batch_sz * (dl->cnt + 1) > mat_nrow(dl->dataset_x))
		return 1;

	unsigned row_sz = mat_ncol(dl->dataset_x);
	unsigned sz = dl->batch_sz * row_sz;
	wml_mat_init(batch_x, dl->dataset_x->arr + dl->cnt * sz, sz, dl->batch_sz, row_sz, 0);

	row_sz = mat_ncol(dl->dataset_y);
	sz = dl->batch_sz * row_sz;
	wml_mat_init(batch_y, dl->dataset_y->arr + dl->cnt * sz, sz, dl->batch_sz, row_sz, 0);

	dl->cnt += 1;
	return 0;
}

