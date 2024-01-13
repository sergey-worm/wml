//#############################################################################
//
//  Collaction of data loaders.
//
//#############################################################################

#include "wml_data_loaders.h"
#include <stdint.h>
#include <assert.h>

//-----------------------------------------------------------------------------
// GENERIC DATA LOADER
//-----------------------------------------------------------------------------

// TODO:  ...

//-----------------------------------------------------------------------------
// STAITIC ARRAY DATA LOADER
//-----------------------------------------------------------------------------

// TODO:  generic DL can just have pointer to funcs read(pos, sz) and use it
// TODO:  or read_train() and read_test()

//-----------------------------------------------------------------------------
// UNIX FILE DATA LOADER
//
// This data loader reads data from file. Since WML is designed for
// microcontrollers, low memory consumption is important.
//-----------------------------------------------------------------------------

static int file_size(FILE* f)
{
	fseek(f, 0, SEEK_END);
	int sz = ftell(f);
	//fseek(f, 0, SEEK_SET);
	return sz;
}

static int file_read(FILE* f, unsigned pos, uint8_t* buf, unsigned sz)
{
	fseek(f, pos, SEEK_SET);
	int rd = fread(buf, 1, sz, f);
	return rd;
}

int wml_dl_file_init(Wml_dl_file_t* dl,
                     const char* fname_train_x, const char* fname_train_y,
                     const char* fname_test_x,  const char* fname_test_y,
                     unsigned skip_x_bytes, unsigned skip_y_bytes,
                     unsigned x_dim, unsigned y_dim, unsigned y_file_dim,
                     unsigned batch_sz)
{
	// init params
	dl->type         = Wml_dl_file;
	dl->file_train_x = fopen(fname_train_x, "r");
	dl->file_train_y = fopen(fname_train_y, "r");
	dl->file_test_x  = fopen(fname_test_x,  "r");
	dl->file_test_y  = fopen(fname_test_y,  "r");
	dl->skip_x_bytes = skip_x_bytes;
	dl->skip_y_bytes = skip_y_bytes;
	dl->x_dim        = x_dim;
	dl->y_dim        = y_dim;
	dl->y_file_dim   = y_file_dim;
	dl->batch_sz     = batch_sz;
	dl->batch_cnt    = 0;

	if (!dl->file_train_x || !dl->file_train_y ||
	    !dl->file_test_x || !dl->file_test_y)
	{
		assert(0 && "There are not MNIST files");
		return -1;
	}

	// get size of datasets
	unsigned train_x_bytes = file_size(dl->file_train_x);
	unsigned train_y_bytes = file_size(dl->file_train_y);
	unsigned test_x_bytes  = file_size(dl->file_test_x);
	unsigned test_y_bytes  = file_size(dl->file_test_y);
	dl->train_samples = (train_x_bytes - skip_x_bytes) / dl->x_dim;
	dl->test_samples = (test_x_bytes - skip_x_bytes) / dl->x_dim;

	// check size of datasets
	unsigned train_targets = (train_y_bytes - skip_y_bytes) / dl->y_file_dim;
	unsigned test_targets = (test_y_bytes - skip_y_bytes) / dl->y_file_dim;
	if (dl->train_samples != train_targets)
	{
		printf("ERR:  %s:  train_samples=%u, train_targets=%u.\n", __func__,
			dl->train_samples, train_targets);
		assert(0 && "train_samples file is not compatible with train_targets");
		return -1;
	}
	if (dl->test_samples != test_targets)
	{
		printf("ERR:  %s:  test_samples=%u, test_targets=%u.\n", __func__,
			dl->test_samples, test_targets);
		assert(0 && "test_samples file is not compatible with test_targets");
		return -1;
	}

	// debug output
	printf("%s:  train_samples=%u, test_samples=%u.\n", __func__,
		dl->train_samples, dl->test_samples);

	return 0;
}

int wml_dl_file_split_test_train(Wml_dl_file_t* dl)
{
	// if we already have test and train files - we don't need to split
	return -1;
}

// read random data and lables and put it to X and Y matrix
static int wml_dl_file_get_next_batch(Wml_dl_file_t* dl,
                                      Wml_mat_t* x,
                                      Wml_mat_t* y,
                                      FILE* file_x,
                                      FILE* file_y,
                                      unsigned samples,
                                      int rand)
{
	if (!rand && (dl->batch_cnt * dl->batch_sz >= samples))
		return 1;  // data is over for rand case

	// check matrix capabilities and alloc if it is zero
	if (!x->arr_cap && !y->arr_cap)
	{
		// allocate X batch matrix
		int rc = wml_mat_inita(x, dl->batch_sz, dl->x_dim, NULL);
		assert(!rc && "wml_mat_init(x) - failed");
		if (rc)
			return -1;

		// allocate Y batch matrix
		rc = wml_mat_inita(y, dl->batch_sz, dl->y_dim, NULL);
		assert(!rc && "wml_mat_init(y) - failed");
		if (rc)
			return -2;
	}
	else
	if (x->arr_cap != dl->x_dim * dl->batch_sz  ||
	    y->arr_cap != dl->y_dim * dl->batch_sz)
	{
		printf("ERR:  x->arr_cap=%u != dl->x_dim=%u * dl->batch_sz=%u.\n", x->arr_cap, dl->x_dim, dl->batch_sz);
		printf("ERR:  y->arr_cap=%u != dl->y_dim=%u * dl->batch_sz=%u.\n", y->arr_cap, dl->y_dim, dl->batch_sz);
		assert(0 && "wml_dl_file_get_next_batch:  wrong x/y cap");
		return -3;
	}

	// allocate buffer for one sample and one target of data
	uint8_t* buf_x = wml_alloc(dl->x_dim);
	uint8_t* buf_y = wml_alloc(dl->y_file_dim);

	// read data from and fill matrixes
	for (unsigned i=0; i<dl->batch_sz; ++i)
	{
		// define index of loaded element - randomly or consequently
		int id = 0;
		if (rand)
			id = wml_rand() % samples;
		else
			id = dl->batch_cnt * dl->batch_sz + i;

		// read X
		unsigned pos = dl->skip_x_bytes + id * dl->x_dim;
		int sz = file_read(file_x, pos, buf_x, dl->x_dim);
		assert(sz == dl->x_dim && "file_read(sample) - failed");
		if (sz != dl->x_dim)
			return -4;

		// read Y
		pos = dl->skip_y_bytes + id * dl->y_file_dim;
		sz = file_read(file_y, pos, buf_y, dl->y_file_dim);
		assert(sz == dl->y_file_dim && "file_read(target) - failed");
		if (sz != dl->y_file_dim)
			return -5;

		// put X to matrix
		for (int j=0; j<dl->x_dim; ++j)
			wml_mat_elem_set(x, i, j, buf_x[j]);

		// put Y to matrix
		for (int j=0; j<dl->y_dim; ++j)
		{
			if (dl->y_dim == dl->y_file_dim)
			{
				// equal dimention - just put value from file to batch
				wml_mat_elem_set(y, i, j, buf_y[j]);
			}
			if (dl->y_file_dim == 1)
			{
				// labels in the file is one value - use one short encoding
				wml_mat_elem_set(y, i, j, buf_y[0] == j);
			}
			else
			{
				// we don't know how to put data from file to Y matrix
				assert("Unsupported label dimention in Y file");
				return -6;
			}
		}
	}

	// deallocate buffer
	wml_free(buf_y);
	wml_free(buf_x);

	dl->batch_cnt += 1;
	return 0;
}

// clear batch counter to start batch iteration from beginning
void wml_dl_file_clear_batch_counter(Wml_dl_file_t* dl)
{
	dl->batch_cnt = 0;
}

// read random data and lables and put it to X and Y matrix
int wml_dl_file_get_next_train_batch(Wml_dl_file_t* dl,
                                     Wml_mat_t* x, Wml_mat_t* y, int rand)
{
	return wml_dl_file_get_next_batch(dl, x, y, 
	                                  dl->file_train_x,
	                                  dl->file_train_y,
	                                  dl->train_samples,
	                                  rand);
}

// get test data as matrixes
int wml_dl_file_get_next_test_batch(Wml_dl_file_t* dl,
                                    Wml_mat_t* x, Wml_mat_t* y, int rand)
{
	return wml_dl_file_get_next_batch(dl, x, y, 
	                                  dl->file_test_x,
	                                  dl->file_test_y,
	                                  dl->test_samples,
	                                  rand);
}

static void _draw_mnist_digit(Wml_dl_file_t* dl,
                              Wml_mat_t* batch_x, Wml_mat_t* batch_y,
                              unsigned id)
{
	printf("Draw digit:\n");
	for (int i=0; i<dl->y_dim; ++i)
		printf("  %d=%d%c", i, (int)wml_mat_elem_get(batch_y, id, i),
			wml_mat_elem_get(batch_y, id, i)>0 ? '*' : ' ');
	printf("\n");


	for (int i=0; i<28; ++i)
	{
		for (int j=0; j<28; ++j)
		{
			int pixel = wml_mat_elem_get(batch_x, id, i * 28 + j);
			printf("%c", pixel > 100 ? '#' : '.');
		}
		printf("\n");
	}
}

void wml_dl_file_test()
{
	enum
	{
		In_dim       = 28 * 28, // dimention of X data
		Out_dim      = 10,      // dimention of target Y
		Out_file_dim = 1,       // dimention of target in Y file
		Batch_sz     = 10,      //
	};

	// create data loader for MNIST dataset
	Wml_dl_file_t dl;
	int rc = wml_dl_file_init(&dl,
	                          "MNIST/raw/train-images-idx3-ubyte",
	                          "MNIST/raw/train-labels-idx1-ubyte",
	                          "MNIST/raw/t10k-images-idx3-ubyte",
	                          "MNIST/raw/t10k-labels-idx1-ubyte",
	                          16, 8, // skip file headers
	                          In_dim, Out_dim, Out_file_dim, Batch_sz);
	assert(!rc && "wml_dl_file_init failed");

	Wml_mat_t batch_x;
	Wml_mat_t batch_y;

	// get and print train data
	unsigned iters = 1; //dl.train_samples / dl.batch_sz;
	for (unsigned batch_cnt=0; batch_cnt<iters; ++batch_cnt)
	{
		// get next batch
		wml_dl_file_get_next_train_batch(&dl, &batch_x, &batch_y, 1);
		printf("%s:  train:  batch_cnt=%d:  batch_sz=%d:\n", __func__,
			batch_cnt, Batch_sz);
		// draw first digits
		for (int k=0; k<5; ++k)
			_draw_mnist_digit(&dl, &batch_x, &batch_y, k);
	}

	// get and print test data
	iters = 1; //dl.test_samples / dl.batch_sz;
	for (unsigned batch_cnt=0; batch_cnt<iters; ++batch_cnt)
	{
		// get next batch
		wml_dl_file_get_next_test_batch(&dl, &batch_x, &batch_y, 0);
		printf("%s:  test:  batch_cnt=%d:  batch_sz=%d:\n", __func__,
			batch_cnt, Batch_sz);
		// draw first digits
		for (int k=0; k<5; ++k)
			_draw_mnist_digit(&dl, &batch_x, &batch_y, k);
	}
}
