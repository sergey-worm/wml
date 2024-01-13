//#############################################################################
//
//  Example of using WML (Wrm Tiny ML framework).
//
//  This file contents:
//
//    1. Unit tests.
//    2. Example of learn and test neural network on Iris dataset.
//    3. Example of learn and test neural network on MNIST dataset.
//
//#############################################################################

#include <assert.h>
#include <string.h>
#include "wml_ds_iris.h"
#include "wml_layers.h"
#include "wml_dl_static.h"
#include "wml_data_loaders.h"
#include "wml_mat.h"
#include "wml_utils.h"
#include "wml_plot.h"

//-----------------------------------------------------------------------------
void unit_tests()
{
	if (0)
	{
		T arr1[6];
		Wml_mat_t m1;
		memset(&m1, 0, sizeof(m1));
		m1.nrow = 2;
		m1.ncol = 3;
		m1.arr = arr1;
		m1.arr_cap = sizeof(arr1) / sizeof(arr1[0]);
		m1.arr[0] = 1;
		m1.arr[1] = 2;
		m1.arr[2] = 3;
		m1.arr[3] = 4;
		m1.arr[4] = 5;
		m1.arr[5] = 6;
		wml_mat_print(&m1, "m1");
		m1.transp = 1;
		wml_mat_print(&m1, "m1");

		T arr2[] = {1,2,3,4,5,6,7,8};
		Wml_mat_t m2;
		wml_mat_init(&m2, arr2, 8, 2, 4, 0);
		wml_mat_print(&m2, "m2");

		Wml_mat_t m3;
		wml_mat_init(&m3, wml_alloct(12), 12, 0, 0, 0);
		wml_mat_prod(&m3, &m1, &m2, NULL);
		wml_mat_print(&m3, "m3");

		assert(m3.arr_cap == 12);
		T expected[] = { 21, 26, 31, 36, 27, 34, 41, 48, 33, 42, 51, 60 };
		for (unsigned i=0; i<m3.arr_cap; ++i)
			assert(m3.arr[i] == expected[i]);
	}

	// test:  allocator
	if (0)
	{
		// test allocator
		wml_allocator_init();
		printf("%s:   0. consume mem=%u.\n", __func__, wml_get_mem_consum());
		T* ptr1 = wml_alloct(10);
		printf("%s:  +1. consume mem=%u.\n", __func__, wml_get_mem_consum());
		T* ptr2 = wml_alloct(20);
		printf("%s:  +2. consume mem=%u.\n", __func__, wml_get_mem_consum());
		T* ptr3 = wml_alloct(30);
		printf("%s:  +3. consume mem=%u.\n", __func__, wml_get_mem_consum());
		wml_free(ptr3);
		printf("%s:  -3. consume mem=%u.\n", __func__, wml_get_mem_consum());
		if (0)
			wml_free(ptr2);
		printf("%s:  -2. consume mem=%u.\n", __func__, wml_get_mem_consum());
		wml_free(ptr1);
		printf("%s:  -1. consume mem=%u.\n", __func__, wml_get_mem_consum());
	}

	// test cross entropy loss
	if (1)
	{
		Wml_mat_t prob, targ;
		wml_mat_inita(&prob, 2, 3, wml_zero);
		wml_mat_inita(&targ, 2, 3, wml_zero);
		prob.arr[0] = 0.001;
		prob.arr[1] = 0.999;
		prob.arr[2] = 0.001;
		prob.arr[3] = 0.999;
		prob.arr[4] = 0.001;
		prob.arr[5] = 0.001;

		targ.arr[0] = 0;
		targ.arr[1] = 1;
		targ.arr[2] = 0;
		targ.arr[3] = 1;
		targ.arr[4] = 0;
		targ.arr[5] = 0;
		T loss = 0;
		wml_mat_cross_entropy(&loss, &prob, &targ, NULL);
		wml_mat_print(&prob, "prob");
		wml_mat_print(&targ, "targ");
		printf("test:  loss=%f.\n", loss);
	}

	// test wml plot
	if (0)
	{
		// test of one step plot
		Wml_mat_t m;
		wml_mat_inita(&m, 1, 10, wml_randi);
		wml_plot_array(m.arr, 10);

		// test of 3 step plot
		// step 1
		wml_plot_init();
		// step 2
		wml_plot_add_value(1);
		wml_plot_add_value(2);
		wml_plot_add_value(6);
		// step 3
		wml_plot_make_grath();
	}

	// test shuffle of matrixes
	if (1)
	{
		T arr1[] = {1,2,3, 4,5,6, 7,8,10};
		Wml_mat_t m1;
		wml_mat_init(&m1, arr1, 9, 3, 3, 0);
		wml_mat_print(&m1, "m1");

		T arr2[] = {1,2,3};
		Wml_mat_t m2;
		wml_mat_init(&m2, arr2, 3, 3, 1, 0);
		wml_mat_print(&m2, "m2");

		wml_mat_shuffle(&m1, &m2);
		wml_mat_print(&m1, "m1 after shuffle");
		wml_mat_print(&m2, "m2 after shuffle");
	}
}

// learn and test NN on Iris dataset from wml_iris.h
void iris()
{
	// general constants
	enum
	{
		IN_DIM     = IRIS_DATA_DIM,
		OUT_DIM    = IRIS_TARGET_CLASSES,
		H_DIM      = 10,
		ALPHA_MUL  = 1,
		ALPHA_DIV  = 10000,
		NUM_EPOCH  = 100,
		BATCH_SIZE = 30,

		TRAIN_SZ   = IRIS_DATASET_SZ * 65 / 100,
		TEST_SZ    = IRIS_DATASET_SZ - TRAIN_SZ

		// for this data accuracy is:  53 / 53 = 100 %
	};

	// create layers
	Layer_linear_t layer1 =
	{
		.type = Layer_linear,
		.dim_in = IN_DIM,
		.dim_out = H_DIM,
		.alpha_mul = ALPHA_MUL,
		.alpha_div = ALPHA_DIV,
		.init_func = wml_rand_10 //wml_randi
	};
	Layer_relu_t layer2 =
	{
		.type = Layer_relu
	};
	Layer_linear_t layer3 =
	{
		.type = Layer_linear,
		.dim_in = H_DIM,
		.dim_out = OUT_DIM,
		.alpha_mul = ALPHA_MUL,
		.alpha_div = ALPHA_DIV,
		.init_func = wml_rand_10 //wml_randi
	};
	Layer_softmax_t layer4 =
	{
		.type = Layer_softmax
	};

	Layer_t* layers[] =
	{
		(Layer_t*) &layer1,
		(Layer_t*) &layer2,
		(Layer_t*) &layer3,
		(Layer_t*) &layer4
	};
	for (int i=0; i<sizeof(layers)/sizeof(layers[0]); ++i)
		layer_init(layers[i]);

	// prepare train (120) and test (30) data
	Wml_mat_t train_x;
	Wml_mat_t test_x;
	Wml_mat_t train_y;
	Wml_mat_t test_y;

	Wml_mat_t all_x;
	Wml_mat_t all_y;
	wml_mat_inita(&all_x, IRIS_DATASET_SZ, IRIS_DATA_DIM, wml_zero);
	wml_mat_inita(&all_y, IRIS_DATASET_SZ, IRIS_TARGET_CLASSES, wml_zero);
	// fill x and y matrix from static iris data
	for (int i=0; i<IRIS_DATASET_SZ; ++i)
	{
		for (int j=0; j<IRIS_DATA_DIM; ++j)
			wml_mat_elem_set(&all_x, i, j, iris_data[i][j]);

		// one hot encoding
		for (int j=0; j<IRIS_TARGET_CLASSES; ++j)
			wml_mat_elem_set(&all_y, i, j, iris_target[i] == j);
	}

	// shuffle data
	wml_mat_shuffle(&all_x, &all_y);

	// TODO:  make wml_ut_split_data() for that
	// split all data to train and test
	wml_mat_init(&train_x, wml_mat_elem_ptr(&all_x, 0, 0),
		TRAIN_SZ * IRIS_DATA_DIM, TRAIN_SZ, IRIS_DATA_DIM, NULL);
	wml_mat_init(&train_y, wml_mat_elem_ptr(&all_y, 0, 0),
		TRAIN_SZ * IRIS_TARGET_CLASSES, TRAIN_SZ, IRIS_TARGET_CLASSES, NULL);
	wml_mat_init(&test_x, wml_mat_elem_ptr(&all_x, TEST_SZ, 0),
		TRAIN_SZ * IRIS_DATA_DIM, TEST_SZ, IRIS_DATA_DIM, NULL);
	wml_mat_init(&test_y, wml_mat_elem_ptr(&all_y, TEST_SZ, 0),
		TRAIN_SZ * IRIS_TARGET_CLASSES, TEST_SZ, IRIS_TARGET_CLASSES, NULL);

	wml_mat_print(&train_x, "train_x");
	wml_mat_print(&train_y, "train_y");
	wml_mat_print(&test_x,  "test_x");
	wml_mat_print(&test_y,  "test_y");

	puts("-------------------------------------------");

	// learn
	wml_plot_init();
	for (unsigned epoch=0; epoch<NUM_EPOCH; ++epoch)
	{
		// shuffle dataset
		wml_mat_shuffle(&train_x, &train_y);

		Wml_mat_t batch_x;
		Wml_mat_t batch_y;

		// init static data loader
		Wml_dl_static_t dl;
		wml_dl_static_init(&dl, &train_x, &train_y, BATCH_SIZE, NULL);

		for (unsigned batch_cnt=0;
		     !wml_dl_static_next(&dl, &batch_x, &batch_y); ++batch_cnt)
		{
			// FORWARD

			Wml_mat_t* prev_layer_out = &batch_x;
			for (int i=0; i<sizeof(layers)/sizeof(layers[0]); ++i)
			{
				layer_forward(layers[i], prev_layer_out);  // out matrix is allocated inside
				prev_layer_out = &layers[i]->out;

				# if 1 // test
				if (i == 0)
					assert(prev_layer_out->nrow == BATCH_SIZE  &&  prev_layer_out->ncol == H_DIM);
				if (i == 1)
					assert(prev_layer_out->nrow == BATCH_SIZE  &&  prev_layer_out->ncol == H_DIM);
				if (i == 2)
					assert(prev_layer_out->nrow == BATCH_SIZE  &&  prev_layer_out->ncol == OUT_DIM);
				if (i == 3)
					assert(prev_layer_out->nrow == BATCH_SIZE  &&  prev_layer_out->ncol == OUT_DIM);
				#endif
			}

			// calc loss
			T loss = 0;
			#if 1
			wml_mat_cross_entropy(&loss, prev_layer_out, &batch_y, NULL);
			#else
			wml_mat_mse(&loss, prev_layer_out, &batch_y, NULL);
			#endif

			// BACKWARD

			// de_din for the last softmax layer is calculated first outside of cycle
			unsigned nlayer = sizeof(layers) / sizeof(layers[0]);
			layer_softmax_dedin((Layer_softmax_t*)layers[nlayer-1], &batch_y);

			// backward cycle for all layers but last! and first!
			for (int i=nlayer-2; i>0; --i)
			{
				layer_backward(layers[i], &layers[i-1]->out, &layers[i+1]->de_din);
			}

			// backward path for the first layer is calculated outside of the cycle
			layer_backward(layers[0], &batch_x, &layers[1]->de_din);

			// UPDATE

			for (int i=0; i<sizeof(layers)/sizeof(layers[0]); ++i)
				layer_update(layers[i]);

			// loss_err.append(E)
			wml_plot_add_value(loss);
			printf("epoch=%u:  batch=%u:  loss=" T_spec " -- mem=%u.\n",
				epoch, batch_cnt, loss, wml_get_mem_consum());

			// DEALLOCATION
			for (int i=0; i<sizeof(layers)/sizeof(layers[0]); ++i)
				layer_free_dedin_matrix(layers[i]);
		}
	}

	wml_plot_make_grath();

	// deallocate de_din matrix after backward path
	for (int i=0; i<sizeof(layers)/sizeof(layers[0]); ++i)
		layer_free_dedin_matrix(layers[i]);

	// deallocate out matrix after forward path
	for (int i=sizeof(layers)/sizeof(layers[0])-1; i>=0; --i)
		layer_free_out_matrix(layers[i]);

	// CHECK the model with test_x and test_y

	// forward

	Wml_mat_t* prev_layer_out = &test_x;
	for (int i=0; i<sizeof(layers)/sizeof(layers[0]); ++i)
	{
		layer_forward(layers[i], prev_layer_out); // out matrix is allocated inside
		prev_layer_out = &layers[i]->out;
	}

	Wml_mat_t* z = prev_layer_out;
	assert(test_y.nrow == z->nrow);
	assert(test_y.ncol == z->ncol);

	// print test info

	printf("TEST:");
	wml_mat_print(z, "Z");
	wml_mat_print(&test_y, "test_y");

	int ok_cnt = 0;

	for (int i=0; i<z->nrow; ++i)
	{
		int idx_z = wml_mat_max(wml_mat_elem_ptr(z, i, 0), z->ncol);
		int idx_y = wml_mat_max(wml_mat_elem_ptr(&test_y, i, 0), z->ncol);
		int ok = idx_z == idx_y;
		if (ok)
			ok_cnt++;
		printf("%d:  idx:  z=%d, y=%d, ok=%d.\n", i, idx_z, idx_y, ok);
	}

	// dealocation out matrix after forward path
	for (int i=sizeof(layers)/sizeof(layers[0])-1; i>=0; --i)
		layer_free_out_matrix(layers[i]);

	// deallocate X and Y
	wml_free(all_y.arr);
	wml_free(all_x.arr);

	// dealocate layers
	for (int i=sizeof(layers)/sizeof(layers[0])-1; i>=0; --i)
		layer_fini(layers[i]);

	puts("");
	printf("OK:  %d / %d = %d %%:  mem=%u.\n",
		ok_cnt, z->nrow, ok_cnt * 100 / z->nrow, wml_get_mem_consum());

}

// go forward with test X and Y
unsigned mnist_test(Layer_t** layers, unsigned nlayer,
                Wml_dl_file_t* dl,
                Wml_mat_t* batch_x,
                Wml_mat_t* batch_y)
{
	int ok_cnt = 0;
	int all_cnt = 0;
	int rand = 0;

	wml_dl_file_clear_batch_counter(dl);

	for (unsigned batch_cnt=0;
	     !wml_dl_file_get_next_test_batch(dl, batch_x, batch_y, rand)  && batch_cnt<10;
	     ++batch_cnt)

	{
		Wml_mat_t* prev_layer_out = batch_x;
		for (int i=0; i<nlayer; ++i)
		{
			layer_forward(layers[i], prev_layer_out); // out matrix is allocated inside
			prev_layer_out = &layers[i]->out;
		}

		Wml_mat_t* z = prev_layer_out;
		assert(batch_y->nrow == z->nrow);
		assert(batch_y->ncol == z->ncol);

		// print test info
		for (int i=0; i<z->nrow; ++i)
		{
			all_cnt++;
			int idx_z = wml_mat_max(wml_mat_elem_ptr(z, i, 0), z->ncol);
			int idx_y = wml_mat_max(wml_mat_elem_ptr(batch_y, i, 0), z->ncol);
			int ok = idx_z == idx_y;
			if (ok)
				ok_cnt++;
			//printf("batch=%u/%u:  %d:  idx:  z=%d, y=%d, ok=%d.\n",
			//	batch_cnt, dl.test_samples / dl.batch_sz, i, idx_z, idx_y, ok);
		}
	}

	printf("TEST:  %d / %d = %d %%:  mem=%u.\n",
		ok_cnt, all_cnt, ok_cnt * 100 / all_cnt, wml_get_mem_consum());

	return ok_cnt * 100 / all_cnt;
}

// learn and test NN on MNIST dataset from python-mnist
void mnist()
{
	// general constants
	enum
	{
		DIM_IN       = 28 * 28, // dimention of X data
		DIM_OUT      = 10,      // dimention of target Y
		DIM_OUT_FILE = 1,       // dimention of target in Y file (1 byte)
		BATCH_SZ     = 100,     //

		DIM_HID_1    = 256,     // dimention of hidding layer 1
		DIM_HID_2    = 100,     // dimention of hidding layer 2

		ALPHA_MUL_1  = 800,
		ALPHA_MUL_2  =  5,
		ALPHA_MUL_3  = 20,
		ALPHA_DIV    = 1000*1000*1000,
		NUM_EPOCH    = 8
	};

	// create layers
	Layer_linear_t layer1 =
	{
		.type = Layer_linear,
		.dim_in = DIM_IN,
		.dim_out = DIM_HID_1,
		.alpha_mul = ALPHA_MUL_1,
		.alpha_div = ALPHA_DIV,
		.init_func = wml_rand_10
	};
	Layer_relu_t layer2 =
	{
		.type = Layer_relu
	};
	Layer_linear_t layer3 =
	{
		.type = Layer_linear,
		.dim_in = DIM_HID_1,
		.dim_out = DIM_HID_2,
		.alpha_mul = ALPHA_MUL_2,
		.alpha_div = ALPHA_DIV,
		.init_func = wml_rand_10
	};
	Layer_relu_t layer4 =
	{
		.type = Layer_relu
	};
	Layer_linear_t layer5 =
	{
		.type = Layer_linear,
		.dim_in = DIM_HID_2,
		.dim_out = DIM_OUT,
		.alpha_mul = ALPHA_MUL_3,
		.alpha_div = ALPHA_DIV,
		.init_func = wml_rand_10
	};
	Layer_softmax_t layer6 =
	{
		.type = Layer_softmax
	};

	Layer_t* layers[] =
	{
		(Layer_t*) &layer1,
		(Layer_t*) &layer2,
		(Layer_t*) &layer3,
		(Layer_t*) &layer4,
		(Layer_t*) &layer5,
		(Layer_t*) &layer6
	};
	unsigned nlayer = sizeof(layers) / sizeof(layers[0]);
	for (int i=0; i<nlayer; ++i)
	{
		layer_init(layers[i]);
		layer_print(layers[i]);
	}

	// create data loader
	Wml_dl_file_t dl;
	int rc = wml_dl_file_init(&dl,
	                          "MNIST/train-images-idx3-ubyte",
	                          "MNIST/train-labels-idx1-ubyte",
	                          "MNIST/t10k-images-idx3-ubyte",
	                          "MNIST/t10k-labels-idx1-ubyte",
	                          16, // skip file header for data file
	                           8, // skip file header for labels file
                              DIM_IN, DIM_OUT, DIM_OUT_FILE, BATCH_SZ);
	assert(!rc && "wml_dl_file_init failed");

	puts("-------------------------------------------");

	Wml_mat_t batch_x;
	Wml_mat_t batch_y;
	wml_mat_init(&batch_x, NULL, 0, 0, 0, NULL);
	wml_mat_init(&batch_y, NULL, 0, 0, 0, NULL);

	unsigned iters = dl.train_samples / dl.batch_sz;
	unsigned accuracy[NUM_EPOCH] = { 0 };

	// learn
	wml_plot_init();
	for (unsigned epoch=0; epoch<NUM_EPOCH; ++epoch)
	{
		printf("------- epoch=%d/%d -----------------------\n", epoch, NUM_EPOCH);
		for (unsigned batch_cnt=0; batch_cnt<iters; ++batch_cnt)
		{
			int rand = 1;
			int rc = wml_dl_file_get_next_train_batch(&dl, &batch_x, &batch_y, rand);
			assert(!rc && "wml_dl_file_get_next_train_batch() - failed");

			// FORWARD

			Wml_mat_t* prev_layer_out = &batch_x;
			for (int i=0; i<nlayer; ++i)
			{
				layer_forward(layers[i], prev_layer_out);  // out matrix is allocated inside
				prev_layer_out = &layers[i]->out;

				# if 1 // test
				if (i == 0)
					assert(prev_layer_out->nrow == BATCH_SZ  &&  prev_layer_out->ncol == DIM_HID_1);
				if (i == 1)
					assert(prev_layer_out->nrow == BATCH_SZ  &&  prev_layer_out->ncol == DIM_HID_1);
				if (i == 2)
					assert(prev_layer_out->nrow == BATCH_SZ  &&  prev_layer_out->ncol == DIM_HID_2);
				if (i == 3)
					assert(prev_layer_out->nrow == BATCH_SZ  &&  prev_layer_out->ncol == DIM_HID_2);
				if (i == 4)
					assert(prev_layer_out->nrow == BATCH_SZ  &&  prev_layer_out->ncol == DIM_OUT);
				if (i == 5)
					assert(prev_layer_out->nrow == BATCH_SZ  &&  prev_layer_out->ncol == DIM_OUT);
				#endif
			}

			// calc loss
			T loss = 0;
			wml_mat_cross_entropy(&loss, prev_layer_out, &batch_y, NULL);

			// print debug info
			int ok_cnt = 0;
			for (int i=0; i<prev_layer_out->nrow; ++i)
			{
				int idx_z = wml_mat_max(wml_mat_elem_ptr(prev_layer_out, i, 0), prev_layer_out->ncol);
				int idx_y = wml_mat_max(wml_mat_elem_ptr(&batch_y, i, 0), prev_layer_out->ncol);
				int ok = idx_z == idx_y;
				if (ok)
					ok_cnt++;
				if (0)
					printf("batch=%u/%u:  %d:  idx:  z=%d, y=%d, ok=%d.\n",
						batch_cnt, dl.test_samples / dl.batch_sz, i, idx_z, idx_y, ok);
			}

			// BACKWARD

			#if 0
			// decrease learning rate during the learning
			layer3.alpha_div = ALPHA_DIV + (epoch*iters+batch_cnt) * 100*1000;
			layer3.alpha_div = ALPHA_DIV + (epoch*iters+batch_cnt) * 100*1000;
			layer5.alpha_div = ALPHA_DIV + (epoch*iters+batch_cnt) * 100*1000;
			#endif

			// de_din for the last softmax layer is calculated first outside of cycle
			layer_softmax_dedin((Layer_softmax_t*)layers[nlayer-1], &batch_y);

			// backward cycle for all layers but last! and first!
			for (int i=nlayer-2; i>0; --i)
				layer_backward(layers[i], &layers[i-1]->out, &layers[i+1]->de_din);

			// backward path for the first layer is calculated outside of the cycle
			layer_backward(layers[0], &batch_x, &layers[1]->de_din);

			// UPDATE PARAMETERS

			for (int i=0; i<nlayer; ++i)
				layer_update(layers[i]);

			// loss_err.append(E)
			wml_plot_add_value(loss);
			printf("epoch=%u/%u:  batch=%u/%u:  loss=" T_spec ", "
			       "ok_cnt=%d/%d,  -- mem=%u.\n",
				epoch, NUM_EPOCH, batch_cnt, iters, loss, ok_cnt, BATCH_SZ,
				wml_get_mem_consum());
		}

		// test the model with test X and Y
		accuracy[epoch] = mnist_test(layers, nlayer, &dl, &batch_x, &batch_y);
	}
	wml_plot_make_grath();

	// DEALLOCATIONS

	// deallocate de_din matrix after backward path
	for (int i=0; i<nlayer; ++i)
		layer_free_dedin_matrix(layers[i]);

	// deallocate out matrix after forward path
	for (int i=nlayer-1; i>=0; --i)
		layer_free_out_matrix(layers[i]);

	// deallocate out matrix after forward path
	for (int i=nlayer-1; i>=0; --i)
		layer_free_out_matrix(layers[i]);

	wml_free(batch_y.arr);
	wml_free(batch_x.arr);

	// dealocation out matrix after forward path
	for (int i=nlayer-1; i>=0; --i)
		layer_free_out_matrix(layers[i]);

	// dealocate layers
	for (int i=nlayer-1; i>=0; --i)
		layer_fini(layers[i]);

	// print statistics
	puts("");
	printf("ALPHA_MUL_1=%u:\n", ALPHA_MUL_1);
	for (unsigned epoch=0; epoch<NUM_EPOCH; ++epoch)
		printf("TEST:  epoch=%u:  accuracy = %d %%.\n", epoch, accuracy[epoch]);

	puts("");
	printf("MEM:  mem=%u.\n", wml_get_mem_consum());
}

// ############################################################################
int main(int argc, const char** argv)
{
	wml_allocator_init();

	// Uuncomment one line to execute the function:
	//unit_tests();
	//iris();
	mnist();

	return 0;
}
