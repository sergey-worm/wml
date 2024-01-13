//#############################################################################
//
//  Layer abstraction and implementation.
//
//#############################################################################

#include "wml_layers.h"
#include <string.h>
#include <stdio.h>
#include <assert.h>

//-----------------------------------------------------------------------------
// Linear layer
//-----------------------------------------------------------------------------

int layer_linear_init(Layer_linear_t* l)
{
	memset(&l->out, 0, sizeof(Wml_mat_t));
	memset(&l->de_din, 0, sizeof(Wml_mat_t));
	// 
	wml_mat_inita(&l->w,     l->dim_in, l->dim_out, l->init_func);
	wml_mat_inita(&l->b,             1, l->dim_out, l->init_func);
	wml_mat_inita(&l->de_dw, l->dim_in, l->dim_out, NULL);
	wml_mat_inita(&l->de_db,         1, l->dim_out, NULL);
	return 0;
}

int layer_linear_fini(Layer_linear_t* l)
{
	// back order
	wml_free(l->de_db.arr);
	wml_free(l->de_dw.arr);
	wml_free(l->b.arr);
	wml_free(l->w.arr);
	return 0;
}

int layer_linear_print(Layer_linear_t* l)
{
	printf("layer:  linear:  dim_in=%d, dim_out=%d.\n", l->dim_in, l->dim_out);
	return 0;
}

// out = in @ w + b
// in->nrow is equil to batch_size
// in->ncol is equil to dim_in
int layer_linear_forward(Layer_linear_t* l, const Wml_mat_t* in)
{
	// allocate space if need
	assert(in->ncol == l->dim_in);
	// if out-matrix-arr is empty -- allocate it
	if (!l->out.arr_cap)
		wml_mat_inita(&l->out, in->nrow, l->dim_out, NULL);
	else
	// if out-matrix-arr is not the same size as required -- return with error
	if (l->out.nrow != in->nrow  ||  l->out.ncol != l->dim_out)
	{
		assert(0 && "frw:  need to dealocate out-matrix first.");
		return -1;
	}

	#if 1
	wml_mat_prod(&l->out, in, &l->w, NULL);
	wml_mat_sum(&l->out, &l->out, &l->b, NULL);
	#else
	// TODO:  Make wml_prod_and_sum()
	#endif
	return 0;
}

// calc de_db and de_dw, fill de_din
int layer_linear_backward(Layer_linear_t* l, const Wml_mat_t* in,
                          const Wml_mat_t* de_dout)
{
	assert(in->ncol == l->dim_in);
	assert(de_dout->ncol == l->dim_out);

	// de_db = sum(de_dout, axis=0, keepdims=True)
	wml_mat_sum_cols(&l->de_db, de_dout, NULL);

	// de_dw = in.t @ de_dout
	((Wml_mat_t*)in)->transp = 1;
	wml_mat_prod(&l->de_dw, in, de_dout, NULL);
	((Wml_mat_t*)in)->transp = 0;

	// de_din = de_dout @ w.t
	if (1) // TODO:  add param need_calc_de_din
	{
		// if de_din-matrix-arr is empty -- allocate it
		if (!l->de_din.arr_cap)
			wml_mat_inita(&l->de_din, in->nrow, l->dim_in, NULL);
		else
		// if de_din-matrix-arr is not the same size as required -- return with error
		if (l->de_din.nrow != in->nrow  ||  l->de_din.ncol != l->dim_in)
		{
			assert(0 && "frw:  need to dealocate de_din-matrix first.");
			return -1;
		}
		l->w.transp = 1;
		wml_mat_prod(&l->de_din, de_dout, &l->w, NULL);
		l->w.transp = 0;
	}
	return 0;
}

// TODO:  make optimization - create wml_mat_mult_div_decr()
int layer_linear_update(Layer_linear_t* l)
{
	// w = w - ALPHA * de_dw
	wml_mat_mult_div(&l->de_dw, l->alpha_mul, l->alpha_div, NULL);
	wml_mat_sub_inplace(&l->w, &l->de_dw, NULL);

	// b = b - ALPHA * de_db
	wml_mat_mult_div(&l->de_db, l->alpha_mul, l->alpha_div, NULL);
	wml_mat_sub_inplace(&l->b, &l->de_db, NULL);

	return 0;
}

//-----------------------------------------------------------------------------
// Relu layer
//-----------------------------------------------------------------------------

int layer_relu_init(Layer_relu_t* l)
{
	return 0;
}

int layer_relu_fini(Layer_relu_t* l)
{
	return 0;
}

int layer_relu_print(Layer_relu_t* l)
{
	printf("layer:  relu.\n");
	return 0;
}

int layer_relu_forward(Layer_relu_t* l, const Wml_mat_t* in)
{
	// allocate space if need
	// if out-matrix-arr is empty -- allocate it
	if (!l->out.arr_cap)
		wml_mat_inita(&l->out, in->nrow, in->ncol, NULL);
	else
	// if out-matrix-arr is not the same size as required -- return with error
	if (l->out.nrow != in->nrow  ||  l->out.ncol != in->ncol)
	{
		assert(0 && "frw:  need to dealocate out-matrix first.");
		return -1;
	}

	// out = relu(in)
	wml_mat_clone(&l->out, in);
	wml_mat_relu(&l->out, NULL);
	return 0;
}

int layer_relu_backward(Layer_relu_t* l, const Wml_mat_t* in,
                        const Wml_mat_t* de_dout)
{
	// allocate de_din if need
	// if de_din-matrix-arr is empty -- allocate it
	if (!l->de_din.arr_cap)
		wml_mat_inita(&l->de_din, in->nrow, in->ncol, NULL);
	else
	// if de_din-matrix-arr is not the same size as required -- return with error
	if (l->de_din.nrow != in->nrow  ||  l->de_din.ncol != in->ncol)
	{
		assert(0 && "frw:  need to dealocate de_din-matrix first.");
		return -1;
	}

	// de_din = relu_deriv(in) * de_dout
	wml_mat_clone(&l->de_din, in);
	wml_mat_relu_deriv(&l->de_din, NULL);
	wml_mat_mult(&l->de_din, &l->de_din, de_dout, NULL);
	return 0;
}

int layer_relu_update(Layer_relu_t* l)
{
	return 0;
}

//-----------------------------------------------------------------------------
// Softmax layer
//-----------------------------------------------------------------------------

int layer_softmax_init(Layer_softmax_t* l)
{
	return 0;
}

int layer_softmax_fini(Layer_softmax_t* l)
{
	return 0;
}

int layer_softmax_print(Layer_softmax_t* l)
{
	printf("layer:  softmax.\n");
	return 0;
}

int layer_softmax_forward(Layer_softmax_t* l, const Wml_mat_t* in)
{
	// if out-matrix-arr is empty -- allocate it
	if (!l->out.arr_cap)
		wml_mat_inita(&l->out, in->nrow, in->ncol, NULL);
	else
	// if out-matrix-arr is not the same size as required -- return with error
	if (l->out.nrow != in->nrow  ||  l->out.ncol != in->ncol)
	{
		assert(0 && "frw:  need to dealocate out-matrix first.");
		return -1;
	}

	// out = softmax(in)
	wml_mat_clone(&l->out, in);
	wml_mat_softmax(&l->out, NULL);
	return 0;
}

int layer_softmax_backward(Layer_softmax_t* l, const Wml_mat_t* in,
                           const Wml_mat_t* de_dout)
{
	assert(0 && "Do you realy need to use layer_softmax_backward?");
	return -1;
}

int layer_softmax_dedin(Layer_softmax_t* l, const Wml_mat_t* target_y)
{
	// allocate de_din if need
	// if de_din-matrix-arr is empty -- allocate it
	if (!l->de_din.arr_cap)
		wml_mat_inita(&l->de_din, target_y->nrow, target_y->ncol, NULL);
	else
	// if de_din-matrix-arr is not the same size as required -- return with error
	if (l->de_din.nrow != target_y->nrow  ||  l->de_din.ncol != target_y->ncol)
	{
		assert(0 && "frw:  need to dealocate de_din-matrix first.");
		return -1;
	}

	// de_din = out - target_y
	wml_mat_sub(&l->de_din, &l->out, target_y, NULL);
	return -0;
}

int layer_softmax_update(Layer_softmax_t* l)
{
	return 0;
}

//-----------------------------------------------------------------------------
// API - abstract layer
//-----------------------------------------------------------------------------

int layer_init(Layer_t* l)
{
	switch (l->type)
	{
		case Layer_linear:
			return layer_linear_init((Layer_linear_t*)l);
		case Layer_relu:
			return layer_relu_init((Layer_relu_t*)l);
		case Layer_softmax:
			return layer_softmax_init((Layer_softmax_t*)l);
	}
	return 0;
}

int layer_fini(Layer_t* l)
{
	switch (l->type)
	{
		case Layer_linear:
			return layer_linear_fini((Layer_linear_t*)l);
		case Layer_relu:
			return layer_relu_fini((Layer_relu_t*)l);
		case Layer_softmax:
			return layer_softmax_fini((Layer_softmax_t*)l);
	}
	return 0;
}

int layer_print(Layer_t* l)
{
	switch (l->type)
	{
		case Layer_linear:
			return layer_linear_print((Layer_linear_t*)l);
		case Layer_relu:
			return layer_relu_print((Layer_relu_t*)l);
		case Layer_softmax:
			return layer_softmax_print((Layer_softmax_t*)l);
	}
	return 0;
}

int layer_free_out_matrix(Layer_t* l)
{
	if (l->out.arr_cap)
		wml_free(l->out.arr);
	l->out.arr_cap = 0;
	return 0;
}

int layer_free_dedin_matrix(Layer_t* l)
{
	if (l->de_din.arr_cap)
		wml_free(l->de_din.arr);
	l->de_din.arr_cap = 0;
	return 0;
}

// out = F(x)
int layer_forward(Layer_t* l, const Wml_mat_t* in)
{
	switch (l->type)
	{
		case Layer_linear:
			return layer_linear_forward((Layer_linear_t*)l, in);
		case Layer_relu:
			return layer_relu_forward((Layer_relu_t*)l, in);
		case Layer_softmax:
			return layer_softmax_forward((Layer_softmax_t*)l, in);
	}
	return 0;
}

int layer_backward(Layer_t* l, const Wml_mat_t* in, const Wml_mat_t* de_dout)
{
	switch (l->type)
	{
		case Layer_linear:
			return layer_linear_backward((Layer_linear_t*)l, in, de_dout);
		case Layer_relu:
			return layer_relu_backward((Layer_relu_t*)l, in, de_dout);
		case Layer_softmax:
			return layer_softmax_backward((Layer_softmax_t*)l, in, de_dout);
	}
	return 0;
}

int layer_update(Layer_t* l)
{
	switch (l->type)
	{
		case Layer_linear:
			return layer_linear_update((Layer_linear_t*)l);
		case Layer_relu:
			return layer_relu_update((Layer_relu_t*)l);
		case Layer_softmax:
			return layer_softmax_update((Layer_softmax_t*)l);
	}
	return 0;
}
