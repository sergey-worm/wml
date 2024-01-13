//#############################################################################
//
//  Layer abstraction and implementation.
//
//#############################################################################

#pragma once

#include "wml_mat.h"
#include "wml_utils.h"

//-----------------------------------------------------------------------------
// Linear layer
//-----------------------------------------------------------------------------

// Linear layer:  out = in * w + b
typedef struct
{
	// comon layer params
	int type;
	Wml_mat_t out;        // allocated inside forward(),  released on request
	Wml_mat_t de_din;     // allocated inside backward(), released on request
	// specific for linear layer params
	unsigned  dim_in;
	unsigned  dim_out;
	unsigned  alpha_mul;
	unsigned  alpha_div;
	Wml_mat_t w;
	Wml_mat_t b;
	Wml_mat_t de_dw;
	Wml_mat_t de_db;
	T (*init_func)();
} Layer_linear_t;


int layer_linear_init(Layer_linear_t* l);
int layer_linear_fini(Layer_linear_t* l);
int layer_linear_print(Layer_linear_t* l);

// out = in @ w + b
// in->nrow is equil to batch_size
// in->ncol is equil to dim_in
int layer_linear_forward(Layer_linear_t* l, const Wml_mat_t* in);

// calc de_db and de_dw, fill de_din
int layer_linear_backward(Layer_linear_t* l, const Wml_mat_t* in,
                          const Wml_mat_t* de_dout);

int layer_linear_update(Layer_linear_t* l);

//-----------------------------------------------------------------------------
// Relu layer
//-----------------------------------------------------------------------------
// Relu layer:  out = relu(in)
typedef struct
{
	int type;
	Wml_mat_t out;        // allocated inside forward(),  released on request
	Wml_mat_t de_din;     // allocated inside backward(), released on request
} Layer_relu_t;

int layer_relu_init(Layer_relu_t* l);
int layer_relu_fini(Layer_relu_t* l);
int layer_relu_print(Layer_relu_t* l);
int layer_relu_forward(Layer_relu_t* l, const Wml_mat_t* in);
int layer_relu_backward(Layer_relu_t* l, const Wml_mat_t* in,
                        const Wml_mat_t* de_dout);
int layer_relu_update(Layer_relu_t* l);

//-----------------------------------------------------------------------------
// Softmax layer
//-----------------------------------------------------------------------------
// Relu layer:  out = suftmax(in)
typedef struct
{
	int type;
	Wml_mat_t out;        // allocated inside forward(),  released on request
	Wml_mat_t de_din;     // allocated inside backward(), released on request
} Layer_softmax_t;

int layer_softmax_init(Layer_softmax_t* l);
int layer_softmax_fini(Layer_softmax_t* l);
int layer_softmax_print(Layer_softmax_t* l);
int layer_softmax_forward(Layer_softmax_t* l, const Wml_mat_t* in);
int layer_softmax_backward(Layer_softmax_t* l, const Wml_mat_t* in,
                           const Wml_mat_t* de_dout);
int layer_softmax_dedin(Layer_softmax_t* l, const Wml_mat_t* target_y);
int layer_softmax_update(Layer_softmax_t* l);

//-----------------------------------------------------------------------------
// API - abstract layer
//-----------------------------------------------------------------------------
typedef struct
{
	int type;
	Wml_mat_t out;        // allocated inside forward(),  released on request
	Wml_mat_t de_din;     // allocated inside backward(), released on request
} Layer_t;

enum
{
	Layer_linear  = 1,
	Layer_relu    = 2,
	Layer_softmax = 3,
};

int layer_init(Layer_t* l);
int layer_fini(Layer_t* l);
int layer_print(Layer_t* l);
int layer_free_out_matrix(Layer_t* l);
int layer_free_dedin_matrix(Layer_t* l);
int layer_forward(Layer_t* l, const Wml_mat_t* in);
int layer_backward(Layer_t* l, const Wml_mat_t* in, const Wml_mat_t* de_dout);
int layer_update(Layer_t* l);
