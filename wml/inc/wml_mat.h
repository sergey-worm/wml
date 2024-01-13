//#############################################################################
//
//  Matrix represantation.
//
//#############################################################################

#pragma once

#include "wml_cfg.h"

typedef struct
{
	T* arr;            //
	unsigned arr_cap;  // array capabilities in elements T
	unsigned nrow;     //
	unsigned ncol;     //
	int transp;        // transpose flag
} Wml_mat_t;

static inline unsigned mat_nrow(const Wml_mat_t* m) { return m->transp ? m->ncol : m->nrow; }
static inline unsigned mat_ncol(const Wml_mat_t* m) { return m->transp ? m->nrow : m->ncol; }

int wml_mat_print_info(const Wml_mat_t* m, const char* name);

T*   wml_mat_elem_ptr(Wml_mat_t* m, unsigned row, unsigned col);
T    wml_mat_elem_get(const Wml_mat_t* m, unsigned row, unsigned col);
void wml_mat_elem_set(Wml_mat_t* m, unsigned row, unsigned col, T val);

// init matrix, apply init_fun if need
int wml_mat_init(Wml_mat_t* m, T* arr, unsigned arr_cap,
                 unsigned nrow, unsigned ncol, T (*init_func)());

// init matrix with allocating memory
int wml_mat_inita(Wml_mat_t* m, unsigned nrow, unsigned ncol, T (*init_func)());

int wml_mat_clone(Wml_mat_t* dst, const Wml_mat_t* src);
int wml_mat_clonea(Wml_mat_t* dst, const Wml_mat_t* src);
int wml_mat_print(const Wml_mat_t* m, const char* name);
int wml_mat_print_info(const Wml_mat_t* m, const char* name);

// fisher-yates shuffle, can be used for shuffle dataset_x and dataset_y
int wml_mat_shuffle(Wml_mat_t* x, Wml_mat_t* y);

// mr = m1 @ m2
int wml_mat_prod(Wml_mat_t* mr, const Wml_mat_t* m1, const Wml_mat_t* m2,
                 const char* label);
// m[i,j] = m1[i,j] * m2[i,j]
int wml_mat_mult(Wml_mat_t* mr, const Wml_mat_t* m1, const Wml_mat_t* m2,
                 const char* label);

// m[i,j] = m[i,j] * mult / div
int wml_mat_mult_div(Wml_mat_t* m, int mult, int div, const char* label);

int wml_mat_sum(Wml_mat_t* mr, const Wml_mat_t* m1, const Wml_mat_t* m2,
                const char* label);

int wml_mat_sum_cols(Wml_mat_t* mr, const Wml_mat_t* m, const char* label);

// m[i,j] = m1[i,j] - m2[i,j]
int wml_mat_sub(Wml_mat_t* mr, const Wml_mat_t* m1, const Wml_mat_t* m2,
                const char* label);

// m[i,j] -= m1[i,j]
int wml_mat_sub_inplace(Wml_mat_t* mr, const Wml_mat_t* m1, const char* label);

// np.maximum(t, 0)
int wml_mat_relu(Wml_mat_t* m, const char* label);

// m[i,j] = (m[i,j] >= 0)
int wml_mat_relu_deriv(Wml_mat_t* m, const char* label);

// loss func:  mean squared error
int wml_mat_mse(T* res, const Wml_mat_t* m1, const Wml_mat_t* m2,
                const char* label);

// loss func:  cross entropy (m1 is probabilities, m2 is targets)
int wml_mat_cross_entropy(T* res, const Wml_mat_t* m1, const Wml_mat_t* m2,
                          const char* label);

int wml_mat_softmax(Wml_mat_t* m, const char* label);

// return index of max element
int wml_mat_max(T* vec, int sz);

// unit rest for some matrix operations
void wml_mat_test();
