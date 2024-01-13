//#############################################################################
//
//  Matrix represantation.
//
//#############################################################################

#include "wml_mat.h"
#include "wml_utils.h"
#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>

#if 0  // measure CPU load to profile
# include <time.h>
# define MEAS_START()                \
	static unsigned cnt = 0;         \
	static double all_time_used = 0; \
	clock_t start, end;              \
	double cpu_time_used;            \
	start = clock();
# define MEAS_END()                                   \
	end = clock();                                    \
	cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC; \
	cnt++;                                            \
	all_time_used += cpu_time_used;                   \
	printf("%s:  cnt=%d, cpu=%f, all_cpu=%f.\n",      \
		__func__, cnt, cpu_time_used, all_time_used);
#else
# define MEAS_START()
# define MEAS_END()
#endif

T* wml_mat_elem_ptr(Wml_mat_t* m, unsigned row, unsigned col)
{
	unsigned idx = m->transp ?
	               col * mat_nrow(m) + row :
	               row * mat_ncol(m) + col;
	#if 0
	printf("%s:  mat:%ux%u=%u:  inc:  row=%u, col=%d -> idx=%u.\n", __func__,
		mat_nrow(m), mat_ncol(m), m->arr_cap, row, col, idx);
	#endif
	#if 1 // DBG
	if (col >= mat_ncol(m))
	{
		wml_mat_print_info(m, "elem_ptr(m)");
		printf("%s:  mat:%ux%u=%u:  inc:  row=%u, col=%u -> idx=%u.\n", __func__,
			mat_nrow(m), mat_ncol(m), m->arr_cap, row, col, idx);
		assert(col < mat_ncol(m) && "wrong col for this matrix");
	}
	#endif // ~DBG
	assert(row < mat_nrow(m));
	assert(col < mat_ncol(m));
	assert(idx < m->arr_cap);
	assert(idx < (m->nrow * m->ncol));
	return &m->arr[idx];
}

T wml_mat_elem_get(const Wml_mat_t* m, unsigned row, unsigned col)
{
	return *wml_mat_elem_ptr((Wml_mat_t*)m, row, col);
}

void wml_mat_elem_set(Wml_mat_t* m, unsigned row, unsigned col, T val)
{
	*wml_mat_elem_ptr(m, row, col) = val;
}

// init matrix, apply init_fun if need
int wml_mat_init(Wml_mat_t* m, T* arr, unsigned arr_cap,
                 unsigned nrow, unsigned ncol, T (*init_func)())
{
	MEAS_START()
	m->arr     = arr;
	m->arr_cap = arr_cap;
	m->nrow    = nrow;
	m->ncol    = ncol;
	m->transp  = 0;

	// init elements if need
	if (init_func)
		for (unsigned i=0; i<m->nrow; ++i)
			for (unsigned j=0; j<m->ncol; ++j)
					wml_mat_elem_set(m, i, j, init_func());
	MEAS_END()
	return 0;
}

// init matrix with allocating memory
int wml_mat_inita(Wml_mat_t* m, unsigned nrow, unsigned ncol, T (*init_func)())
{
	return wml_mat_init(m, wml_alloct(nrow*ncol), nrow*ncol, nrow, ncol, init_func);
}

int wml_mat_clone(Wml_mat_t* dst, const Wml_mat_t* src)
{
	MEAS_START()
	#if 0
	printf("%s:  dst.cap=%u:  src=(%u %u).\n", __func__,
		dst->arr_cap, src->nrow, src->ncol);
	#endif
	assert(dst->arr_cap >=  src->nrow * src->ncol);
	dst->nrow = src->nrow;
	dst->ncol = src->ncol;
	dst->transp = src->transp;

	for (unsigned i=0; i<dst->nrow; ++i)
		for (unsigned j=0; j<dst->ncol; ++j)
			wml_mat_elem_set(dst, i, j, wml_mat_elem_get(src, i, j));

	MEAS_END()
	return 0;
}

int wml_mat_clonea(Wml_mat_t* dst, const Wml_mat_t* src)
{
	wml_mat_inita(dst, src->nrow, src->ncol, NULL);
	dst->transp = src->transp;

	for (unsigned i=0; i<dst->nrow; ++i)
		for (unsigned j=0; j<dst->ncol; ++j)
			wml_mat_elem_set(dst, i, j, wml_mat_elem_get(src, i, j));

	return 0;
}

int wml_mat_print(const Wml_mat_t* m, const char* name)
{
	unsigned nrow = mat_nrow(m);
	unsigned ncol = mat_ncol(m);

	printf("Matrix '%s' %ux%u, cap=%u, t=%d:\n", name, m->nrow, m->ncol, m->arr_cap, m->transp);
	for (unsigned i=0; i<nrow; ++i)
	{
		for (unsigned j=0; j<ncol; ++j)
			printf("  " T_spec, wml_mat_elem_get(m, i, j));
		printf("\n");
	}
	return 0;
}

int wml_mat_print_info(const Wml_mat_t* m, const char* name)
{
	printf("Matrix '%s' %ux%u, cap=%u, t=%d:\n", name, m->nrow, m->ncol, m->arr_cap, m->transp);
	return 0;
}

// fisher-yates shuffle, could be used for shuffle dataset_x and dataset_y
int wml_mat_shuffle(Wml_mat_t* x, Wml_mat_t* y)
{
	MEAS_START()
	assert(mat_nrow(x) == mat_nrow(y));
	unsigned size = mat_nrow(x);

	for (int i=size-1; i>0; --i)
	{
		int j = wml_rand() % (i + 1);
		enum { Max = 16 };
		T tmp[Max];

		// swap rows x[i] and x[j]
		assert(mat_ncol(x) < Max);
		unsigned row_sz = sizeof(T) * mat_ncol(x);
		T* ptr_i = wml_mat_elem_ptr(x, i, 0);
		T* ptr_j = wml_mat_elem_ptr(x, j, 0);
		memcpy(tmp, ptr_i, row_sz);
		memcpy(ptr_i, ptr_j, row_sz);
		memcpy(ptr_j, tmp, row_sz);

		// swap rows y[i] and y[j]
		assert(mat_ncol(y) < Max);
		row_sz = sizeof(T) * mat_ncol(y);
		ptr_i = wml_mat_elem_ptr(y, i, 0);
		ptr_j = wml_mat_elem_ptr(y, j, 0);
		memcpy(tmp, ptr_i, row_sz);
		memcpy(ptr_i, ptr_j, row_sz);
		memcpy(ptr_j, tmp, row_sz);
	}
	MEAS_END()
	return 0;
}

// mr = m1 @ m2
int wml_mat_prod(Wml_mat_t* mr, const Wml_mat_t* m1, const Wml_mat_t* m2,
                 const char* label)
{
	MEAS_START()
	if (label)
		printf("prod:  %s:  mr.%ux%u=%u = m1.%ux%u @ m2.%ux%u.\n", label,
			mat_nrow(mr), mat_ncol(mr), mr->arr_cap,
			mat_nrow(m1), mat_ncol(m1),
			mat_nrow(m2), mat_ncol(m2));

	assert(mat_ncol(m1) == mat_nrow(m2));
	assert(mr->arr_cap >= mat_nrow(m1) * mat_ncol(m2));

	mr->nrow = mat_nrow(m1);
	mr->ncol = mat_ncol(m2);
	mr->transp = 0;
	memset(mr->arr, 0, mr->arr_cap * sizeof(T));

	// escape use of elem_get() and elem_set() inside nested cycle
	// to decrease CPU load
	unsigned s = mat_ncol(m1);
	#if 0  // conditions are inside of nested cycles
	for (unsigned ri=0; ri<mr->nrow; ++ri)      // for each result matrix row
	{
		for (unsigned rj=0; rj<mr->ncol; ++rj)  // for each result matrix col
		{
			for (unsigned i=0; i<s; ++i)  // for each elem in m1.col/m2.row
			{
				// *elem_ptr(mr, ri, rj) += elem_get(m1, ri, i) * elem_get(m2, i, rj);
				T t = 0;
				if (!m1->transp && !m2->transp)
					t = m1->arr[ri * m1->ncol + i]  *  m2->arr[i * m2->ncol + rj];
				else if (m1->transp && !m2->transp)
					t = m1->arr[i * m1->ncol + ri]  *  m2->arr[i * m2->ncol + rj];
				else if (!m1->transp && m2->transp)
					t = m1->arr[ri * m1->ncol + i]  *  m2->arr[rj * m2->ncol + i];
				else // m1->transp && m2->transp
					t = m1->arr[i * m1->ncol + ri]  *  m2->arr[rj * m2->ncol + i];
				mr->arr[ri * mr->ncol + rj] += t;
			}
		}
	}
	#else // conditions are outside of nested cycles, it decreased CPU load
	if (!m1->transp && !m2->transp)
		for (unsigned ri=0; ri<mr->nrow; ++ri)      // for each result matrix row
			for (unsigned rj=0; rj<mr->ncol; ++rj)  // for each result matrix col
				for (unsigned i=0; i<s; ++i)        // for each elem in m1.col/m2.row
					mr->arr[ri * mr->ncol + rj] +=
						m1->arr[ri * m1->ncol + i] * m2->arr[i * m2->ncol + rj];
	else if (m1->transp && !m2->transp)
		for (unsigned ri=0; ri<mr->nrow; ++ri)      // for each result matrix row
			for (unsigned rj=0; rj<mr->ncol; ++rj)  // for each result matrix col
				for (unsigned i=0; i<s; ++i)        // for each elem in m1.col/m2.row
					mr->arr[ri * mr->ncol + rj] +=
						m1->arr[i * m1->ncol + ri] * m2->arr[i * m2->ncol + rj];
	else if (!m1->transp && m2->transp)
		for (unsigned ri=0; ri<mr->nrow; ++ri)      // for each result matrix row
			for (unsigned rj=0; rj<mr->ncol; ++rj)  // for each result matrix col
				for (unsigned i=0; i<s; ++i)        // for each elem in m1.col/m2.row
					mr->arr[ri * mr->ncol + rj] +=
						m1->arr[ri * m1->ncol + i] * m2->arr[rj * m2->ncol + i];
	else // m1->transp && m2->transp
		for (unsigned ri=0; ri<mr->nrow; ++ri)      // for each result matrix row
			for (unsigned rj=0; rj<mr->ncol; ++rj)  // for each result matrix col
				for (unsigned i=0; i<s; ++i)        // for each elem in m1.col/m2.row
					mr->arr[ri * mr->ncol + rj] +=
						m1->arr[i * m1->ncol + ri] * m2->arr[rj * m2->ncol + i];
	#endif
	MEAS_END()
	return 0;
}

// m[i,j] = m1[i,j] * m2[i,j]
int wml_mat_mult(Wml_mat_t* mr, const Wml_mat_t* m1, const Wml_mat_t* m2,
                 const char* label)
{
	MEAS_START()
	if (label)
		printf("sub:  %s:  mr.%ux%u.%u = m1.%ux%u * m2.%ux%u.\n", label,
			mat_nrow(mr), mat_ncol(mr), mr->arr_cap,
			mat_nrow(m1), mat_ncol(m1),
			mat_nrow(m2), mat_ncol(m2));

	assert(mat_nrow(m1) == mat_nrow(m2));
	assert(mat_ncol(m1) == mat_ncol(m2));

	mr->nrow = mat_nrow(m1);
	mr->ncol = mat_ncol(m1);
	mr->transp = 0;

	for (unsigned i=0; i<mr->nrow; ++i)
		for (unsigned j=0; j<mr->ncol; ++j)
			wml_mat_elem_set(mr, i, j,
				wml_mat_elem_get(m1, i, j) * wml_mat_elem_get(m2, i, j));

	MEAS_END()
	return 0;
}

// m[i,j] = m[i,j] * mult / div
int wml_mat_mult_div(Wml_mat_t* m, int mult, int div, const char* label)
{
	MEAS_START()
	if (label)
		printf("mult_div:  %s:  m.%ux%u.%u.\n", label,
			mat_nrow(m), mat_ncol(m), m->arr_cap);

	unsigned nrow = mat_nrow(m);
	unsigned ncol = mat_ncol(m);

	for (unsigned i=0; i<nrow; ++i)
		for (unsigned j=0; j<ncol; ++j)
			wml_mat_elem_set(m, i, j, wml_mat_elem_get(m, i, j) * mult / div);

	MEAS_END()
	return 0;
}

int wml_mat_sum(Wml_mat_t* mr, const Wml_mat_t* m1, const Wml_mat_t* m2,
                const char* label)
{
	MEAS_START()
	if (label)
		printf("sum:  %s:  mr.%ux%u.%u = m1.%ux%u + m2.%ux%u.\n", label,
			mat_nrow(mr), mat_ncol(mr), mr->arr_cap,
			mat_nrow(m1), mat_ncol(m1),
			mat_nrow(m2), mat_ncol(m2));

	assert(mat_ncol(m1) == mat_ncol(m2));
	assert(mat_nrow(m2) == 1);  // now is supported only matrix with 1 row
	assert(mr->arr_cap >= (mat_nrow(m1) * mat_ncol(m1)));

	mr->nrow = mat_nrow(m1);
	mr->ncol = mat_ncol(m1);
	mr->transp = 0;

	for (unsigned i=0; i<mr->nrow; ++i)
		for (unsigned j=0; j<mr->ncol; ++j)
			wml_mat_elem_set(mr, i, j,
				wml_mat_elem_get(m1, i, j) + wml_mat_elem_get(m2, 0, j));

	MEAS_END()
	return 0;
}

// mr[i] = sum_j(m[i,j])
int wml_mat_sum_cols(Wml_mat_t* mr, const Wml_mat_t* m, const char* label)
{
	MEAS_START()
	if (label)
		printf("sum_cols:  %s:  mr.%ux%u.%u = m.%ux%u.\n", label,
			mat_nrow(mr), mat_ncol(mr), mr->arr_cap, mat_nrow(m), mat_ncol(m));

	assert(mat_ncol(mr) == mat_ncol(m));
	assert(mat_nrow(mr) == 1);

	unsigned nrow = mat_nrow(m);
	unsigned ncol = mat_ncol(m);

	for (unsigned j=0; j<ncol; ++j)
	{
		wml_mat_elem_set(mr, 0, j, 0);
		for (unsigned i=0; i<nrow; ++i)
			*wml_mat_elem_ptr(mr, 0, j) += wml_mat_elem_get(m, i, j);
	}
	MEAS_END()
	return 0;
}

// m[i,j] = m1[i,j] - m2[i,j]
int wml_mat_sub(Wml_mat_t* mr, const Wml_mat_t* m1, const Wml_mat_t* m2,
                const char* label)
{
	MEAS_START()
	if (label)
		printf("sub:  %s:  mr.%ux%u.%u = m1.%ux%u - m2.%ux%u.\n", label,
			mat_nrow(mr), mat_ncol(mr), mr->arr_cap,
			mat_nrow(m1), mat_ncol(m1),
			mat_nrow(m2), mat_ncol(m2));

	assert(mat_nrow(m1) == mat_nrow(m2));
	assert(mat_ncol(m1) == mat_ncol(m2));
	assert(mat_nrow(m1) == mat_nrow(mr));
	assert(mat_ncol(m1) == mat_ncol(mr));

	unsigned nrow = mat_nrow(m1);
	unsigned ncol = mat_ncol(m1);

	for (unsigned i=0; i<nrow; ++i)
		for (unsigned j=0; j<ncol; ++j)
			wml_mat_elem_set(mr, i, j,
				wml_mat_elem_get(m1, i, j) - wml_mat_elem_get(m2, i, j));

	MEAS_END()
	return 0;
}

// m[i,j] -= m1[i,j]
int wml_mat_sub_inplace(Wml_mat_t* mr, const Wml_mat_t* m1, const char* label)
{
	MEAS_START()
	if (label)
		printf("sub_inplace:  %s:  mr.%ux%u.%u -= m1.%ux%u.\n", label,
			mat_nrow(mr), mat_ncol(mr), mr->arr_cap, mat_nrow(m1), mat_ncol(m1));

	assert(mat_nrow(mr) == mat_nrow(m1));
	assert(mat_ncol(mr) == mat_ncol(m1));

	unsigned nrow = mat_nrow(m1);
	unsigned ncol = mat_ncol(m1);

	for (unsigned i=0; i<nrow; ++i)
		for (unsigned j=0; j<ncol; ++j)
			*wml_mat_elem_ptr(mr, i, j) -= wml_mat_elem_get(m1, i, j);

	MEAS_END()
	return 0;
}

// np.maximum(t, 0)
int wml_mat_relu(Wml_mat_t* m, const char* label)
{
	MEAS_START()
	if (label)
		printf("relu:  %s:  m.%ux%u.%u.\n", label,
			mat_nrow(m), mat_ncol(m), m->arr_cap);

	for (unsigned i=0; i<m->nrow; ++i)
		for (unsigned j=0; j<m->ncol; ++j)
			wml_mat_elem_set(m, i, j, MAX(wml_mat_elem_get(m, i, j), 0));

	MEAS_END()
	return 0;
}

// m[i,j] = (m[i,j] >= 0)
int wml_mat_relu_deriv(Wml_mat_t* m, const char* label)
{
	MEAS_START()
	if (label)
		printf("relu_deriv:  %s:  m.%ux%u.%u.\n", label,
			mat_nrow(m), mat_ncol(m), m->arr_cap);

	for (unsigned i=0; i<m->nrow; ++i)
		for (unsigned j=0; j<m->ncol; ++j)
			wml_mat_elem_set(m, i, j, wml_mat_elem_get(m, i, j) >= 0);

	MEAS_END()
	return 0;
}

// loss func:  mean squared error
int wml_mat_mse(T* res, const Wml_mat_t* m1, const Wml_mat_t* m2,
                const char* label)
{
	MEAS_START()
	if (label)
		printf("mse:  %s:  m1.%ux%u.%u | m2.%u.%u.%u.\n", label,
			mat_nrow(m1), mat_ncol(m1), m1->arr_cap,
			mat_nrow(m2), mat_ncol(m2), m2->arr_cap);

	assert(mat_ncol(m1) == mat_ncol(m2));
	assert(mat_nrow(m1) == mat_nrow(m2));

	unsigned nrow = mat_nrow(m1);
	unsigned ncol = mat_ncol(m1);

	T sum = 0;
	for (unsigned i=0; i<nrow; ++i)
	{
		for (unsigned j=0; j<ncol; ++j)
		{
			T diff = wml_mat_elem_get(m1, i, j) - wml_mat_elem_get(m2, i, j);
			sum += diff * diff;
		}
	}

	T mse = sum / (nrow * ncol);
	*res = mse;

	MEAS_END()
	return 0;
}

// loss func:  cross entropy (m1 is probabilities, m2 is targets)
int wml_mat_cross_entropy(T* res, const Wml_mat_t* m1, const Wml_mat_t* m2,
                          const char* label)
{
	MEAS_START()
	if (label)
		printf("cross entropy:  %s:  m1.%ux%u.%u | m2.%u.%u.%u.\n", label,
			mat_nrow(m1), mat_ncol(m1), m1->arr_cap,
			mat_nrow(m2), mat_ncol(m2), m2->arr_cap);

	assert(mat_ncol(m1) == mat_ncol(m2));
	assert(mat_nrow(m1) == mat_nrow(m2));

	unsigned nrow = mat_nrow(m1);
	unsigned ncol = mat_ncol(m1);

	T sum = 0;
	for (unsigned i=0; i<nrow; ++i)
		for (unsigned j=0; j<ncol; ++j)
			// OPTIMIZTION: first multimplier is not 0
			if (wml_mat_elem_get(m2, i, j) != 0)
				// WA:  add small value to avoid log(0)
				sum -= wml_mat_elem_get(m2, i, j) *
				                       log(wml_mat_elem_get(m1, i, j) + 1e-10);

	*res = sum;

	MEAS_END()
	return 0;
}

// Apply classical softmax() to array.
static void softmax_exp(T* vec, int sz)
{
	// find max value in vec
	T max = vec[0];
	for (int i=1; i<sz; i++)
		if (vec[i] > max)
			max = vec[i];

	// decrease all values by max value to avoid float overflown in exp()
	for (int i=0; i<sz; i++)
		vec[i] -= max;

	// apply exp
	for (int i=0; i<sz; i++)
		vec[i] = exp(vec[i]);

	// calc sum
	T sum = 0.0;
	for (int i=0; i<sz; i++)
		sum += vec[i];
	assert(sum && "Sum of all vec elements is 0, need to process it.");

	// normalize
	for (int i=0; i<sz; i++)
		vec[i] = vec[i] / sum;
}

// Apply linear softmax() to array.
// NOTE:  Using softmax with exponent is not good for embedded CPU or FPU.
// NOTE:  Use activation func (relu) if you ise softmax lineary instead of exp.
static void softmax_linear(T* vec, int sz)
{
	// find min value in vec
	T min = vec[0];
	for (int i=1; i<sz; i++)
		if (vec[i] < min)
			min = vec[i];

	// decrease all values by min value, calc sum
	T sum = 0.0;
	for (int i=0; i<sz; i++)
	{
		vec[i] -= min;
		sum += vec[i];
	}
	assert(sum && "Sum of all vec elements is 0, need to process it.");
	if (sum != 0)
	{
		printf("WRN:  Sum of all vec elements is 0.");
		// normalize
		for (int i=0; i<sz; i++)
			vec[i] = vec[i] / sum;
	}
	else
	{
		// CHECK ME
		for (int i=0; i<sz; i++)
			vec[i] = 1 / sz;  // equal probabilities
	}
}

int wml_mat_softmax(Wml_mat_t* m, const char* label)
{
	MEAS_START()
	if (label)
		printf("sfmax:  %s:  m.%ux%u.%u.\n", label,
			mat_nrow(m), mat_ncol(m), m->arr_cap);

	for (unsigned i=0; i<m->nrow; ++i)
		if (1)
			softmax_exp(wml_mat_elem_ptr(m, i, 0), m->ncol);
		else
			softmax_linear(wml_mat_elem_ptr(m, i, 0), m->ncol);

	MEAS_END()
	return 0;
}

// return index of max element
int wml_mat_max(T* vec, int sz)
{
	MEAS_START()
	assert(sz >= 1);

	unsigned id = 0;
	float mx = vec[0];
	for (int i=1; i<sz; ++i)
	{
		if (vec[i] > mx)
		{
			mx = vec[i];
			id = i;
		}
	}
	MEAS_END()
	return id;
}

// unit rest for some matrix operations
void wml_mat_test()
{
	// base test
	if (1)
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

	// test of wml_mat_prod()
	if (1)
	{
		T arr1[] = {1,2,3,4,5,6};
		Wml_mat_t m1;
		wml_mat_init(&m1, arr1, 6, 3, 2, 0);
		wml_mat_print(&m1, "m1");

		T arr2[] = {7,8,9,10,11,12,13,14};
		Wml_mat_t m2;
		wml_mat_init(&m2, arr2, 8, 2, 4, 0);
		wml_mat_print(&m2, "m2");

		Wml_mat_t m3;
		wml_mat_init(&m3, wml_alloct(12), 12, 0, 0, 0);

		printf("\nTest:  !m1.transp && !m2.transp:\n");
		m1.transp = 0;
		m2.transp = 0;
		wml_mat_print(&m1, "m1");
		wml_mat_print(&m2, "m2");
		wml_mat_prod(&m3, &m1, &m2, NULL);
		wml_mat_print(&m3, "m3");
		assert(m3.arr_cap == 12);
		T expected1[] = { 29,32,35,38, 65,72,79,86, 101,112,123,134 };
		for (unsigned i=0; i<m3.arr_cap; ++i)
			assert(m3.arr[i] == expected1[i]);

		printf("\nTest:  m1.transp && !m2.transp:\n");
		wml_mat_init(&m1, arr1, 6, 2, 3, 0);
		wml_mat_init(&m2, arr2, 8, 2, 4, 0);
		m1.transp = 1;
		m2.transp = 0;
		wml_mat_print(&m1, "m1");
		wml_mat_print(&m2, "m2");
		wml_mat_prod(&m3, &m1, &m2, NULL);
		wml_mat_print(&m3, "m3");
		assert(m3.arr_cap == 12);
		T expected2[] = { 51,56,61,66, 69,76,83,90, 87,96,105,114 };
		for (unsigned i=0; i<m3.arr_cap; ++i)
			assert(m3.arr[i] == expected2[i]);

		printf("\nTest:  !m1.transp && m2.transp:\n");
		wml_mat_init(&m1, arr1, 6, 3, 2, 0);
		wml_mat_init(&m2, arr2, 8, 4, 2, 0);
		m1.transp = 0;
		m2.transp = 1;
		wml_mat_print(&m1, "m1");
		wml_mat_print(&m2, "m2");
		wml_mat_prod(&m3, &m1, &m2, NULL);
		wml_mat_print(&m3, "m3");
		assert(m3.arr_cap == 12);
		T expected3[] = { 23,29,35,41, 53,67,81,95, 83,105,127,149 };
		for (unsigned i=0; i<m3.arr_cap; ++i)
			assert(m3.arr[i] == expected3[i]);

		printf("\nTest:  m1.transp && m2.transp:\n");
		wml_mat_init(&m1, arr1, 6, 2, 3, 0);
		wml_mat_init(&m2, arr2, 8, 4, 2, 0);
		m1.transp = 1;
		m2.transp = 1;
		wml_mat_print(&m1, "m1");
		wml_mat_print(&m2, "m2");
		wml_mat_prod(&m3, &m1, &m2, NULL);
		wml_mat_print(&m3, "m3");
		assert(m3.arr_cap == 12);
		T expected4[] = { 39,49,59,69, 54,68,82,96, 69,87,105,123 };
		for (unsigned i=0; i<m3.arr_cap; ++i)
			assert(m3.arr[i] == expected4[i]);
	}
}
