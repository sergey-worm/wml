//#############################################################################
//
//  Collection of utilites for WML.
//
//#############################################################################

#include "wml_utils.h"
#include <assert.h>

//-----------------------------------------------------------------------------
// Linear congruential generator (pseudo-random numbers)
//-----------------------------------------------------------------------------
unsigned wml_rand()
{
	static unsigned seed = 42;  // initial value should be changed
	unsigned a = 1103515245;
	unsigned c = 12345;
	unsigned m = 2147483648;    // 2^31, should be power of 2
	seed = (a * seed + c) % m;
	return seed;
}

//-----------------------------------------------------------------------------
// Dummy memory stack allocator of arrays of type=T.
// It warrks as stack allocator:
//    allocating and deallocating always on the top of memory area (stack).
//-----------------------------------------------------------------------------

enum
{
	Wml_alc_cap = WML_CFG_ALC_CAP,     // allocator capability
	Wml_alc_chank_info_sz = sizeof(T), // size of chunk info
	                                   // (area before chank with chunk number)
};

struct
{
	uint8_t space[Wml_alc_cap];
	unsigned cnt;
} mem;

void wml_allocator_init()
{
	mem.cnt = 0;
}

void* wml_alloc(unsigned num)
{
	// allign num to sizeof(T) to up
	if (num & (sizeof(T) - 1))
		num = (num & ~(sizeof(T) - 1)) + sizeof(T);

	uint8_t* ret = mem.space + mem.cnt + Wml_alc_chank_info_sz;

	assert(!((uintptr_t)ret & (sizeof(T) - 1)));

	if ((mem.cnt + num) > Wml_alc_cap)
		ret = 0;  // out of memory

	assert(ret && "out of memory");
	if (!ret)
		return 0;

	// write chank info before the chunk
	*(unsigned*)(ret - Wml_alc_chank_info_sz) = num;

	// store new mem.cnt
	mem.cnt += num + Wml_alc_chank_info_sz;
	return ret;
}

T* wml_alloct(unsigned num)
{
	return wml_alloc(num * sizeof(T));
}

// not safe
int wml_free(void* p)
{
	uint8_t* ptr = (uint8_t*) p;

	unsigned new_cnt = ptr - mem.space - Wml_alc_chank_info_sz;
	unsigned num = mem.cnt - new_cnt;
	assert(num <= mem.cnt);

	// check chank info
	unsigned _num = *(unsigned*)(ptr - Wml_alc_chank_info_sz);
	assert((num - Wml_alc_chank_info_sz) == _num && "Wrong dealloc order!");

	// store new mem.cnt
	mem.cnt -= num;
	return 0;
}

int wml_get_mem_consum()
{
	return mem.cnt;
}
