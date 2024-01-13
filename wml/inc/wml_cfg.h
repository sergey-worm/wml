//#############################################################################
//
//
//
//#############################################################################

#pragma once

#define WML_FLOAT_NO   1
#define WML_FLOAT_SOFT 2
#define WML_FLOAT_HARD 3
#define WML_FLOAT WML_FLOAT_HARD

#if WML_FLOAT == WML_FLOAT_NO
# define T int
# define T_spec "%6d"
#elif WML_FLOAT == WML_FLOAT_SOFT
# define T float
# define T_spec "%9.2f"
#elif WML_FLOAT == WML_FLOAT_HARD
# define T float
# define T_spec "%9.2f"
#endif

#define WML_CFG_ALC_CAP (5 * 1024 * 1024)  // allocator capability
