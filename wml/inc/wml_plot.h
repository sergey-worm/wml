//#############################################################################
//
//  Functions to plog grath of loss by using gnuplot in Linux system.
//
//#############################################################################

#pragma once
#include "wml_cfg.h"

// plot array
void wml_plot_array(T* arr, unsigned sz);

// 3 step plot:  step 1 - init
void wml_plot_init();

// 3 step plot:  step 2 - add value
void wml_plot_add_value(T val);

// 3 step plot:  step 3 - plot grath
void wml_plot_make_grath();
