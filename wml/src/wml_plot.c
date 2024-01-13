//#############################################################################
//
//  Functions to plog grath of loss by using gnuplot in Linux system.
//
//#############################################################################

#include "wml_plot.h"
#include <stdio.h>

// plot array
void wml_plot_array(T* arr, unsigned sz)
{
	FILE* gnuplotPipe = popen("gnuplot -persist", "w");
	if (!gnuplotPipe)
	{
		printf("No se pudo abrir GNUPLOT.\n");
		return;
	}

	fprintf(gnuplotPipe, "plot '-' with lines\n");
	for (int i=0; i<sz; ++i)
		fprintf(gnuplotPipe, "%f\n", arr[i]);
	fprintf(gnuplotPipe, "e\n");
	fflush(gnuplotPipe);
	fprintf(gnuplotPipe, "exit\n");
	pclose(gnuplotPipe);
}

static FILE* s_gnuplotPipe = 0;

// 3 step plot:  step 1 - init
void wml_plot_init()
{
	s_gnuplotPipe = popen("gnuplot -persist", "w");
	if (!s_gnuplotPipe)
	{
		printf("%s:  ERR:  No se pudo abrir GNUPLOT.\n", __func__);
		return;
	}
	fprintf(s_gnuplotPipe, "plot '-' with lines\n");
}

// 3 step plot:  step 2 - add value
void wml_plot_add_value(T val)
{
	if (!s_gnuplotPipe)
	{
		printf("%s:  ERR:  Call plot_init() first.\n", __func__);
		return;
	}
	fprintf(s_gnuplotPipe, "%f\n", val);
}

// 3 step plot:  step 3 - plot grath
void wml_plot_make_grath()
{
	if (!s_gnuplotPipe)
	{
		printf("%s:  ERR:  Call plot_init() first.\n", __func__);
		return;
	}
	fprintf(s_gnuplotPipe, "e\n");
	fflush(s_gnuplotPipe);
	fprintf(s_gnuplotPipe, "exit\n");
	pclose(s_gnuplotPipe);
}
