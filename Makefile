###############################################################################
#
#  Main makefile to compile and clean the project.
#
#  Targets:
#
#    make prog     - compile all binaries and build the executable file
#    make rebuild  - clean and build all again
#    make clean    - clean all binaries and executable
#
###############################################################################

CC     = gcc
LD     = gcc
BIN    = prog
BLDDIR = bld
WMLDIR = wml/src

# sanitizer flags
SANFLAGS  = -fsanitize=address -fsanitize=undefined -fno-sanitize-recover=all
SANFLAGS += -fsanitize=float-divide-by-zero -fsanitize=float-cast-overflow
SANFLAGS += -fno-sanitize=null -fno-sanitize=alignment

CFLAGS  = -Wall -Werror -Wpedantic -Iwml/inc
CFLAGS += $(SANFLAGS)
CFLAGS += -MMD -MP  # generate dependencies

LDFLAGS = -lasan -lubsan -lm

SRCS  = main.c 
SRCS += $(WMLDIR)/wml_utils.c
SRCS += $(WMLDIR)/wml_mat.c
SRCS += $(WMLDIR)/wml_plot.c
SRCS += $(WMLDIR)/wml_dl_static.c
SRCS += $(WMLDIR)/wml_layers.c
SRCS += $(WMLDIR)/wml_data_loaders.c

OBJS = $(SRCS:%.c=$(BLDDIR)/%.o)
DEPS = $(SRCS:%.c=$(BLDDIR)/%.d)

$(BLDDIR)/$(BIN):  Makefile $(OBJS)
	$(LD) -o $@ $(OBJS) $(LDFLAGS)

# use dependencies
-include $(DEPS)

$(BLDDIR)/%.o:  %.c Makefile
	mkdir -p $(shell dirname $@)
	$(CC) $(CFLAGS) -c $< -o $@
	@echo

clean:
	rm -f $(BLDDIR)/$(BIN)
	rm -f $(OBJS)
	rm -f $(DEPS)

rebuild:
	make clean
	make
