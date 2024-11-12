INCPATHS = -I$(UTIL_DIR)

BENCHMARK = $(shell basename `pwd`)
EXE = $(BENCHMARK)_acc
SRC = $(BENCHMARK).c
HEADERS = $(BENCHMARK).h

SRC += $(UTIL_DIR)/polybench.c

DEPS        := Makefile.dep
DEP_FLAG    := -MM

CC=clang
LD=ld
OBJDUMP=objdump

OPT=-O2 -g -Wall -pg -fopenmp
OMP=-fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda

CFLAGS=$(OPT) $(OMP) -I. $(EXT_CFLAGS)
LDFLAGS=-lm $(EXT_LDFLAGS)

.PHONY: all exe clean veryclean

all : exe

exe : $(EXE)

$(EXE) : $(SRC)
	$(CC) $(CFLAGS) $(INCPATHS) $^ -o $@ $(LDFLAGS)

clean :
	-rm -vf -vf $(EXE) *~ 

veryclean : clean
	-rm -vf $(DEPS)

run: $(EXE)
	./$(EXE)

profile: $(EXE)
	sudo LD_LIBRARY_PATH=/usr/local/cuda/lib:/usr/ext/lib:${LD_LIBRARY_PATH} LIBRARY_PATH=/usr/ext/lib:${LIBRARY_PATH} /usr/local/cuda/bin/nvprof ./$(EXE) $(EXT_ARGS)

$(DEPS): $(SRC) $(HEADERS)
	$(CC) $(INCPATHS) $(DEP_FLAG) $(SRC) > $(DEPS)

-include $(DEPS)
