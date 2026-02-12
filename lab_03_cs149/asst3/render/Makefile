
EXECUTABLE := render

CU_FILES   := cudaRenderer.cu benchmark.cu

CU_DEPS    :=

CC_FILES   := main.cpp display.cpp refRenderer.cpp \
              noise.cpp ppm.cpp sceneLoader.cpp

LOGS	   := logs

###########################################################

ARCH := $(shell uname -m)

ifeq ($(ARCH), aarch64)
    CXX = g++-13
else
    CXX = g++-13 -m64
endif

OBJDIR=objs
CXXFLAGS=-O3 -Wall -g
HOSTNAME=$(shell hostname)

LIBS       :=
FRAMEWORKS :=

NVCCFLAGS=-O3 -m64
LIBS += GL glut cudart cuda

LDFLAGS := -L/usr/local/cuda-12.4/lib64

LDLIBS  := $(addprefix -l, $(LIBS))
LDFRAMEWORKS := $(addprefix -framework , $(FRAMEWORKS))

NVCC=nvcc -ccbin /usr/bin/g++-13

OBJS=$(OBJDIR)/main.o $(OBJDIR)/display.o $(OBJDIR)/benchmark.o $(OBJDIR)/refRenderer.o \
     $(OBJDIR)/cudaRenderer.o $(OBJDIR)/noise.o $(OBJDIR)/ppm.o $(OBJDIR)/sceneLoader.o


.PHONY: dirs clean

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *~ $(EXECUTABLE) $(LOGS)

check:	default
		./checker.pl

$(EXECUTABLE): dirs $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $(OBJS) $(LDFLAGS) $(LDLIBS) $(LDFRAMEWORKS)

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@
