


# the name of the executable file
#######################################################################################################
EXECUTABLE := PoissonGravity



# simulation options
#######################################################################################################
# use GPU (must be turned on)
SIMU_OPTION += -DGPU

# Poisson solver: SOR/MG (successive-overrelaxation/multigrid)
SIMU_OPTION += -DPOT_SCHEME=SOR

# intel compiler (default: GNU compiler)
SIMU_OPTION += -DINTEL

# double precision
#SIMU_OPTION += -DFLOAT8

# GPU architecture: FERMI/KEPLER/MAXWELL/PASCAL
SIMU_OPTION += -DGPU_ARCH=KEPLER

# debug mode
#SIMU_OPTION += -DGAMER_DEBUG

# enable OpenMP parallelization
SIMU_OPTION += -DOPENMP

# self-gravity (always turn on here)
SIMU_OPTION += -DGRAVITY



# simulation parameters
#######################################################################################################
PATCH_SIZE           = 8      # number of grids of a single patch in x, y, and z direction
NEWTON_G             = 1.e-5  # Newtonian gravity constant
POT_GHOST_SIZE       = 5      # number of ghost zones for the Poisson solver
GRA_GHOST_SIZE       = 2      # number of ghost zones for advancing fluid by gravity


# other parameters
DH                   = 1.0    # grid size
DT                   = 1.e-2  # time interval
CPULOAD              = 420000 # CPU workload for the CPU+GPU concurrent test
INIT_WIDTH           = 410.0  # width of the initial power-law density distribution
INIT_INDEX           = 2.1    # power-law index of the initial power-law density distribution
#INIT_OFFSET          = 0.0    # offset of the initial power-law density distribution for each patch
INIT_OFFSET          = 0.09   # offset of the initial power-law density distribution for each patch

ifeq "$(findstring FLOAT8, $(SIMU_OPTION))" "FLOAT8"
   ifeq "$(findstring SOR, $(SIMU_OPTION))" "SOR"
   POI_MAXERR        = 3.e-15 # maximum allowed error for the error check of potential (double)
   GRA_MAXERR        = 1.e-14 # maximum allowed error for the error check of fluid variables (double)
   else
   POI_MAXERR        = 3.e-14 # maximum allowed error for the error check of potential (double)
   GRA_MAXERR        = 3.e-14 # maximum allowed error for the error check of fluid variables (double)
   endif
else
   ifeq "$(findstring SOR, $(SIMU_OPTION))" "SOR"
   POI_MAXERR        = 5.e-6  # maximum allowed error for the error check of potential (single)
   GRA_MAXERR        = 8.e-6  # maximum allowed error for the error check of fluid variables (single)
   else
   POI_MAXERR        = 4.e-5  # maximum allowed error for the error check of potential (single)
   GRA_MAXERR        = 4.e-5  # maximum allowed error for the error check of fluid variables (single)
   endif
endif


PATCH_SIZE           := $(strip $(PATCH_SIZE))
NEWTON_G             := $(strip $(NEWTON_G))
POT_GHOST_SIZE       := $(strip $(POT_GHOST_SIZE))
GRA_GHOST_SIZE       := $(strip $(GRA_GHOST_SIZE))
DH                   := $(strip $(DH))
DT                   := $(strip $(DT))
POI_MAXERR           := $(strip $(POI_MAXERR))
GRA_MAXERR           := $(strip $(GRA_MAXERR))
CPULOAD              := $(strip $(CPULOAD))
INIT_WIDTH           := $(strip $(INIT_WIDTH))
INIT_INDEX           := $(strip $(INIT_INDEX))
INIT_OFFSET          := $(strip $(INIT_OFFSET))


SIMU_PARA = -DPATCH_SIZE=$(PATCH_SIZE) -DNEWTON_G=$(NEWTON_G) -DPOT_GHOST_SIZE=$(POT_GHOST_SIZE) -DDH=$(DH) \
            -DCPULOAD=$(CPULOAD) -DINIT_WIDTH=$(INIT_WIDTH) -DINIT_INDEX=$(INIT_INDEX) \
            -DINIT_OFFSET=$(INIT_OFFSET) -DGRA_GHOST_SIZE=$(GRA_GHOST_SIZE) -DDT=$(DT) \
            -DPOI_MAXERR=$(POI_MAXERR) -DGRA_MAXERR=$(GRA_MAXERR)



# source files
#######################################################################################################
# Cuda source files (compiled with NVCC)
CUDA_FILE  := CUAPI_Set_Diagnose_Device.cu  CUAPI_Synchronize.cu  CUAPI_Asyn_PoissonGravitySolver.cu \
              CUAPI_MemAllocate_PoissonGravity.cu  CUAPI_MemFree_PoissonGravity.cu
CUDA_FILE  += CUPOT_PoissonSolver_SOR_10to14cube.cu  CUPOT_PoissonSolver_SOR_16to18cube.cu \
	      CUPOT_GravitySolver.cu  CUPOT_PoissonSolver_MG.cu

# C/C++ source files (compiled with CXX)
CC_FILE    := Main.cpp  Aux_Message.cpp  Aux_Error.cpp  CPU_GravitySolver.cpp  CPU_PoissonGravitySolver.cpp \
	      CPU_PoissonSolver_SOR.cpp  CPU_PoissonSolver_MG.cpp


# location of all source files
# -------------------------------------------------------------------------------
vpath %.cpp   CPU_Solver Auxiliary
vpath %.cu    GPU_Solver GPU_API



# Rules and targets
#######################################################################################################
NVCC  = $(CUDA_TOOLKIT_PATH)/bin/nvcc
ifeq "$(findstring INTEL, $(SIMU_OPTION))" "INTEL"
CXX  := icpc
else
CXX  := g++
endif


# CUDA location
# -------------------------------------------------------------------------------
#CUDA_TOOLKIT_PATH := /usr/local/cuda
#CUDA_TOOLKIT_PATH := /opt/gpu/cuda/default
CUDA_TOOLKIT_PATH := /usr/local/cuda-7.5
#CUDA_TOOLKIT_PATH := /export/cuda
#CUDA_TOOLKIT_PATH := /usr/common/usg/cuda/4.0
# -------------------------------------------------------------------------------


CUDA_FILE_PATH    := .
OBJ_FILE_PATH     := ./Object

LIB := -L$(CUDA_TOOLKIT_PATH)/lib64 -lcudart
#LIB := -L$(CUDA_TOOLKIT_PATH)/lib -lcudart
ifeq "$(findstring INTEL, $(SIMU_OPTION))" "INTEL"
LIB += -limf
endif

ifeq "$(findstring OPENMP, $(SIMU_OPTION))" "OPENMP"
   ifeq "$(findstring INTEL, $(SIMU_OPTION))" "INTEL"
      OPENMP = -openmp
   else
      OPENMP = -fopenmp
   endif
endif

INCLUDE := -I./Header

COMMONFLAG := $(INCLUDE) $(SIMU_OPTION) $(SIMU_PARA)

ifeq "$(findstring INTEL, $(SIMU_OPTION))" "INTEL"
CXXWARN_FLAG := -w1
else
CXXWARN_FLAG := -Wall -Wextra -Wimplicit -Wswitch -Wformat -Wchar-subscripts -Wparentheses -Wmultichar \
                -Wtrigraphs -Wpointer-arith -Wcast-align -Wreturn-type -Wno-unused-function
endif

ifeq "$(findstring OPENMP, $(SIMU_OPTION))" ""
CXXWARN_FLAG += -Wno-unknown-pragmas
endif

ifeq "$(findstring INTEL, $(SIMU_OPTION))" "INTEL"
CXXFLAG  := $(CXXWARN_FLAG) $(COMMONFLAG) $(OPENMP) -O3 -fp-model precise
else
CXXFLAG  := $(CXXWARN_FLAG) $(COMMONFLAG) $(OPENMP) -O3
endif

ifeq "$(findstring GAMER_DEBUG, $(SIMU_OPTION))" "GAMER_DEBUG"
   ifeq "$(findstring INTEL, $(SIMU_OPTION))" "INTEL"
      CXXFLAG += -g -debug
   else
      CXXFLAG += -g
   endif
endif

NVCCFLAG_COM := -Xcompiler $(COMMONFLAG) -Xptxas -v

ifeq      "$(findstring GPU_ARCH=FERMI, $(SIMU_OPTION))" "GPU_ARCH=FERMI"
   NVCCFLAG_COM += -gencode arch=compute_20,code=\"compute_20,sm_20\"
else ifeq "$(findstring GPU_ARCH=KEPLER, $(SIMU_OPTION))" "GPU_ARCH=KEPLER"
   NVCCFLAG_COM += -gencode arch=compute_35,code=\"compute_35,sm_35\"
else ifeq "$(findstring GPU_ARCH=MAXWELL, $(SIMU_OPTION))" "GPU_ARCH=MAXWELL"
   NVCCFLAG_COM += -gencode arch=compute_52,code=\"compute_52,sm_52\"
else ifeq "$(findstring GPU_ARCH=PASCAL, $(SIMU_OPTION))" "GPU_ARCH=PASCAL"
   NVCCFLAG_COM += -gencode arch=compute_61,code=\"compute_61,sm_61\"
else
   $(error unknown GPU_ARCH !!)
endif

NVCCFLAG_POT += -Xptxas -dlcm=ca

ifeq "$(findstring GAMER_DEBUG, $(SIMU_OPTION))" "GAMER_DEBUG"
NVCCFLAG_COM += -O0 -D_DEBUG -g -G #-Xptxas -v #-deviceemu
else
NVCCFLAG_COM += -O3 # -lineinfo
endif

OBJ := $(patsubst %.cpp, $(OBJ_FILE_PATH)/%.o, $(CC_FILE))
OBJ += $(patsubst %.cu,  $(OBJ_FILE_PATH)/%.o, $(CUDA_FILE))


# implicit rules
$(OBJ_FILE_PATH)/%.o : %.cpp
	 $(CXX) $(CXXFLAG) -o $@ -c $<

$(OBJ_FILE_PATH)/CUAPI_%.o : $(CUDA_FILE_PATH)/CUAPI_%.cu
	 $(NVCC) $(NVCCFLAG_COM) -o $@ -c $<

$(OBJ_FILE_PATH)/CUPOT_%.o : $(CUDA_FILE_PATH)/CUPOT_%.cu
	 $(NVCC) $(NVCCFLAG_COM) $(NVCCFLAG_POT) -o $@ -c $<


# link all object files
$(EXECUTABLE) : $(OBJ)
	 $(CXX) -fPIC -o $@ $^ $(LIB) $(OPENMP)
	 rm ./*linkinfo -f
	 cp $(EXECUTABLE) ./Run/


clean :
	 rm -f $(OBJ)
	 rm -f $(EXECUTABLE)
	 rm ./*linkinfo -f


