
                        =====================================================
                        ||                                                 ||
                        ||    GPU Poisson and Gravity solvers for GAMER    ||
                        ||                                                 ||
                        =====================================================

-----------------------------------------------------------------------------------------------------------------


Versiona 1.0         10/17/2008
===============================

1. contain both CPU and GPU solvers

2. asynchronous test

3. the version 1.0 only works for 
   a. FLU_NXT2       = 8
   b. POT_GHOST_SIZE = 2 ( 5-points stencil )

4. the entire potential data of a single patch are loaded into the shared memory before entering the SOR iteration

5. CUCAL_Poisson_Global.cu    : only use the global memory
   CUCAL_Poisson_Sharel.cu    : use three slices of shared memory (including both even and odd grids)
   CUCAL_Poisson_Share2.cu    : use three slices of shared memory (including only even or odd grids)
   CUCAL_Poisson_Share3.cu    : load all potential data into shared memory in the begining 

   performance : Share3 > Share2 > Share1 > Global

   but to fit all potential data into shared memory, the FLU_NXT2 is restricted to 8 (16 is too large)



Versiona 1.1         10/18/2008
===============================
1. optimization
   a. coalescing memory access for the global memory (both read and write)
   b. do NOT upload the ghost-zone data of the potential
   c. no bank conflict for writing data into the shared memory (still with bank conflict for reading data)

2. works for 3P_STENCIL and 5P_STENCIL

3. still only works for FLU_NXT2 = 8

4. speed-up ratio over CPU

   POT_GPU_NPATCH        : 128
   PATCH_SIZE            : 8
   FLU_NXT2              : 8
   POT_GHOST_SIZE        : 2
   POT_NXT               : 12
   
   a. POT_MAX_ITER          : 10

      CPU Processing Time   :  13.327 ms
      CPU SOR Time          :  10.321 ms
      CPU Advance Time      :   2.586 ms
      GPU Processing Time   :   2.193 ms
      GPU Invoking   Time   :   1.007 ms

      Speedup Ratio         : 6.077063
   

   b. POT_MAX_ITER          : 20

      CPU Processing Time   :  22.184 ms
      CPU SOR Time          :  19.184 ms
      CPU Advance Time      :   2.579 ms
      GPU Processing Time   :   2.491 ms
      GPU Invoking   Time   :   1.009 ms
      
      Speedup Ratio         : 8.905661


   c. POT_MAX_ITER          : 50

      CPU Processing Time   :  48.114 ms
      CPU SOR Time          :  45.119 ms
      CPU Advance Time      :   2.567 ms
      GPU Processing Time   :   3.386 ms
      GPU Invoking   Time   :   1.008 ms
      
      Speedup Ratio         : 14.209687


   c. POT_MAX_ITER          : 1000

      CPU Processing Time   : 869.129 ms
      CPU SOR Time          : 866.145 ms
      CPU Advance Time      :   2.579 ms
      GPU Processing Time   :  31.602 ms
      GPU Invoking   Time   :   1.009 ms
      
      Speedup Ratio         : 27.502342


5. somehow the performance has suddenly dropped about 2 ms (also drop for the version 1.0 )
   ~ before the performance dropped, the speed-up ratio over CPU for POT_MAX_ITER = 10 is about 8



Versiona 1.2         03/10/2009
===============================
1. construct two kernels "CUCAL_Poisson_10cube" and "CUCAL_Poisson_14cube" for POT_GHOST_SIZE = 1 and 3, 
   respectively

2. do NOT advance fluid inside the Poisson solver

3. determine the termination criteria for the SOR iteration

4. use GPU streams for the asynchronous data copy between CPU and GPU

5. performance : 

   POT_GHOST_SIZE    = 1
   G_POT_GPU_NPATCH  = 256
   
   CPU Processing Time   :  43.377 ms
   CPU SOR Time          :  42.061 ms
   CPU Advance Time      :   0.000 ms
   GPU Processing Time   :   2.696 ms
   GPU Invoking   Time   :   0.611 ms
   
   Speedup Ratio         : 16.089390


   CPU -> GPU  : 0.66 ms
   GPU -> CPU  : 0.38 ms
   Kernel      : 1.76 ms


   ===================================


   POT_GHOST_SIZE    = 3
   G_POT_GPU_NPATCH  = 256

   CPU Processing Time   : 180.158 ms
   CPU SOR Time          : 178.591 ms
   CPU Advance Time      :   0.000 ms
   GPU Processing Time   :   9.831 ms
   GPU Invoking   Time   :   1.605 ms
   
   Speedup Ratio         : 18.325500


   CPU -> GPU  : 1.51 ms
   GPU -> CPU  : 0.40 ms
   Kernel      : 7.92 ms



Versiona 1.2.1       04/19/2009
===============================
1. work with CUDA 2.1

2. add some C++ header files in order to be compiled in the NCHC GPU cluster



Versiona 1.2.2       04/25/2009
===============================
1. for POT_GHOST_SIZE == 1 ( the function "CUCAL_PoissonSolver_10cube.cu" ), store the density field in the 
   shared memory 
   --> better performance



Versiona 1.3         04/29/2009
===============================
1. add the function "CUCAL_Poisson_18cube"
   --> work with POT_NXT_IN == 18

2. since 18*18*18*4 = 22.78Kb > 16Kb, it CANNOT be stored in the shared memory
   --> we store half the data in the per-thread registers

3. the internal potential is initialized as zero in both CPU and GPU solvers



Versiona 2.0         09/14/2009
===============================
1. too many modifications ...

2. integrate into two GPU kernels
   (1) CUCAL_Poisson_10to14cube : works for POT_GHOST_SIZE = 1, 2, 3
       --> use only the shared memory to store all potential data 

   (2) CUCAL_Poisson_16to18cube : works for POT_GHOST_SIZE = 4, 5
       --> use both the shared memory and registers to store all potential data

3. power-law initial condition for the density field
   --> output the radial distribution of potential and also compare to the analytical solutions

4. perform spatial interpolation in GPU
   --> only send the coarse-grid potential into GPU

5. only send PATCH_SIZE^3 fine-grid potential back to CPU

6. re-format the Makefile

7. add the header file "Symbolic_Constant.h"

8. rename several variables to be consistent with GAMER

9. the CPU_PoissonSolver can work with arbitrary POT_GHOST_SIZE
   --> it also performs the spatial interpolation

10. count the average number of SOR iterations in the CPU Poisson solver



Versiona 2.1         09/24/2009
===============================
1. add the "conservative quadratic interpolation" in both CPU and GPU Poisson solvers 
   --> use the input parameter "IntScheme" to control the interpolation scheme



Versiona 2.2         01/13/2010
===============================
1. add the "quadratic interpolation" in both CPU and GPU Poisson solvers 
   --> set "IntScheme == 4" to use this interpolation scheme
2. for POT_GHOST_SIZE == 3, the symbolic constant POT_BLOCK_SIZE_Z is set to "2" in order to have enough registers



Versiona 2.3         02/25/2010
===============================
1. Include the CPU and GPU Gravity solvers.
2. The potential output array now stores "GRA_NXT^3" data instead of "PATCH_SIZE^3" data.



Versiona 2.3.1       03/09/2010
===============================
1. For the Gravity solver, we replace 
   "[PATCH_SIZE][PATCH_SIZE][PATCH_SIZE][NCOMP]" by "[NCOMP][PATCH_SIZE][PATCH_SIZE][PATCH_SIZE]".
   --> It is mainly for the out-of-core + GAMER computing (GAMER.1.0.beta4.0).



Versiona 2.4         07/27/2010
===============================
1. Input the POT_GPU_NPGROUP, GPU_NSTREAM, GPU_ID, INT_SCHEME, and OPT__GRA_P5_GRADIENT from the command line.
   --> Please refer to "./PoissonGravity -h" for the usage instruction.
   --> If these parameters are not set by the command-line input, they will be set to the default values 
       according to the device properties.
       -->  (a) GPU_NSTREAM = 1/4 for compute capability == 1.0 / > 1.0
            (b) FLU_GPU_NPGROUP = 2*GPU_NSTREAM*(# of multiprocessors)
            (c) GPU_ID = 0
            (d) INT_SCHEME = 4
            (e) OPT__GRA_P5_GRADIENT = true

2. Replace the "float' by "real" in all CUDA functions.



Versiona 2.5         08/31/2010
===============================
1. Revise the function "CUAPI_Set_Diagnose_Device".
2. The file "CUAPI.cu" is separated into several CUAPI_XXX files.
3. Add the header file "CUCAL.h".
4. Specify the virtual and GPU architectures in the Makefile.
5. Support the Fermi architecture.
   --> Add the option "FERMI" for the optimization in Fermi GPUs.
6. All CUCAL_XXX functions are renamed as CUPOT_XXX.
7. Declare the "volatile" qualifier for the reduction operation.



Versiona 2.6         12/12/2010
===============================
1. Replace all "!Disp" operation by "Disp^1"
   --> The "!Disp" operation does NOT work in CUDA.3.2
2. Classify all files 
   --> Add directories "CPU_Solver, GPU_Solver, GPU_API, Header"
3. Add the header "CUAPI.h"
   --> Replace all "CUDA_SAFE_CALL" and "CUT_CHECK_ERROR" by "CUDA_CHECK_ERROR"
       defined in the file "CUAPI.h"
4. Output the total memory requirement of the GPU solver
5. Rename the file "CUCAL.h" as "CUPOT.h"
6. Record the elapsed time per patch per step and the number of grids per sec
7. Record the CPU information
8. Replace "fabs" by "FABS"
9. Set the cacahe configuration in Fermi GPUs to "cudaFuncCachePreferShared"
10. Type casting all literal numbers by (real) in CPU solvers.
11. The variables "Mp, Mm" are declared as constant variables and set in the
    function "CUAPI_Set_Diagnose_Device"
12. For Fermi GPUs, we save density in the shared memory for higher
    performance
13. Support "OpenMP parallelization" in all CPU solvers.
    --> Add the option "OPENMP" in the Makefile
14. Add the option "INTEL" --> use the Intel compiler
15. The variables storing the size of memory allocation in the function
    "CUAPI_MemAllocate_PoissonGravity.cu" is declared as "long"
16. The default number of patch groups in Fermi GPUs is set to
    "2*GPU_NStream*DeviceProp.multiProcessorCount"


***BUG FIXED***
1. Fix the bug in "CUPOT_PoissonSolver_16to18cube.cu"
   --> Add "__syncthreads()" after "Residual_Total_Old = s_Residual_Total[0];"
   --> Ensure that "Residual_Total_Old" records the correct value 
       (before "s_Residual_Total[0]" is modified by the thread 0)



Versiona 2.6.1    07/16/2011
============================
1. Re-order the issue of CUDA streams for higher performance



Versiona 3.0      08/07/2011
============================
1. Support CPU/GPU multigrid solvers
   --> Add the "POT_SCHEME" option in the makefile 
   --> Only work with Fermi GPUs
   --> Higher convergence rate as compared with SOR solver, but the performance
       is lower than SOR solver for PATCH_SIZE == 8
2. Correct the reduction operation in "CUPOT_PoissonSolver_SOR_10to14cube.cu"
   and "CUPOT_PoissonSolver_SOR_16to18cube.cu"
   --> update versions from 2.4 to 2.5
3. Add the command-line option "-o" for data output
4. Performance of GPU SOR solver with double precision is highly improved



Versiona 3.0.1    05/05/2012
============================
1. Rearrange the implementation of CUDA streams to enable the overlapping of
   data transfer and kernel execution even for the GPUs not capable of the
   concurrrent upstream/downstream memory copies.
2. Replace the device property "deviceOverlap" by "asyncEngineCount" in the
   function "CUAPI_DiagnoseDevice.cu" to query both the overlapping between
   memory copy and kernel execution and the overlapping between upstream and
   downstream memory copies.



Versiona 3.1.0    03/06/2016
============================
1. Optimized for K40 GPU
   ==> Fine tune POT_BLOCK_SIZE_Z, GPU_NSTREAM
2. Support the new makefile option "GPU_ARCH=FERMI/KERPLER"
3. No longer support TESLA GPU (compute capability < 2.0)
4. Update Poisson solvers to the latest versions




***************************************************************************************** 
After v3.0, please use the new CPU/GPU solvers adopted in (or after) GAMER.1.0.beta5.2.0
--> support "IntScheme_t" data type and GRA_GHOST_SIZE == 0 in
    "CUPOT_PoissonSolver_SOR_16to18cube"
***************************************************************************************** 



BUG               08/03/2011
============================



Unfinished Works  11/13/2011
============================
1. Further optimize the GPU multigrid solver if PATCH_SIZE > 8 is required
