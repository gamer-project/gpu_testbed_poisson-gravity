#include "GAMER.h"
#include "CUPOT.h"

#if   ( POT_SCHEME == SOR )
#ifdef USE_PSOLVER_10TO14
__global__ void CUPOT_PoissonSolver_SOR_10to14cube( const real g_Rho_Array    [][ RHO_NXT*RHO_NXT*RHO_NXT ], 
                                                    const real g_Pot_Array_In [][ POT_NXT*POT_NXT*POT_NXT ], 
                                                          real g_Pot_Array_Out[][ GRA_NXT*GRA_NXT*GRA_NXT ],
                                                    const int Min_Iter, const int Max_Iter, const real Omega_6,
                                                    const real Const, const IntScheme_t IntScheme );
#else
__global__ void CUPOT_PoissonSolver_SOR_16to18cube( const real g_Rho_Array    [][ RHO_NXT*RHO_NXT*RHO_NXT ], 
                                                    const real g_Pot_Array_In [][ POT_NXT*POT_NXT*POT_NXT ], 
                                                          real g_Pot_Array_Out[][ GRA_NXT*GRA_NXT*GRA_NXT ],
                                                    const int Min_Iter, const int Max_Iter, const real Omega_6, 
                                                    const real Const, const IntScheme_t IntScheme );
#endif // #ifdef USE_PSOLVER_10TO14 ... else ...
#elif ( POT_SCHEME == MG  )
__global__ void CUPOT_PoissonSolver_MG( const real g_Rho_Array    [][ RHO_NXT*RHO_NXT*RHO_NXT ], 
                                        const real g_Pot_Array_In [][ POT_NXT*POT_NXT*POT_NXT ], 
                                              real g_Pot_Array_Out[][ GRA_NXT*GRA_NXT*GRA_NXT ],
                                        const real dh_Min, const int Max_Iter, const int NPre_Smooth,
                                        const int NPost_Smooth, const real Tolerated_Error, const real Poi_Coeff,
                                        const IntScheme_t IntScheme );
#endif // POT_SCHEME
__global__ void CUPOT_GravitySolver(       real g_Flu_Array[][5][ PATCH_SIZE*PATCH_SIZE*PATCH_SIZE ],
                                     const real g_Pot_Array[][ GRA_NXT*GRA_NXT*GRA_NXT ],
                                     const real Gra_Const, const bool P5_Gradient );


// declare all device pointers
real (*d_Rho_Array_P    )[ RHO_NXT*RHO_NXT*RHO_NXT ]                 = NULL;
real (*d_Pot_Array_P_In )[ POT_NXT*POT_NXT*POT_NXT ]                 = NULL;
real (*d_Pot_Array_P_Out)[ GRA_NXT*GRA_NXT*GRA_NXT ]                 = NULL;
real (*d_Flu_Array_G    )[NCOMP][ PATCH_SIZE*PATCH_SIZE*PATCH_SIZE ] = NULL;

// REPLACE in the actual implementation
// #########################################################################
cudaStream_t *Stream = NULL;
// extern cudaStream_t *Stream;
// #########################################################################




//-------------------------------------------------------------------------------------------------------
// Function    :  CUAPI_Asyn_PoissonGravitySolver
// Description :  Invoke the CUPOT_PoissonSolver_XXtoXXcube and/or CUPOT_GravitySolver kernel(s) to evaluate 
//                the gravitational potential and/or advance the fluid variables by the gravitational
//                acceleration for a group of patches
//
//                ***********************************************************
//                **                Asynchronous Function                  **
//                **                                                       ** 
//                **  will return before the execution in GPU is complete  **
//                ***********************************************************
//
// Note        :  a. Use streams for the asychronous memory copy between device and host
//                b. Prefix "d" : for pointers pointing to the "Device" memory space
//                   Prefix "h" : for pointers pointing to the "Host"   memory space
//
// Parameter   :  h_Rho_Array          : Host array to store the input density 
//                h_Pot_Array_In       : Host array to store the input "coarse-grid" potential for interpolation
//                h_Pot_Array_Out      : Host array to store the output potential
//                h_Flu_Array          : Host array to store the fluid variables for the Gravity solver
//                NPatchGroup          : Number of patch groups evaluated simultaneously by GPU 
//                dt                   : Time interval to advance solution
//                dh                   : Grid size
//                SOR_Min_Iter         : Minimum # of iterations for SOR
//                SOR_Max_Iter         : Maximum # of iterations for SOR
//                SOR_Omega            : Over-relaxation parameter
//                MG_Max_Iter          : Maximum number of iterations for multigrid
//                MG_NPre_Smooth       : Number of pre-smoothing steps for multigrid
//                MG_NPos_tSmooth      : Number of post-smoothing steps for multigrid
//                MG_Tolerated_Error   : Maximum tolerated error for multigrid
//                Poi_Coeff            : Coefficient in front of density in the Poisson equation (4*Pi*Newton_G*a)
//                IntScheme            : Interpolation scheme for potential : 
//                                       4 --> conservative quadratic interpolation
//                                       5 --> quadratic interpolation
//                P5_Gradient          : Use 5-points stencil to evaluate the potential gradient
//                Poisson              : true --> invoke the Poisson solver
//                GraAcc               : true --> invoke the Gravity solver
//                GPU_NStream          : Number of CUDA streams for the asynchronous memory copy
//-------------------------------------------------------------------------------------------------------
void CUAPI_Asyn_PoissonGravitySolver( const real h_Rho_Array    [][RHO_NXT][RHO_NXT][RHO_NXT], 
                                            real h_Pot_Array_In [][POT_NXT][POT_NXT][POT_NXT],
                                            real h_Pot_Array_Out[][GRA_NXT][GRA_NXT][GRA_NXT],
                                            real h_Flu_Array    [][5][PATCH_SIZE][PATCH_SIZE][PATCH_SIZE], 
                                      const int NPatchGroup, const real dt, const real dh, const int SOR_Min_Iter,
                                      const int SOR_Max_Iter, const real SOR_Omega, const int MG_Max_Iter,
                                      const int MG_NPre_Smooth, const int MG_NPost_Smooth, 
                                      const real MG_Tolerated_Error, const real Poi_Coeff,
                                      const IntScheme_t IntScheme, const bool P5_Gradient, const bool Poisson, 
                                      const bool GraAcc, const int GPU_NStream )
{

#  if   ( POT_SCHEME == SOR )
   const dim3 Poi_Block_Dim( RHO_NXT/2, RHO_NXT, POT_BLOCK_SIZE_Z );
#  elif ( POT_SCHEME == MG )
   const dim3 Poi_Block_Dim( POT_BLOCK_SIZE_X, 1, 1 );
#  endif
   const dim3 Gra_Block_Dim( PATCH_SIZE, PATCH_SIZE, GRA_BLOCK_SIZE_Z );
   const int  NPatch      = NPatchGroup*8;
   const int  Poi_NThread = Poi_Block_Dim.x * Poi_Block_Dim.y * Poi_Block_Dim.z;
#  if   ( POT_SCHEME == SOR )
   const real Poi_Const   = Poi_Coeff*dh*dh;
   const real SOR_Omega_6 = SOR_Omega/6.0;
#  endif

   real Gra_Const;

   if ( P5_Gradient )   Gra_Const = dt/(12.0*dh);
   else                 Gra_Const = dt/( 2.0*dh);


// minimum number of threads for spatial interpolation
   if ( Poi_NThread < (POT_NXT-2)*(POT_NXT-2) )
      Aux_Error( ERROR_INFO, "Poi_NThread (%d) < (POT_NXT-2)*(POT_NXT-2) (%d) !!\n", 
                 Poi_NThread, (POT_NXT-2)*(POT_NXT-2) );

// constraint due to the reduction operation in "CUPOT_Poisson_10to14cube" and "CUPOT_PoissonSolver_MG"
#  if (  ( POT_SCHEME == SOR && defined USE_PSOLVER_10TO14 )  ||  POT_SCHEME == MG  )
   if ( Poi_NThread < 64 )
      Aux_Error( ERROR_INFO, "incorrect parameter %s = %d (must >= 64) !!\n", "Poi_NThread", Poi_NThread );
#  endif

// constraint in "CUPOT_PoissonSolver_SOR_16to18cube"
#  if ( POT_SCHEME == SOR  &&  !defined USE_PSOLVER_10TO14 )
   if ( Poi_NThread != RHO_NXT*RHO_NXT/2 )
      Aux_Error( ERROR_INFO, "incorrect parameter %s = %d (must == %d) !!\n", "Poi_NThread", Poi_NThread,
                 RHO_NXT*RHO_NXT/2 );
#  endif

   if ( IntScheme != INT_CQUAD  &&  IntScheme != INT_QUAD )
      Aux_Error( ERROR_INFO, "incorrect parameter %s = %d !!\n", "IntScheme", IntScheme );

#  if ( GRA_GHOST_SIZE == 1 )
   if ( P5_Gradient )
      Aux_Error( ERROR_INFO, "incorrect parameter %s = %d !!\n", "PT_Gradient", P5_Gradient );
#  endif


   int *NPatch_per_Stream = new int [GPU_NStream];
   int *Rho_MemSize       = new int [GPU_NStream];
   int *Pot_MemSize_In    = new int [GPU_NStream];
   int *Pot_MemSize_Out   = new int [GPU_NStream];
   int *Flu_MemSize       = new int [GPU_NStream];
   int *UsedPatch         = new int [GPU_NStream];


// set the number of patches in each stream
   UsedPatch[0] = 0;

   if ( GPU_NStream == 1 )    NPatch_per_Stream[0] = NPatch;
   else
   {
      for (int s=0; s<GPU_NStream-1; s++)    
      {
         NPatch_per_Stream[s] = NPatch/GPU_NStream;
         UsedPatch[s+1] = UsedPatch[s] + NPatch_per_Stream[s];
      }

      NPatch_per_Stream[GPU_NStream-1] = NPatch - UsedPatch[GPU_NStream-1];
   }


// set the size of data to be transferred into GPU in each stream
   for (int s=0; s<GPU_NStream; s++)
   {
      Rho_MemSize    [s] = NPatch_per_Stream[s]*RHO_NXT   *RHO_NXT   *RHO_NXT   *sizeof(real);
      Pot_MemSize_In [s] = NPatch_per_Stream[s]*POT_NXT   *POT_NXT   *POT_NXT   *sizeof(real);
      Pot_MemSize_Out[s] = NPatch_per_Stream[s]*GRA_NXT   *GRA_NXT   *GRA_NXT   *sizeof(real);
      Flu_MemSize    [s] = NPatch_per_Stream[s]*PATCH_SIZE*PATCH_SIZE*PATCH_SIZE*sizeof(real)*NCOMP;
   }


// a. copy data from host to device
//=========================================================================================
   for (int s=0; s<GPU_NStream; s++)
   {
      if ( NPatch_per_Stream[s] == 0 )    continue;

      if ( Poisson )
      {
         CUDA_CHECK_ERROR(  cudaMemcpyAsync( d_Rho_Array_P     + UsedPatch[s], h_Rho_Array     + UsedPatch[s], 
                                             Rho_MemSize[s],     cudaMemcpyHostToDevice, Stream[s] )  );

         CUDA_CHECK_ERROR(  cudaMemcpyAsync( d_Pot_Array_P_In  + UsedPatch[s], h_Pot_Array_In  + UsedPatch[s],
                                             Pot_MemSize_In[s],  cudaMemcpyHostToDevice, Stream[s] )  );
      }

      if ( GraAcc )
      {
         if ( !Poisson )
         CUDA_CHECK_ERROR(  cudaMemcpyAsync( d_Pot_Array_P_Out + UsedPatch[s], h_Pot_Array_Out + UsedPatch[s],
                                             Pot_MemSize_Out[s], cudaMemcpyHostToDevice, Stream[s] )  );

         CUDA_CHECK_ERROR(  cudaMemcpyAsync( d_Flu_Array_G     + UsedPatch[s], h_Flu_Array     + UsedPatch[s], 
                                             Flu_MemSize[s],     cudaMemcpyHostToDevice, Stream[s] )  );
      }
   } // for (int s=0; s<GPU_NStream; s++)


// b. execute the kernel 
//=========================================================================================
   for (int s=0; s<GPU_NStream; s++)
   {
      if ( NPatch_per_Stream[s] == 0 )    continue;

      if ( Poisson )
      {
#        if ( POT_SCHEME == SOR )

#        ifdef USE_PSOLVER_10TO14
         CUPOT_PoissonSolver_SOR_10to14cube <<< NPatch_per_Stream[s], Poi_Block_Dim, 0, Stream[s] >>> 
                                            ( d_Rho_Array_P + UsedPatch[s], d_Pot_Array_P_In + UsedPatch[s], 
                                              d_Pot_Array_P_Out + UsedPatch[s], SOR_Min_Iter, SOR_Max_Iter, 
                                              SOR_Omega_6, Poi_Const, IntScheme );
#        else
         CUPOT_PoissonSolver_SOR_16to18cube <<< NPatch_per_Stream[s], Poi_Block_Dim, 0, Stream[s] >>> 
                                            ( d_Rho_Array_P + UsedPatch[s], d_Pot_Array_P_In + UsedPatch[s], 
                                              d_Pot_Array_P_Out + UsedPatch[s], SOR_Min_Iter, SOR_Max_Iter, 
                                              SOR_Omega_6, Poi_Const, IntScheme );
#        endif // #ifdef USE_PSOLVER_10TO14 ... else ...

#        elif ( POT_SCHEME == MG  )

         CUPOT_PoissonSolver_MG             <<< NPatch_per_Stream[s], Poi_Block_Dim, 0, Stream[s] >>> 
                                            ( d_Rho_Array_P + UsedPatch[s], d_Pot_Array_P_In + UsedPatch[s], 
                                              d_Pot_Array_P_Out + UsedPatch[s],
                                              dh, MG_Max_Iter, MG_NPre_Smooth, MG_NPost_Smooth, 
                                              MG_Tolerated_Error, Poi_Coeff, IntScheme );

#        else

         #error : unsupported GPU Poisson solver

#        endif
      } // if ( Poisson )

      if ( GraAcc )
      {
         CUPOT_GravitySolver <<< NPatch_per_Stream[s], Gra_Block_Dim, 0, Stream[s] >>> 
                             ( d_Flu_Array_G + UsedPatch[s], d_Pot_Array_P_Out + UsedPatch[s], Gra_Const, 
                               P5_Gradient );
      }

      CUDA_CHECK_ERROR( cudaGetLastError() );
   } // for (int s=0; s<GPU_NStream; s++)


// c. copy data from device to host
//=========================================================================================
   for (int s=0; s<GPU_NStream; s++)
   {
      if ( NPatch_per_Stream[s] == 0 )    continue;

      if ( Poisson )
         CUDA_CHECK_ERROR(  cudaMemcpyAsync( h_Pot_Array_Out + UsedPatch[s], d_Pot_Array_P_Out + UsedPatch[s], 
                                             Pot_MemSize_Out[s], cudaMemcpyDeviceToHost, Stream[s] )  );

      if ( GraAcc )
         CUDA_CHECK_ERROR(  cudaMemcpyAsync( h_Flu_Array     + UsedPatch[s], d_Flu_Array_G     + UsedPatch[s], 
                                             Flu_MemSize[s],     cudaMemcpyDeviceToHost, Stream[s] )  );
   } // for (int s=0; s<GPU_NStream; s++)


   delete [] NPatch_per_Stream;
   delete [] Rho_MemSize;
   delete [] Pot_MemSize_In;
   delete [] Pot_MemSize_Out;
   delete [] Flu_MemSize;
   delete [] UsedPatch;

} // FUNCTION : CUAPI_Asyn_PoissonGravitySolver



