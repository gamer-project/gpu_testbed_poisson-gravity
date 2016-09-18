#include "GAMER.h"

extern real (*d_Rho_Array_P    )[ RHO_NXT*RHO_NXT*RHO_NXT ];
extern real (*d_Pot_Array_P_In )[ POT_NXT*POT_NXT*POT_NXT ];
extern real (*d_Pot_Array_P_Out)[ GRA_NXT*GRA_NXT*GRA_NXT ];
extern real (*d_Flu_Array_G    )[NCOMP][ PATCH_SIZE*PATCH_SIZE*PATCH_SIZE ];

// REMOVE in the actual implementation
// #########################################################################
extern cudaStream_t *Stream;
extern int GPU_NSTREAM;
// #########################################################################




//-------------------------------------------------------------------------------------------------------
// Function    :  CUAPI_MemFree_PoissonGravity
// Description :  Free device and host memory previously allocated by the function
//                "CUAPI_MemAllocate_PoissonGravity"
//
// Parameter   :  Rho_Array_P       : Array to store the input density for the Poisson solver
//                Pot_Array_P_In    : Array to store the input coarse-grid potential for the Poisson solver
//                Pot_Array_P_Out   : Array to store the output fine-grid potential for the Poisson solver
//                Flu_Array_G       : Array to store the input and output fluid variables for the gravity solver
//-------------------------------------------------------------------------------------------------------
void CUAPI_MemFree_PoissonGravity( real (**Rho_Array_P    )[RHO_NXT][RHO_NXT][RHO_NXT],
                                   real (**Pot_Array_P_In )[POT_NXT][POT_NXT][POT_NXT],
                                   real (**Pot_Array_P_Out)[GRA_NXT][GRA_NXT][GRA_NXT],
                                   real (**Flu_Array_G    )[NCOMP][PATCH_SIZE][PATCH_SIZE][PATCH_SIZE] )
{

// free the device memory
   if ( d_Rho_Array_P     != NULL )    CUDA_CHECK_ERROR(  cudaFree( d_Rho_Array_P     )  );
   if ( d_Pot_Array_P_In  != NULL )    CUDA_CHECK_ERROR(  cudaFree( d_Pot_Array_P_In  )  );
   if ( d_Pot_Array_P_Out != NULL )    CUDA_CHECK_ERROR(  cudaFree( d_Pot_Array_P_Out )  );
   if ( d_Flu_Array_G     != NULL )    CUDA_CHECK_ERROR(  cudaFree( d_Flu_Array_G     )  );

   d_Rho_Array_P     = NULL;
   d_Pot_Array_P_In  = NULL;
   d_Pot_Array_P_Out = NULL;
   d_Flu_Array_G     = NULL;


// free the host memory allocated by CUDA
// REPLACE in the actual implementation
// #########################################################################
   for (int t=0; t<1; t++)
// for (int t=0; t<2; t++)
// #########################################################################
   {
      if ( Rho_Array_P    [t] != NULL )   CUDA_CHECK_ERROR(  cudaFreeHost( Rho_Array_P    [t] )  );
      if ( Pot_Array_P_In [t] != NULL )   CUDA_CHECK_ERROR(  cudaFreeHost( Pot_Array_P_In [t] )  );
      if ( Pot_Array_P_Out[t] != NULL )   CUDA_CHECK_ERROR(  cudaFreeHost( Pot_Array_P_Out[t] )  );
      if ( Flu_Array_G    [t] != NULL )   CUDA_CHECK_ERROR(  cudaFreeHost( Flu_Array_G    [t] )  );

      Rho_Array_P    [t] = NULL;
      Pot_Array_P_In [t] = NULL;
      Pot_Array_P_Out[t] = NULL;
      Flu_Array_G    [t] = NULL;
   }


// REMOVE in the actual implementation
// #########################################################################
// destroy streams
   for (int s=0; s<GPU_NSTREAM; s++)   cudaStreamDestroy( Stream[s] );
// #########################################################################

} // FUNCTION : CUAPI_MemFree_PoissonGravity





