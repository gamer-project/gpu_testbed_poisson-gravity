#include "GAMER.h"

#if   ( POT_SCHEME == SOR )
void CPU_PoissonSolver_SOR( const real Rho_Array    [][RHO_NXT][RHO_NXT][RHO_NXT], 
                            const real Pot_Array_In [][POT_NXT][POT_NXT][POT_NXT], 
                                  real Pot_Array_Out[][GRA_NXT][GRA_NXT][GRA_NXT], 
                            const int NPatchGroup, const real dh, const int Min_Iter, const int Max_Iter, 
                            const real Omega, const real Poi_Coeff, const IntScheme_t IntScheme );
#elif ( POT_SCHEME == MG  )
void CPU_PoissonSolver_MG( const real Rho_Array    [][RHO_NXT][RHO_NXT][RHO_NXT],
                           const real Pot_Array_In [][POT_NXT][POT_NXT][POT_NXT],
                                 real Pot_Array_Out[][GRA_NXT][GRA_NXT][GRA_NXT],
                           const int NPatchGroup, const real dh_Min, const int Max_Iter, const int NPre_Smooth,
                           const int NPost_Smooth, const real Tolerated_Error, const real Poi_Coeff, 
                           const IntScheme_t IntScheme );
#endif

void CPU_GravitySolver(       real Flu_Array[][5][PATCH_SIZE][PATCH_SIZE][PATCH_SIZE], 
                        const real Pot_Array[][GRA_NXT][GRA_NXT][GRA_NXT],
                        const int NPatchGroup, const real dt, const real dh, const bool P5_Gradient );




//-------------------------------------------------------------------------------------------------------
// Function    :  CPU_PoissonGravitySolver
// Description :  Invoke the CPU_PoissonSolver and/or CPU_GravitySolver to evaluate the potential and/or 
//                advance the fluid variables by the gravitational acceleration for a group of patches
//
// Parameter   :  Rho_Array            : Array to store the input density 
//                Pot_Array_In         : Array to store the input "coarse-grid" potential for interpolation
//                Pot_Array_Out        : Array to store the output potential
//                Flu_Array            : Array to store the fluid variables for the Gravity solver
//                NPatchGroup          : Number of patch groups evaluated simultaneously by GPU 
//                dt                   : Time interval to advance solution
//                dh                   : Grid size
//                SOR_Min_Iter         : Minimum number of iterations for SOR
//                SOR_Max_Iter         : Maximum number of iterations for SOR
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
//-------------------------------------------------------------------------------------------------------
void CPU_PoissonGravitySolver( const real Rho_Array    [][RHO_NXT][RHO_NXT][RHO_NXT], 
                                     real Pot_Array_In [][POT_NXT][POT_NXT][POT_NXT],
                                     real Pot_Array_Out[][GRA_NXT][GRA_NXT][GRA_NXT],
                                     real Flu_Array    [][5][PATCH_SIZE][PATCH_SIZE][PATCH_SIZE], 
                               const int NPatchGroup, const real dt, const real dh, const int SOR_Min_Iter, 
                               const int SOR_Max_Iter, const real SOR_Omega, const int MG_Max_Iter,
                               const int MG_NPre_Smooth, const int MG_NPost_Smooth, const real MG_Tolerated_Error,
                               const real Poi_Coeff, const IntScheme_t IntScheme, const bool P5_Gradient, 
                               const bool Poisson, const bool GraAcc )
{

   if ( Poisson )
   {
#     if   ( POT_SCHEME == SOR )

      CPU_PoissonSolver_SOR( Rho_Array, Pot_Array_In, Pot_Array_Out, NPatchGroup, dh, 
                             SOR_Min_Iter, SOR_Max_Iter, SOR_Omega, 
                             Poi_Coeff, IntScheme );

#     elif ( POT_SCHEME == MG  )

      CPU_PoissonSolver_MG ( Rho_Array, Pot_Array_In, Pot_Array_Out, NPatchGroup, dh, 
                             MG_Max_Iter, MG_NPre_Smooth, MG_NPost_Smooth, MG_Tolerated_Error, 
                             Poi_Coeff, IntScheme );

#     else

      #error : unsupported CPU Poisson solver

#     endif
   }

   if ( GraAcc )
      CPU_GravitySolver( Flu_Array, Pot_Array_Out, NPatchGroup, dt, dh, P5_Gradient );

} // FUNCTION : CPU_PoissonGravitySolver

