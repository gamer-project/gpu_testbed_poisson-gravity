#ifndef __PROTOTYPE_H__
#define __PROTOTYPE_H__



void CPU_PoissonGravitySolver( const real Rho_Array    [][RHO_NXT][RHO_NXT][RHO_NXT], 
                                     real Pot_Array_In [][POT_NXT][POT_NXT][POT_NXT],
                                     real Pot_Array_Out[][GRA_NXT][GRA_NXT][GRA_NXT],
                                     real Flu_Array    [][5][PATCH_SIZE][PATCH_SIZE][PATCH_SIZE], 
                               const int NPatchGroup, const real dt, const real dh, const int SOR_Min_Iter, 
                               const int SOR_Max_Iter, const real SOR_Omega, const int MG_Max_Iter,
                               const int MG_NPre_Smooth, const int MG_NPost_Smooth, const real MG_Tolerated_Error,
                               const real Poi_Coeff, const IntScheme_t IntScheme, const bool P5_Gradient, 
                               const bool Poisson, const bool GraAcc );
void CUAPI_Set_Diagnose_Device( int &FLU_GPU_NPGroup, int &GPU_NStream, const int InputID );
void CUAPI_MemAllocate_PoissonGravity( const int Pot_NPatchGroup, 
                                       real (**Rho_Array_P    )[RHO_NXT][RHO_NXT][RHO_NXT],
                                       real (**Pot_Array_P_In )[POT_NXT][POT_NXT][POT_NXT],
                                       real (**Pot_Array_P_Out)[GRA_NXT][GRA_NXT][GRA_NXT],
                                       real (**Flu_Array_G    )[NCOMP][PATCH_SIZE][PATCH_SIZE][PATCH_SIZE] );
void CUAPI_MemFree_PoissonGravity( real (**Rho_Array_P    )[RHO_NXT][RHO_NXT][RHO_NXT],
                                   real (**Pot_Array_P_In )[POT_NXT][POT_NXT][POT_NXT],
                                   real (**Pot_Array_P_Out)[GRA_NXT][GRA_NXT][GRA_NXT],
                                   real (**Flu_Array_G    )[NCOMP][PATCH_SIZE][PATCH_SIZE][PATCH_SIZE] );
void CUAPI_Asyn_PoissonGravitySolver( const real h_Rho_Array    [][RHO_NXT][RHO_NXT][RHO_NXT], 
                                            real h_Pot_Array_In [][POT_NXT][POT_NXT][POT_NXT],
                                            real h_Pot_Array_Out[][GRA_NXT][GRA_NXT][GRA_NXT],
                                            real h_Flu_Array    [][5][PATCH_SIZE][PATCH_SIZE][PATCH_SIZE], 
                                      const int NPatchGroup, const real dt, const real dh, const int SOR_Min_Iter,
                                      const int SOR_Max_Iter, const real SOR_Omega, const int MG_Max_Iter,
                                      const int MG_NPre_Smooth, const int MG_NPost_Smooth, 
                                      const real MG_Tolerated_Error, const real Poi_Coeff,
                                      const IntScheme_t IntScheme, const bool P5_Gradient, const bool Poisson, 
                                      const bool GraAcc, const int GPU_NStream );
void CUAPI_Synchronize();


// Auxiliary
void Aux_Message( FILE *Type, const char *Format, ... );
void Aux_Error( const char *File, const int Line, const char *Func, const char *Format, ... );


// just to make it more straightforward to implement this test program in GAMER
inline void MPI_Exit()  {  exit(-1);   }



#endif
