#include "GAMER.h"
#include "CUPOT.h"
#include "Timer.h"

#ifdef FLOAT8
#  define POW  pow 
#else
#  define POW  powf 
#endif


using namespace std;


void Initialize( real Rho_Array[][RHO_NXT][RHO_NXT][RHO_NXT],
                 real Pot_Array[][POT_NXT][POT_NXT][POT_NXT],
                 real Flu_Array[][NCOMP][PATCH_SIZE][PATCH_SIZE][PATCH_SIZE] );
void Output_Radial( const int P, 
                    const real Rho_Array    [][RHO_NXT][RHO_NXT][RHO_NXT], 
                    const real Pot_Array_In [][POT_NXT][POT_NXT][POT_NXT],
                    const real Pot_Array_Out[][GRA_NXT][GRA_NXT][GRA_NXT],
                    const char Comment[20] );
void Output_Rho    ( const int P, const real Rho_Array[][RHO_NXT][RHO_NXT][RHO_NXT], const char FileName[20] );
void Output_Pot_In ( const int P, const real Pot_Array[][POT_NXT][POT_NXT][POT_NXT], const char FileName[20] );
void Output_Pot_Out( const int P, const real Pot_Array[][GRA_NXT][GRA_NXT][GRA_NXT], const char FileName[20], 
                     const bool Binary );
void Output_Flu( const int P, const real Flu_Array[][NCOMP][PATCH_SIZE][PATCH_SIZE][PATCH_SIZE], 
                 const char FileName[20], const bool Binary );
void DataCompare();
void AsynTest();
void Init_Set_Default_SOR_Parameter();
void Init_Set_Default_MG_Parameter( int &Max_Iter, int &NPre_Smooth, int &NPost_Smooth, real &Tolerated_Error );
void ReadParameters( int argc, char **argv, int &POT_GPU_NPGROUP, int &GPU_NSTREAM, int &GPU_ID, int &OMP_NTHREAD,
                     IntScheme_t &INT_SCHEME, bool &OPT__GRA_P5_GRADIENT, bool &Performance_Test, 
                     bool &Asynchronous_Test );

bool Performance_Test         = false;       // perform the performance test
bool Asynchronous_Test        = false;       // perform the test of the concurrent execution between CPU and GPU
int POT_GPU_NPGROUP           = NULL_INT;
int GPU_NSTREAM               = NULL_INT;
int GPU_ID                    = 0;
int OMP_NTHREAD               = NULL_INT;
IntScheme_t INT_SCHEME        = INT_CQUAD;   // interpolation scheme (4/5 : conservative-quadratic/quadratic)
bool OPT__GRA_P5_GRADIENT     = false;       // 5-point stencil for evaluating the potential gradient 
bool OutputData               = false;       // output initial and final data for the performance test

Timer_t *timer_CPU        = NULL;
Timer_t *timer_GPU        = NULL;
Timer_t *timer_CPU_Only   = NULL;
Timer_t *timer_GPU_Only   = NULL;
Timer_t *timer_Concurrent = NULL;

real (*CPU_Rho_Array_P    )[RHO_NXT][RHO_NXT][RHO_NXT];
real (*CPU_Pot_Array_P_In )[POT_NXT][POT_NXT][POT_NXT];
real (*CPU_Pot_Array_P_Out)[GRA_NXT][GRA_NXT][GRA_NXT];
real (*CPU_Flu_Array_G    )[NCOMP][PATCH_SIZE][PATCH_SIZE][PATCH_SIZE];

real (*GPU_Rho_Array_P    )[RHO_NXT][RHO_NXT][RHO_NXT];
real (*GPU_Pot_Array_P_In )[POT_NXT][POT_NXT][POT_NXT];
real (*GPU_Pot_Array_P_Out)[GRA_NXT][GRA_NXT][GRA_NXT];
real (*GPU_Flu_Array_G    )[NCOMP][PATCH_SIZE][PATCH_SIZE][PATCH_SIZE];

// SOR parameters
int  SOR_MAX_ITER, SOR_MIN_ITER;
real SOR_OMEGA;
#if ( POT_SCHEME == SOR )
extern double AveIter;
#endif

// multigrid parameters
int  MG_MAX_ITER=-1, MG_NPRE_SMOOTH=-1, MG_NPOST_SMOOTH=-1;
real MG_TOLERATED_ERROR=-1.0;




//-------------------------------------------------------------------------------------------------------
// Function    :  main
// Description :  
//-------------------------------------------------------------------------------------------------------
int main( int argc, char **argv )
{

// initial check
#  if ( POT_SCHEME != SOR  &&  POT_SCHEME != MG )
#     error : ERROR : unsupported Poisson scheme in the makefile !!
#  endif

#  if ( POT_GHOST_SIZE > 5 )
      #error ERROR : the current GPU Poisson solver does NOT support POT_GHOST_SIZE > 5 !!
#  endif

#  if ( GRA_GHOST_SIZE != 1  &&  GRA_GHOST_SIZE != 2 )
      #error ERROR : the current GPU Poisson and Gravity solvers only support GRA_GHOST_SIZE = 1 or 2 !!
#  endif

#  if ( POT_GHOST_SIZE < GRA_GHOST_SIZE )
      #error ERROR : POT_GHOST_SIZE < GRA_GHOST_SIZE !!
#  endif

#  if ( NCOMP != 5 )
      #error ERROR : NCOMP != 5 does NOT work !!
#  endif

#  if ( defined OPENMP  &&  !defined _OPENMP )
#     error : ERROR : something is wrong in OpenMP, the macro "_OPENMP" is not defined !!
#  endif

   if ( INIT_INDEX == 2.0 )
   {
      fprintf( stderr, "ERROR : INIT_INDEX = 2 does NOT work !!\n" );
      exit(-1);
   }


// read the command-line parameters
   ReadParameters( argc, argv, POT_GPU_NPGROUP, GPU_NSTREAM, GPU_ID, OMP_NTHREAD, INT_SCHEME, 
                   OPT__GRA_P5_GRADIENT, Performance_Test, Asynchronous_Test  );


// initialize GPU and set uninitialized parameters to the default values
   CUAPI_Set_Diagnose_Device( POT_GPU_NPGROUP, GPU_NSTREAM, GPU_ID );


// initialize timers
   timer_CPU        = new Timer_t( 1 );
   timer_GPU        = new Timer_t( 1 );
   timer_CPU_Only   = new Timer_t( 1 );
   timer_GPU_Only   = new Timer_t( 1 );
   timer_Concurrent = new Timer_t( 1 );


// allocate the CPU arrays
   CPU_Rho_Array_P     = new real [POT_GPU_NPGROUP*8][RHO_NXT][RHO_NXT][RHO_NXT];
   CPU_Pot_Array_P_In  = new real [POT_GPU_NPGROUP*8][POT_NXT][POT_NXT][POT_NXT];
   CPU_Pot_Array_P_Out = new real [POT_GPU_NPGROUP*8][GRA_NXT][GRA_NXT][GRA_NXT];
   CPU_Flu_Array_G     = new real [POT_GPU_NPGROUP*8][NCOMP][PATCH_SIZE][PATCH_SIZE][PATCH_SIZE];


// allocate the GPU arrays
   CUAPI_MemAllocate_PoissonGravity( POT_GPU_NPGROUP, &GPU_Rho_Array_P, &GPU_Pot_Array_P_In, &GPU_Pot_Array_P_Out,
                                     &GPU_Flu_Array_G );


// initialize the SOR/MG parameters
#  if   ( POT_SCHEME == SOR )
   Init_Set_Default_SOR_Parameter();
#  elif ( POT_SCHEME == MG  )
   Init_Set_Default_MG_Parameter( MG_MAX_ITER, MG_NPRE_SMOOTH, MG_NPOST_SMOOTH, MG_TOLERATED_ERROR );
#  endif


// initialize the input density and coarse-grid potential
   Initialize( CPU_Rho_Array_P, CPU_Pot_Array_P_In, CPU_Flu_Array_G );
   Initialize( GPU_Rho_Array_P, GPU_Pot_Array_P_In, GPU_Flu_Array_G );


// perform the performance test
   if ( Performance_Test )
   {
      cout << "Performance Test ... " << endl;
   
//    output the input density, coarse-grid potential, and input fluid array 
//    (here we output all data to ensure that the memory is properly allocated before the timing measurements)
      if ( OutputData )
      {
         Output_Rho( -1, CPU_Rho_Array_P, "Rho_CPU" );
         Output_Rho( -1, GPU_Rho_Array_P, "Rho_GPU" );
         Output_Pot_In( -1, CPU_Pot_Array_P_In, "PotIn_CPU" );
         Output_Pot_In( -1, GPU_Pot_Array_P_In, "PotIn_GPU" );
         Output_Flu( -1, CPU_Flu_Array_G, "FluIn_CPU", false );
         Output_Flu( -1, GPU_Flu_Array_G, "FluIn_GPU", false );
      }



//    ======================
//    [ CPU Poisson solver ]
//    ======================

timer_CPU->Start();
// =========================================================================================================
      cout << "   Invoking the CPU Poisson+Gravity solver ... " << flush;

      CPU_PoissonGravitySolver       ( CPU_Rho_Array_P, CPU_Pot_Array_P_In, CPU_Pot_Array_P_Out, CPU_Flu_Array_G,
                                       POT_GPU_NPGROUP, DT, DH, SOR_MIN_ITER, SOR_MAX_ITER, SOR_OMEGA,
                                       MG_MAX_ITER, MG_NPRE_SMOOTH, MG_NPOST_SMOOTH, MG_TOLERATED_ERROR, 
                                       4.0*M_PI*NEWTON_G, INT_SCHEME, OPT__GRA_P5_GRADIENT, true, true );
      cout << "done" << endl;
// =========================================================================================================
timer_CPU->Stop( false );



//    ======================
//    [ GPU Poisson solver ]
//    ======================

timer_GPU->Start();
//=========================================================================================================
      cout << "   Invoking the GPU Poisson+Gravity solver ... " << flush;

      CUAPI_Asyn_PoissonGravitySolver( GPU_Rho_Array_P, GPU_Pot_Array_P_In, GPU_Pot_Array_P_Out, GPU_Flu_Array_G,
                                       POT_GPU_NPGROUP, DT, DH, SOR_MIN_ITER, SOR_MAX_ITER, SOR_OMEGA, 
                                       MG_MAX_ITER, MG_NPRE_SMOOTH, MG_NPOST_SMOOTH, MG_TOLERATED_ERROR, 
                                       4.0*M_PI*NEWTON_G, INT_SCHEME, OPT__GRA_P5_GRADIENT, true, true, 
                                       GPU_NSTREAM );
      CUAPI_Synchronize();

      cout << "done" << endl;
//=========================================================================================================
timer_GPU->Stop( false );



      if ( OutputData )
      {
//       output the evaluated fine-grid potential
         Output_Pot_Out( 0, CPU_Pot_Array_P_Out, "PotOut_CPU", false );
         Output_Pot_Out( 0, GPU_Pot_Array_P_Out, "PotOut_GPU", false );

//       output the advanced fluid data
         Output_Flu( 0, CPU_Flu_Array_G, "FluOut_CPU", false );
         Output_Flu( 0, GPU_Flu_Array_G, "FluOut_GPU", false );

//       output the evaluated fine-grid potential as a function of radius
         Output_Radial( 0, CPU_Rho_Array_P, CPU_Pot_Array_P_In, CPU_Pot_Array_P_Out, "CPU" );
         Output_Radial( 0, GPU_Rho_Array_P, GPU_Pot_Array_P_In, GPU_Pot_Array_P_Out, "GPU" );
      }


//    compare the CPU and GPU results
      DataCompare();

      cout << "Performance Test ... done" << endl;
   } // if ( Performance_Test )



// perform the CPU/GPU concurrent execution test
// =========================================================================================================
   if ( Asynchronous_Test )   AsynTest();
// =========================================================================================================



// output the note file
   const int NPatch = POT_GPU_NPGROUP*8;
   FILE *Note = fopen( "Note", "a" );

   fprintf( Note, "\n" );

   fprintf( Note, "PATCH_SIZE            : %d\n"         , PATCH_SIZE );
   fprintf( Note, "NEWTON_G              : %13.7e\n"     , NEWTON_G );
   fprintf( Note, "POT_GPU_NPGROUP       : %d\n"         , POT_GPU_NPGROUP );
   fprintf( Note, "POT_GHOST_SIZE        : %d\n"         , POT_GHOST_SIZE );
   fprintf( Note, "GRA_GHOST_SIZE        : %d\n"         , GRA_GHOST_SIZE );
   fprintf( Note, "OPT__GRA_P5_GRADIENT  : %d\n"         , OPT__GRA_P5_GRADIENT );
   fprintf( Note, "GPU_NSTREAM           : %d\n"         , GPU_NSTREAM );
   fprintf( Note, "OMP_NTHREAD           : %d\n"         , OMP_NTHREAD );
   fprintf( Note, "INT_SCHEME            : %d\n"         , INT_SCHEME );
   fprintf( Note, "POT_NXT               : %d\n"         , POT_NXT );
   fprintf( Note, "RHO_NXT               : %d\n"         , RHO_NXT );
   fprintf( Note, "GRA_NXT               : %d\n"         , GRA_NXT );
#  if   ( POT_SCHEME == SOR )
   fprintf( Note, "POT_BLOCK_SIZE_Z      : %d\n"         , POT_BLOCK_SIZE_Z );
   fprintf( Note, "SOR_MAX_ITER          : %d\n"         , SOR_MAX_ITER );
   fprintf( Note, "SOR_MIN_ITER          : %d\n"         , SOR_MIN_ITER );
   fprintf( Note, "SOR_OMEGA             : %13.7e\n"     , SOR_OMEGA );
#  elif ( POT_SCHEME == MG  )
   fprintf( Note, "POT_BLOCK_SIZE_X      : %d\n"         , POT_BLOCK_SIZE_X );
   fprintf( Note, "MG_MAX_ITER           : %d\n"         , MG_MAX_ITER );
   fprintf( Note, "MG_NPRE_SMOOTH        : %d\n"         , MG_NPRE_SMOOTH );
   fprintf( Note, "MG_NPOST_SMOOTH       : %d\n"         , MG_NPOST_SMOOTH );
   fprintf( Note, "MG_TOLERATED_ERROR    : %13.7e\n"     , MG_TOLERATED_ERROR );
#  endif
   fprintf( Note, "DH                    : %4.2f\n"      , DH );
   fprintf( Note, "DT                    : %4.2f\n"      , DT );
   fprintf( Note, "POI_MAXERR            : %13.7e\n"     , POI_MAXERR );
   fprintf( Note, "GRA_MAXERR            : %13.7e\n"     , GRA_MAXERR );
   fprintf( Note, "CPULOAD               : %d\n"         , CPULOAD );
   fprintf( Note, "INIT_WIDTH            : %13.7e\n"     , INIT_WIDTH );
   fprintf( Note, "INIT_INDEX            : %13.7e\n"     , INIT_INDEX );
   fprintf( Note, "INIT_OFFSET           : %13.7e\n"     , INIT_OFFSET );

   if ( Performance_Test )
   fprintf( Note, "Performance_Test      : ON\n" );
   else
   fprintf( Note, "Performance_Test      : OFF\n" );

   if ( Asynchronous_Test )
   fprintf( Note, "Asynchronous_Test     : ON\n" );
   else
   fprintf( Note, "Asynchronous_Test     : OFF\n" );

#  if   ( POT_SCHEME == SOR )
   fprintf( Note, "POT_SCHEME            : SOR\n" );
#  elif ( POT_SCHEME == MG  )
   fprintf( Note, "POT_SCHEME            : MG\n" );
#  else
   fprintf( Note, "POT_SCHEME            : UNKNOWN\n" );
#  endif

#  ifdef INTEL
   fprintf( Note, "Compiler              : Intel\n" );
#  else
   fprintf( Note, "Compiler              : GNU\n" );
#  endif

#  ifdef FLOAT8
   fprintf( Note, "FLOAT8                : ON\n" );
#  else
   fprintf( Note, "FLOAT8                : OFF\n" );
#  endif

#  if   ( GPU_ARCH == FERMI )
   fprintf( Note, "GPU_ARCH              : FERMI\n" );
#  elif ( GPU_ARCH == KEPLER )
   fprintf( Note, "GPU_ARCH              : KEPLER\n" );
#  else
   fprintf( Note, "GPU_ARCH              : UNKNOWN\n" );
#  endif

#  ifdef GAMER_DEBUG
   fprintf( Note, "GAMER_DEBUG           : ON\n" );
#  else
   fprintf( Note, "GAMER_DEBUG           : OFF\n" );
#  endif

#  ifdef OPENMP
   fprintf( Note, "OPENMP                : ON\n" );
#  else
   fprintf( Note, "OPENMP                : OFF\n" );
#  endif

#  ifdef USE_PSOLVER_10TO14
   fprintf( Note, "USE_PSOLVER_10TO14    : ON\n" );
#  else
   fprintf( Note, "USE_PSOLVER_10TO14    : OFF\n" );
#  endif

   fprintf( Note, "\n\n" );


   if ( Performance_Test )
   {
      fprintf( Note, "Performance Test :\n" );
      fprintf( Note, "------------------------------------------------------------------------------\n" );
#     if ( POT_SCHEME == SOR )
      fprintf( Note, "Average Iteration     : %9.3f\n", AveIter/NPatch );
#     endif
      fprintf( Note, "CPU Processing Time   : %9.3f ms ( %8.4f ms/patch/step, %13.7e grids/sec )\n", 
               timer_CPU->GetValue(0)*1.e3, timer_CPU->GetValue(0)/NPatch*1.e3, 
               NPatch*PS1*PS1*PS1/timer_CPU->GetValue(0) );
      fprintf( Note, "GPU Processing Time   : %9.3f ms ( %8.4f ms/patch/step, %13.7e grids/sec )\n",
               timer_GPU->GetValue(0)*1.e3, timer_GPU->GetValue(0)/NPatch*1.e3, 
               NPatch*PS1*PS1*PS1/timer_GPU->GetValue(0) );
      fprintf( Note, "Speedup Ratio         : %9.3f\n", timer_CPU->GetValue(0)/timer_GPU->GetValue(0) );
      fprintf( Note, "\n\n");
   }


   if ( Asynchronous_Test )
   {
      fprintf( Note, "CUDA Asynchronous Test :\n" );
      fprintf( Note, "------------------------------------------------------------------------------\n" );
      fprintf( Note, "CPU Array Size        : %8d KB\n"    , CPULOAD ); 
      fprintf( Note, "CPU Only  Time        : %8.3f ms\n"  , 1.e3*timer_CPU_Only->GetValue(0) );
      fprintf( Note, "GPU Only  Time        : %8.3f ms\n"  , 1.e3*timer_GPU_Only->GetValue(0) );
      fprintf( Note, "CPU + GPU Time        : %8.3f ms\n"  , 1.e3*timer_Concurrent->GetValue(0) );
      fprintf( Note, "Overlap   Time        : %8.3f ms\n\n", 1.e3*( timer_CPU_Only->GetValue(0) +
                                                                    timer_GPU_Only->GetValue(0) -
                                                                    timer_Concurrent->GetValue(0) ) );
      fprintf( Note, "\n\n");
   }

   fclose( Note );
   

// free memory
   CUAPI_MemFree_PoissonGravity( &GPU_Rho_Array_P, &GPU_Pot_Array_P_In, &GPU_Pot_Array_P_Out, &GPU_Flu_Array_G );

   delete timer_CPU;
   delete timer_GPU;
   delete timer_CPU_Only;
   delete timer_GPU_Only;
   delete timer_Concurrent;

   delete [] CPU_Rho_Array_P;
   delete [] CPU_Pot_Array_P_In;
   delete [] CPU_Pot_Array_P_Out;
   delete [] CPU_Flu_Array_G;

   exit(0);

}



//-------------------------------------------------------------------------------------------------------
// Function    :  ReadParameters
// Description :  Read the command-line parameters
//-------------------------------------------------------------------------------------------------------
void ReadParameters( int argc, char **argv, int &POT_GPU_NPGROUP, int &GPU_NSTREAM, int &GPU_ID, int &OMP_NTHREAD,
                     IntScheme_t &INT_SCHEME, bool &OPT__GRA_P5_GRADIENT, bool &Performance_Test, 
                     bool &Asynchronous_Test )
{

   int c;
   while ( (c = getopt(argc, argv, "hoPSn:s:g:i:p:t:m:b:a:e:")) != -1 )
   {
      switch ( c )
      {
         case 'n' : POT_GPU_NPGROUP      = atoi(optarg);                break;
         case 's' : GPU_NSTREAM          = atoi(optarg);                break;
         case 'g' : GPU_ID               = atoi(optarg);                break;
         case 'P' : Performance_Test     = true;                        break;
         case 'S' : Asynchronous_Test    = true;                        break;
         case 'i' : INT_SCHEME           = (IntScheme_t) atoi(optarg);  break;
         case 'p' : OPT__GRA_P5_GRADIENT = atoi(optarg);                break;
         case 't' : OMP_NTHREAD          = atoi(optarg);                break;
         case 'm' : MG_MAX_ITER          = atoi(optarg);                break;
         case 'b' : MG_NPRE_SMOOTH       = atoi(optarg);                break;
         case 'a' : MG_NPOST_SMOOTH      = atoi(optarg);                break;
         case 'e' : MG_TOLERATED_ERROR   = atof(optarg);                break;
         case 'o' : OutputData           = true;                        break;
         case 'h' :
         case '?' : cerr << endl << "usage: " << argv[0] 
                    << " [-h (for help)] [-n # of patch groups] [-s # of CUDA streams] [-g GPU ID [0]]" 
                    << endl << "                       " 
                    << " [-t # of OpenMP threads (<=0 -> default value) [default]]" 
                    << endl << "                       " 
                    << " [-P (performance test) [off]] [-S (asynchronous test) [off]]" 
                    << endl << "                       " 
                    << " [-i (4/5)=(c-quad/quad) interpolation [4]]" 
                    << endl << "                       " 
                    << " [-p (0/1)=(3/5)-point stencil for potential gradient [0]]"
                    << endl << "                       " 
                    << " [-m maximum number of iterations for multigrid (<0 -> default value) [default]]" 
                    << endl << "                       " 
                    << " [-b number of pre-smoothing steps for multigrid (<0 -> default value) [default]]" 
                    << endl << "                       " 
                    << " [-a number of post-smoothing steps for multigrid (<0 -> default value) [default]]" 
                    << endl << "                       " 
                    << " [-e maximum tolerated error for multigrid (<0 -> default value) [default]]" 
                    << endl << "                       " 
                    << " [-o (output data) [off]]"
                    << endl;
                    exit( -1 );
      }
   }


// set the number of OpenMP threads
#  ifdef OPENMP
   const int OMP_Max_NThread = omp_get_max_threads();

   if ( OMP_NTHREAD == NULL_INT  ||  OMP_NTHREAD <= 0 )  OMP_NTHREAD = OMP_Max_NThread;
   else if ( OMP_NTHREAD > OMP_Max_NThread )
      fprintf( stderr, "WARNING : OMP_NTHREAD (%d) > omp_get_max_threads (%d) !!\n", 
               OMP_NTHREAD, OMP_Max_NThread );

   omp_set_num_threads( OMP_NTHREAD );
#  else 
   if ( OMP_NTHREAD != NULL_INT )
      fprintf( stderr, "WARNING: the option \"-t\" has no effect if \"OPENMP\" is not turned on !!\n" );
#  endif


// check
   if ( GPU_ID < 0 )
   {
      fprintf( stderr, "ERROR : incorrect GPU ID (%d) !!\n", GPU_ID );
      exit(-1);
   }

   if ( OPT__GRA_P5_GRADIENT  &&  GRA_GHOST_SIZE == 1 )
   {
      fprintf( stderr, "ERROR : please set GRA_GHOST_SIZE = 2 for using the option OPT__GRA_P5_GRADIENT !!\n" );
      exit( -1 );
   }

   if (  !Performance_Test  &&  !Asynchronous_Test  )
   {
      fprintf( stderr, "ERROR : please enable at least one test (-P/-S) !!\n" );
      exit(-1);
   }

}



//-------------------------------------------------------------------------------------------------------
// Function    :  Initialize
// Description :  initialize the density, coarse-grid potential, and fluid variables
//
// Note        :  a. default : power-law initial condition
//                b. each patch is slightly offset in the z direction
//-------------------------------------------------------------------------------------------------------
void Initialize( real Rho_Array[][RHO_NXT][RHO_NXT][RHO_NXT],
                 real Pot_Array[][POT_NXT][POT_NXT][POT_NXT],
                 real Flu_Array[][NCOMP][PATCH_SIZE][PATCH_SIZE][PATCH_SIZE] )
{

   const int   NPatch = POT_GPU_NPGROUP*8;
   const real _Width  = 1.0 / INIT_WIDTH;

   real x, y, z, Radius, Offset;


// initialize the fine-grid density
   const real C_Rho[3] = { 0.5*RHO_NXT*DH, 0.5*RHO_NXT*DH, 0.5*RHO_NXT*DH };

   for (int P=0; P<NPatch; P++)     {  Offset = INIT_OFFSET*P*DH;
   for (int k=0; k<RHO_NXT; k++)    {  z      = (k+0.5)*DH + Offset;
   for (int j=0; j<RHO_NXT; j++)    {  y      = (j+0.5)*DH;
   for (int i=0; i<RHO_NXT; i++)    {  x      = (i+0.5)*DH;

      Radius = sqrtf( (x-C_Rho[0])*(x-C_Rho[0]) + (y-C_Rho[1])*(y-C_Rho[1]) + (z-C_Rho[2])*(z-C_Rho[2]) );

      Rho_Array[P][k][j][i] = POW( Radius*_Width, -INIT_INDEX );

   }}}}


// initialize the coarse-grid potential
   const real DH2      = DH*2.0;
   const real A        = 4.0*M_PI*NEWTON_G*POW(INIT_WIDTH,INIT_INDEX)/(INIT_INDEX-3.0)/(INIT_INDEX-2.0);
   const real C_Pot[3] = { 0.5*POT_NXT*DH2, 0.5*POT_NXT*DH2, 0.5*POT_NXT*DH2 };

   for (int P=0; P<NPatch; P++)     {  Offset = INIT_OFFSET*P*DH;
   for (int k=0; k<POT_NXT; k++)    {  z      = (k+0.5)*DH2 + Offset;
   for (int j=0; j<POT_NXT; j++)    {  y      = (j+0.5)*DH2;
   for (int i=0; i<POT_NXT; i++)    {  x      = (i+0.5)*DH2;

      Radius = sqrtf( (x-C_Pot[0])*(x-C_Pot[0]) + (y-C_Pot[1])*(y-C_Pot[1]) + (z-C_Pot[2])*(z-C_Pot[2]) );

      Pot_Array[P][k][j][i] = A*POW( Radius, -INIT_INDEX+2.0 );

   }}}}


// initialize the input fluid data
   const real Width = 33.3;

   for (int P=0; P<NPatch; P++)     {  Offset = INIT_OFFSET*P*DH; 
   for (int v=0; v<NCOMP; v++)      {
   for (int k=0; k<PATCH_SIZE; k++) {  z      = (k+0.5)*DH + Offset;
   for (int j=0; j<PATCH_SIZE; j++) {  y      = (j+0.5)*DH;
   for (int i=0; i<PATCH_SIZE; i++) {  x      = (i+0.5)*DH;

      Flu_Array[P][v][k][j][i] = 2.0 + sin( 2.0*M_PI/Width*(x+y+z+v) );

   }}}}}

}



//-------------------------------------------------------------------------------------------------------
// Function    :  Output_Radial
// Description :  output the radial distribution of input variables 
//
// Note        :  it also compare the simulation results with the analytical solutions 
//
// Parameter   :  P >= 0 --> output the patch "P"
//                  <  0 --> output all patches
//-------------------------------------------------------------------------------------------------------
void Output_Radial( const int P, 
                    const real Rho_Array    [][RHO_NXT][RHO_NXT][RHO_NXT], 
                    const real Pot_Array_In [][POT_NXT][POT_NXT][POT_NXT],
                    const real Pot_Array_Out[][GRA_NXT][GRA_NXT][GRA_NXT],
                    const char Comment[20] )
{

   const int NPatch = POT_GPU_NPGROUP*8;

   if ( P >= NPatch )
   {
      fprintf( stderr, "Error : the input patch ID exceeds the total number of patches !!\n" );
      exit( -1 );
   }


// set the output range
   int StartP, EndP;

   if ( P < 0 )
   {
      StartP = 0;
      EndP   = NPatch-1; 
   }

   else
   {
      StartP = P;
      EndP   = P;
   }

   real x, y, z, Radius, Offset;


// a. output the input density
// ------------------------------------------------------------------------------------
   const real C_Rho[3] = { 0.5*RHO_NXT*DH, 0.5*RHO_NXT*DH, 0.5*RHO_NXT*DH };

   char FileName1[100];
   sprintf( FileName1, "Radial_Rho_%s", Comment );
   FILE *File1 = fopen( FileName1, "w" );

   fprintf( File1, "%14s\t\t%14s\n", "Radius", "Density" );

   for (int PID=StartP; PID<=EndP; PID++)    {  Offset = INIT_OFFSET*PID*DH;
   for (int k=0; k<RHO_NXT; k++)             {  z      = (k+0.5)*DH + Offset;
   for (int j=0; j<RHO_NXT; j++)             {  y      = (j+0.5)*DH;
   for (int i=0; i<RHO_NXT; i++)             {  x      = (i+0.5)*DH;

      Radius = sqrtf( (x-C_Rho[0])*(x-C_Rho[0]) + (y-C_Rho[1])*(y-C_Rho[1]) + (z-C_Rho[2])*(z-C_Rho[2]) );

      fprintf( File1, "%14.7e\t\t%14.7e\n", Radius, Rho_Array[PID][k][j][i] );

   }}}}

   fclose( File1 );



// b. output the input coarse-grid potential
// ------------------------------------------------------------------------------------
   const real DH2        = DH*2.0;
   const real C_PotIn[3] = { 0.5*POT_NXT*DH2, 0.5*POT_NXT*DH2, 0.5*POT_NXT*DH2 };

   char FileName2[100];
   sprintf( FileName2, "Radial_PotIn_%s", Comment );
   FILE *File2 = fopen( FileName2, "w" );

   fprintf( File2, "%14s\t\t%14s\n", "Radius", "Potential" );

   for (int PID=StartP; PID<=EndP; PID++)    {  Offset = INIT_OFFSET*PID*DH;
   for (int k=0; k<POT_NXT; k++)             {  z      = (k+0.5)*DH2 + Offset;
   for (int j=0; j<POT_NXT; j++)             {  y      = (j+0.5)*DH2;
   for (int i=0; i<POT_NXT; i++)             {  x      = (i+0.5)*DH2;

      Radius = sqrtf( (x-C_PotIn[0])*(x-C_PotIn[0]) + (y-C_PotIn[1])*(y-C_PotIn[1]) + 
                      (z-C_PotIn[2])*(z-C_PotIn[2]) );

      fprintf( File2, "%14.7e\t\t%14.7e\n", Radius, Pot_Array_In[PID][k][j][i] );

   }}}}

   fclose( File2 );



// c. output the output fine-grid potential
// ------------------------------------------------------------------------------------
   const real C_PotOut[3] = { 0.5*GRA_NXT*DH, 0.5*GRA_NXT*DH, 0.5*GRA_NXT*DH };
   const real A           = 4.0*M_PI*NEWTON_G*POW(INIT_WIDTH,INIT_INDEX)/(INIT_INDEX-3.0)/(INIT_INDEX-2.0);

   real AnalPot, SolPot, RelErr;;

   char FileName3[100];
   sprintf( FileName3, "Radial_PotOut_%s", Comment );
   FILE *File3 = fopen( FileName3, "w" );

   fprintf( File3, "%14s\t\t%14s\t\t%14s\t\t%14s\n", "Radius", "Potential", "Analytical", "RelErr" );

   for (int PID=StartP; PID<=EndP; PID++)                         {  Offset = INIT_OFFSET*PID*DH;
   for (int k=GRA_GHOST_SIZE; k<GRA_GHOST_SIZE+PATCH_SIZE; k++)   {  z      = (k+0.5)*DH + Offset;
   for (int j=GRA_GHOST_SIZE; j<GRA_GHOST_SIZE+PATCH_SIZE; j++)   {  y      = (j+0.5)*DH;
   for (int i=GRA_GHOST_SIZE; i<GRA_GHOST_SIZE+PATCH_SIZE; i++)   {  x      = (i+0.5)*DH;

      Radius = sqrtf( (x-C_PotOut[0])*(x-C_PotOut[0]) + (y-C_PotOut[1])*(y-C_PotOut[1]) + 
                      (z-C_PotOut[2])*(z-C_PotOut[2]) );

      SolPot  = Pot_Array_Out[PID][k][j][i];
      AnalPot = A*POW( Radius, -INIT_INDEX+2.0 );
      RelErr  = (SolPot-AnalPot) / AnalPot;

      fprintf( File3, "%14.7e\t\t%14.7e\t\t%14.7e\t\t%14.7e\n", Radius, SolPot, AnalPot, RelErr );

   }}}}

   fclose( File3 );

}



//-------------------------------------------------------------------------------------------------------
// Function    :  Output_Flu
// Description :  output the data of the input fluid array
//
// Parameter   :  P >= 0 --> output the patch "P"
//                  <  0 --> output all patches
//             :  Binary = (true / false) --> output the (binary / text) data
//-------------------------------------------------------------------------------------------------------
void Output_Flu( const int P, const real Flu_Array[][NCOMP][PATCH_SIZE][PATCH_SIZE][PATCH_SIZE], 
                 const char FileName[20], const bool Binary )
{

   const int NPatch = POT_GPU_NPGROUP*8;

   if ( P >= NPatch )
   {
      fprintf( stderr, "Error : the input patch ID exceeds the total number of patches !!\n" );
      exit( -1 );
   }


// set the output range
   int StartP, EndP;

   if ( P < 0 )
   {
      StartP = 0;
      EndP   = NPatch-1; 
   }

   else
   {
      StartP = P;
      EndP   = P;
   }


// output
   if ( Binary )
   {
      FILE *File = fopen( FileName, "wb" );

      for (int PID=StartP; PID<=EndP; PID++)
         fwrite( &Flu_Array[PID][0][0][0][0], sizeof(real), PATCH_SIZE*PATCH_SIZE*PATCH_SIZE*NCOMP, File );

      fclose( File );
   }

   else
   {
      FILE *File = fopen( FileName, "w" );

      fprintf( File, "%4s %2s %2s %2s  %14s  %14s  %14s  %14s  %14s\n", 
               "PID", "i", "j", "k", "Rho", "Px", "Py", "Pz", "Egy" ); 

      for (int PID=StartP; PID<=EndP; PID++)
      for (int k=0; k<PATCH_SIZE; k++)
      for (int j=0; j<PATCH_SIZE; j++)
      for (int i=0; i<PATCH_SIZE; i++)
      {
         fprintf( File, "%4d %2d %2d %2d  %14.7e  %14.7e  %14.7e  %14.7e  %14.7e\n", 
                  PID, i, j, k, 
                  Flu_Array[PID][0][k][j][i], Flu_Array[PID][1][k][j][i], Flu_Array[PID][2][k][j][i], 
                  Flu_Array[PID][3][k][j][i], Flu_Array[PID][4][k][j][i] );
      }

      fclose( File );
   }

}



//-------------------------------------------------------------------------------------------------------
// Function    :  Output_Rho
// Description :  output the data of the input density array
//
// Parameter   :  P >= 0 --> output the patch "P"
//                  <  0 --> output all patches
//-------------------------------------------------------------------------------------------------------
void Output_Rho( const int P, const real Rho_Array[][RHO_NXT][RHO_NXT][RHO_NXT], const char FileName[20] )
{

   const int NPatch = POT_GPU_NPGROUP*8;

   if ( P >= NPatch )
   {
      fprintf( stderr, "Error : the input patch ID exceeds the total number of patches !!\n" );
      exit( -1 );
   }


// set the output range
   int StartP, EndP;

   if ( P < 0 )
   {
      StartP = 0;
      EndP   = NPatch-1; 
   }

   else
   {
      StartP = P;
      EndP   = P;
   }


// output
   FILE *File = fopen( FileName, "w" );

   for (int PID=StartP; PID<=EndP; PID++)
   {
      fprintf( File, "\nPatch %d\n\n", PID );

      for (int k=0; k<RHO_NXT; k++)
      {
         fprintf( File, "k = %d\n\n", k );

         for (int j=RHO_NXT-1; j>=0; j--)
         {
            for (int i=0; i<RHO_NXT; i++)
            {
               fprintf( File, "%10.5e ", Rho_Array[PID][k][j][i] );
            }

            fprintf( File, "\n" );
         }

         fprintf( File, "\n\n" );
      }
   }

   fclose( File );

}



//-------------------------------------------------------------------------------------------------------
// Function    :  Output_Pot_In
// Description :  output the data of the input potential array
//
// Parameter   :  P >= 0 --> output the patch "P"
//                  <  0 --> output all patches
//-------------------------------------------------------------------------------------------------------
void Output_Pot_In( const int P, const real Pot_Array[][POT_NXT][POT_NXT][POT_NXT], const char FileName[20] )
{

   const int NPatch = POT_GPU_NPGROUP*8;

   if ( P >= NPatch )
   {
      fprintf( stderr, "Error : the input patch ID exceeds the total number of patches !!\n" );
      exit( -1 );
   }


// set the output range
   int StartP, EndP;

   if ( P < 0 )
   {
      StartP = 0;
      EndP   = NPatch-1; 
   }

   else
   {
      StartP = P;
      EndP   = P;
   }


// output
   FILE *File = fopen( FileName, "w" );

   for (int PID=StartP; PID<=EndP; PID++)
   {
      fprintf( File, "\nPatch %d\n\n", PID );

      for (int k=0; k<POT_NXT  ; k++)
      {
         fprintf( File, "k = %d\n\n", k );

         for (int j=POT_NXT-1; j>=0; j--)
         {
            for (int i=0; i<POT_NXT  ; i++)
            {
               fprintf( File, "%10.5e ", Pot_Array[PID][k][j][i] );
            }

            fprintf( File, "\n" );
         }

         fprintf( File, "\n\n" );
      }
   }

   fclose( File );

}



//-------------------------------------------------------------------------------------------------------
// Function    :  Output_Pot_Out
// Description :  output the data of the output potential array
//
// Parameter   :  P >= 0 --> output the patch "P"
//                  <  0 --> output all patches
//             :  Binary = (true / false) --> output the (binary / text) data
//-------------------------------------------------------------------------------------------------------
void Output_Pot_Out( const int P, const real Pot_Array[][GRA_NXT][GRA_NXT][GRA_NXT], const char FileName[20], 
                     const bool Binary )
{

   const int NPatch = POT_GPU_NPGROUP*8;

   if ( P >= NPatch )
   {
      fprintf( stderr, "Error : the input patch ID exceeds the total number of patches !!\n" );
      exit( -1 );
   }


// set the output range
   int StartP, EndP;

   if ( P < 0 )
   {
      StartP = 0;
      EndP   = NPatch-1; 
   }

   else
   {
      StartP = P;
      EndP   = P;
   }


// output
   if ( Binary )
   {
      FILE *File = fopen( FileName, "wb" );

      for (int PID=StartP; PID<=EndP; PID++)
      for (int k=0; k<GRA_NXT; k++)
      for (int j=0; j<GRA_NXT; j++)
      for (int i=0; i<GRA_NXT; i++)
         fwrite( &Pot_Array[PID][k][j][i], sizeof(real), 1, File );

      fclose( File );
   }

   else
   {
      FILE *File = fopen( FileName, "w" );

      for (int PID=StartP; PID<=EndP; PID++)
      {
         fprintf( File, "\nPatch %d\n\n", PID );

         for (int k=0; k<GRA_NXT; k++)
         {
            fprintf( File, "k = %d\n\n", k-GRA_GHOST_SIZE );

            for (int j=GRA_NXT-1; j>=0; j--)
            {
               for (int i=0; i<GRA_NXT; i++)
               {
                  fprintf( File, "%13.6e ", Pot_Array[PID][k][j][i] );
               }

               fprintf( File, "\n" );
            }

            fprintf( File, "\n\n" );
         }
      } 

      fclose( File );
   }

}



//-------------------------------------------------------------------------------------------------------
// Function    :  DataCompare
// Description :  compare the CPU and GPU results 
//-------------------------------------------------------------------------------------------------------
void DataCompare()
{

   const int NPatch = POT_GPU_NPGROUP*8;
   bool Pass = true;
   real Err;


// compare the potential data
   for (int P=0; P<NPatch; P++)
   for (int k=0; k<GRA_NXT; k++)
   for (int j=0; j<GRA_NXT; j++)
   for (int i=0; i<GRA_NXT; i++)
   {
      Err = fabsf(  ( CPU_Pot_Array_P_Out[P][k][j][i] - GPU_Pot_Array_P_Out[P][k][j][i] )
                    / CPU_Pot_Array_P_Out[P][k][j][i]  );

      if ( Err > POI_MAXERR ) 
      {
         fprintf( stderr, "Patch %3d, (i,j,k) = (%d, %d, %d), CPU = %14.7e, GPU = %14.7e, Err = %14.7e\n",
                  P, i, j, k, CPU_Pot_Array_P_Out[P][k][j][i], GPU_Pot_Array_P_Out[P][k][j][i], Err );  

         Pass = false;
      }
   }

   if ( Pass )    fprintf( stdout, "   Accuracy check of potential : Passed !!\n" );
   else           fprintf( stdout, "   Accuracy check of potential : Failed !!\n" );

   
// compare the fluid data
   Pass = true;

   for (int P=0; P<NPatch; P++)
   for (int v=0; v<NCOMP; v++)
   for (int k=0; k<PATCH_SIZE; k++)
   for (int j=0; j<PATCH_SIZE; j++)
   for (int i=0; i<PATCH_SIZE; i++)
   {
      Err = fabsf(  ( CPU_Flu_Array_G[P][v][k][j][i] - GPU_Flu_Array_G[P][v][k][j][i] )
                    / CPU_Flu_Array_G[P][v][k][j][i]  );

      if ( Err > GRA_MAXERR ) 
      {
         fprintf( stderr, "Patch %3d, (i,j,k,v) = (%d, %d, %d, %d), CPU = %14.7e, GPU = %14.7e, Err = %14.7e\n",
                  P, i, j, k, v, CPU_Flu_Array_G[P][v][k][j][i], GPU_Flu_Array_G[P][v][k][j][i], Err ); 

         Pass = false;
      }
   }

   if ( Pass )    fprintf( stdout, "   Accuracy check of fluid     : Passed !!\n" );
   else           fprintf( stdout, "   Accuracy check of fluid     : Failed !!\n" );

}



//----------------------------------------------------------------------
// Function    :  AsynTest
// Description :  test the concurrent execution between CPU and GPU
//----------------------------------------------------------------------
void AsynTest() 
{

   cout << "Asynchronous Test ... " << flush;


   const int ArraySize = CPULOAD*256;

   int *CPUArray = new int [ArraySize];


// initialize CPU array
   for (int i=0; i<ArraySize; i++)     CPUArray[i] = i;


// CPU only
//----------------------------------------------------
timer_CPU_Only->Start();

   for (int i=0; i<ArraySize; i++)     CPUArray[i]++;

timer_CPU_Only->Stop( false );


// GPU only
//----------------------------------------------------
timer_GPU_Only->Start();

   CUAPI_Asyn_PoissonGravitySolver( GPU_Rho_Array_P, GPU_Pot_Array_P_In, GPU_Pot_Array_P_Out, GPU_Flu_Array_G,
                                    POT_GPU_NPGROUP, DT, DH, SOR_MIN_ITER, SOR_MAX_ITER, SOR_OMEGA, 
                                    MG_MAX_ITER, MG_NPRE_SMOOTH, MG_NPOST_SMOOTH, MG_TOLERATED_ERROR, 
                                    4.0*M_PI*NEWTON_G, INT_SCHEME, OPT__GRA_P5_GRADIENT, true, true, 
                                    GPU_NSTREAM );
   CUAPI_Synchronize();

timer_GPU_Only->Stop( false );


// concurrent execution between CPU and GPU
//----------------------------------------------------
timer_Concurrent->Start();

   CUAPI_Asyn_PoissonGravitySolver( GPU_Rho_Array_P, GPU_Pot_Array_P_In, GPU_Pot_Array_P_Out, GPU_Flu_Array_G,
                                    POT_GPU_NPGROUP, DT, DH, SOR_MIN_ITER, SOR_MAX_ITER, SOR_OMEGA, 
                                    MG_MAX_ITER, MG_NPRE_SMOOTH, MG_NPOST_SMOOTH, MG_TOLERATED_ERROR, 
                                    4.0*M_PI*NEWTON_G, INT_SCHEME, OPT__GRA_P5_GRADIENT, true, true,
                                    GPU_NSTREAM );

   for (int i=0; i<ArraySize; i++)     CPUArray[i]++;

   CUAPI_Synchronize();

timer_Concurrent->Stop( false );


   if ( CPUArray[ArraySize-1] != ArraySize+1 )  
      fprintf( stderr, "Error : incorrect result for CPU in the asynchronous test!!\n" );

   delete [] CPUArray;


   cout << "done" << endl;

}



//-------------------------------------------------------------------------------------------------------
// Function    :  Init_Set_Default_SOR_Parameter
// Description :  set the SOR parameters by the default values 
//
// Note        :  the default values are determined empirically from the cosmological simulations
//-------------------------------------------------------------------------------------------------------
void Init_Set_Default_SOR_Parameter()
{

   const real Default_Omega[5] = { 1.49, 1.57, 1.62, 1.65, 1.69 };  // for POT_GHOST_SIZE = [1,2,3,4,5]
#  ifdef FLOAT8
   const int  Default_MaxIter  = 100;
#  else
   const int  Default_MaxIter  = 60;
#  endif
   const int  Default_MinIter  = 10;

   SOR_OMEGA    = Default_Omega[POT_GHOST_SIZE-1];
   SOR_MAX_ITER = Default_MaxIter; 
   SOR_MIN_ITER = Default_MinIter; 

}



//-------------------------------------------------------------------------------------------------------
// Function    :  Init_Set_Default_MG_Parameter
// Description :  Set the multigrid parameters by the default values 
//
// Note        :  1. Work only when the corresponding input parameters are negative
//                2. Default values are determined empirically
//
// Parameter   :  Max_Iter          : Maximum number of iterations for multigrid
//                NPre_Smooth       : Number of pre-smoothing steps for multigrid
//                NPos_tSmooth      : Number of post-smoothing steps for multigrid
//                Tolerated_Error   : Maximum tolerated error for multigrid
//-------------------------------------------------------------------------------------------------------
void Init_Set_Default_MG_Parameter( int &Max_Iter, int &NPre_Smooth, int &NPost_Smooth, real &Tolerated_Error )
{

// REMOVE in the actual implementation
// #########################################################################
   const int MPI_Rank = 0;
// #########################################################################

#  ifdef FLOAT8
   const int  Default_Max_Iter        = 20;
   const real Default_Tolerated_Error = 1.e-15;
#  else
   const int  Default_Max_Iter        = 10;
   const real Default_Tolerated_Error = 1.e-6;
#  endif
   const int  Default_NPre_Smooth     = 3;
   const int  Default_NPost_Smooth    = 3;

   if ( Max_Iter < 0 )     
   {
      Max_Iter = Default_Max_Iter; 

      if ( MPI_Rank == 0 )  Aux_Message( stdout, "NOTE : parameter \"%s\" is set to the default value = %d\n",
                                         "MG_MAX_ITER", Default_Max_Iter );
   }

   if ( NPre_Smooth < 0 )     
   {
      NPre_Smooth = Default_NPre_Smooth;

      if ( MPI_Rank == 0 )  Aux_Message( stdout, "NOTE : parameter \"%s\" is set to the default value = %d\n",
                                         "MG_NPRE_SMOOTH", Default_NPre_Smooth );
   }

   if ( NPost_Smooth < 0 )     
   {
      NPost_Smooth = Default_NPost_Smooth;

      if ( MPI_Rank == 0 )  Aux_Message( stdout, "NOTE : parameter \"%s\" is set to the default value = %d\n",
                                         "MG_NPOST_SMOOTH", Default_NPost_Smooth );
   }

   if ( Tolerated_Error < 0.0 )     
   {
      Tolerated_Error = Default_Tolerated_Error;

      if ( MPI_Rank == 0 )  Aux_Message( stdout, "NOTE : parameter \"%s\" is set to the default value = %13.7e\n",
                                         "MG_TOLERATED_ERROR", Default_Tolerated_Error );
   }

} // FUNCTION : Init_Set_Default_MG_Parameter

