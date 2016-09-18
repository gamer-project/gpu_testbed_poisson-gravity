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

int CUPOT_PoissonSolver_SetConstMem();

void Aux_GetCPUInfo( const char *FileName );




//-----------------------------------------------------------------------------------------
// Function    :  CUAPI_Set_Diagnose_Device
// Description :  set the active device and take a diagnosis of it
//
// Parameter   :  Pot_GPU_NPGroup   : POT_GPU_NPGROUP
//                GPU_NStream       : GPU_NSTREAM
//                InputID           : GPU ID
//-----------------------------------------------------------------------------------------
void CUAPI_Set_Diagnose_Device( int &Pot_GPU_NPGroup, int &GPU_NStream, const int InputID )
{

//  get the hostname and PID of each process
    char Host[1024];
    gethostname( Host, 1024 );
    const int PID = getpid();

// verify that there are GPU supporing CUDA
   int DeviceCount;
   cudaGetDeviceCount( &DeviceCount );
   if ( DeviceCount == 0 )   
   {
      fprintf( stderr, "ERROR : no devices supporting CUDA at %8s !!\n", Host );
      exit(-1);
   }

   if ( InputID >= DeviceCount )
   {
      fprintf( stderr, "ERROR : Input GPU ID (%d) >= # of GPUs (%d) !!\n", InputID, DeviceCount );
      exit(-1);
   }

// set the device ID (e.g. NGPU_PER_NODE=2 : Rank = [0,1,2,3,4,5...] <--> DeviceID = [0,1,0,1,0,1...] )
   const int SetDeviceID = InputID;
   cudaSetDevice( SetDeviceID );

// verify the device ID
   int GetDeviceID = 999;
   cudaGetDevice( &GetDeviceID );
   if ( GetDeviceID != SetDeviceID )
   {
      fprintf( stderr, "ERROR : GetDeviceID (%d) != SetDeviceID (%d) !!\n", GetDeviceID, SetDeviceID );
      exit(-1);
   }

// load the device properties
   cudaDeviceProp DeviceProp;
   cudaGetDeviceProperties( &DeviceProp, SetDeviceID );

// verify the device version
   if ( DeviceProp.major < 1 )
   {
      fprintf( stderr, "ERROR : device major version < 1 at %8s !!\n", Host );
      exit(-1);
   }

// verify the capability of double precision 
#  ifdef FLOAT8
   if ( DeviceProp.major < 2  &&  DeviceProp.minor < 3 )
   {
      fprintf( stderr, "ERROR : the GPU \"%s\" does not support double precision !!\n", DeviceProp.name );
      MPI_Exit();
   }
#  endif

// verify the GPU architecture
#  if   ( GPU_ARCH == FERMI )
   if ( DeviceProp.major != 2 )
   {
      fprintf( stderr, "ERROR : the GPU \"%s\" is incompatible to the Fermi architecture (major revision = %d) !!\n", 
               DeviceProp.name, DeviceProp.major );
      MPI_Exit();
   }

#  elif ( GPU_ARCH == KEPLER )
   if ( DeviceProp.major != 3 )
   {
      fprintf( stderr, "ERROR : the GPU \"%s\" is incompatible to the Kepler architecture (major revision = %d) !!\n", 
               DeviceProp.name, DeviceProp.major );
      MPI_Exit();
   }

#  else
#  error : UNKNOWN GPU_ARCH !!
#  endif

// get the number of cores per multiprocessor
   int NCorePerMP;
   if      ( DeviceProp.major == 2  &&  DeviceProp.minor == 0 )  NCorePerMP =  32;
   else if ( DeviceProp.major == 2  &&  DeviceProp.minor == 1 )  NCorePerMP =  48;
   else if ( DeviceProp.major == 3 )                             NCorePerMP = 192;
   else if ( DeviceProp.major == 5 )                             NCorePerMP = 128;
   else
      fprintf( stderr, "WARNING : unable to determine then umber of cores per multiprocessor for version %d.%d ...\n",
               DeviceProp.major, DeviceProp.minor );

// get the version of driver and CUDA
   int DriverVersion = 0, RuntimeVersion = 0;     

   cudaDriverGetVersion( &DriverVersion );
   cudaRuntimeGetVersion( &RuntimeVersion );
// record the device properties
   const char FileName[] = "Note";
   FILE *Note = fopen( FileName, "a" );
   fprintf( Note, "Device Diagnosis\n" );
   fprintf( Note, "***********************************************************************************\n" );
   
   fprintf( Note, "hostname = %8s ; PID = %d\n\n", Host, PID );
   fprintf( Note, "CPU Info :\n" );
   fflush( Note );

   Aux_GetCPUInfo( FileName );
   
   fprintf( Note, "\n" );
   fprintf( Note, "GPU Info :\n" );
   fprintf( Note, "Number of Devices                 : %d\n"    , DeviceCount );
   fprintf( Note, "Device ID                         : %d\n"    , SetDeviceID );
   fprintf( Note, "Device Name                       : %s\n"    , DeviceProp.name );
   fprintf( Note, "CUDA Driver Version               : %d.%d\n" , DriverVersion/1000, DriverVersion%100 );
   fprintf( Note, "CUDA Runtime Version              : %d.%d\n" , RuntimeVersion/1000, RuntimeVersion%100 );
   fprintf( Note, "CUDA Major Revision Number        : %d\n"    , DeviceProp.major );
   fprintf( Note, "CUDA Minor Revision Number        : %d\n"    , DeviceProp.minor );
   fprintf( Note, "Clock Rate                        : %f GHz\n", DeviceProp.clockRate/1.e6);
   fprintf( Note, "Global Memory Size                : %d MB\n" , DeviceProp.totalGlobalMem/1024/1024 ); 
   fprintf( Note, "Constant Memory Size              : %d KB\n" , DeviceProp.totalConstMem/1024 ); 
   fprintf( Note, "Shared Memory Size per Block      : %d KB\n" , DeviceProp.sharedMemPerBlock/1024 ); 
   fprintf( Note, "Number of Registers per Block     : %d\n"    , DeviceProp.regsPerBlock );
   fprintf( Note, "Warp Size                         : %d\n"    , DeviceProp.warpSize );
   fprintf( Note, "Number of Multiprocessors:        : %d\n"    , DeviceProp.multiProcessorCount );
   fprintf( Note, "Number of Cores per Multiprocessor: %d\n"    , NCorePerMP );
   fprintf( Note, "Total Number of Cores:            : %d\n"    , DeviceProp.multiProcessorCount * NCorePerMP );
   fprintf( Note, "Max Number of Threads per Block   : %d\n"    , DeviceProp.maxThreadsPerBlock );
   fprintf( Note, "Max Size of the Block X-Dimension : %d\n"    , DeviceProp.maxThreadsDim[0] );
   fprintf( Note, "Max Size of the Grid X-Dimension  : %d\n"    , DeviceProp.maxGridSize[0] );
   fprintf( Note, "Concurrent Copy and Execution     : %s\n"    , DeviceProp.asyncEngineCount>0  ? "Yes" : "No" );
   fprintf( Note, "Concurrent Up/Downstream Copies   : %s\n"    , DeviceProp.asyncEngineCount==2 ? "Yes" : "No" );
#  if ( CUDART_VERSION >= 3000 )
   fprintf( Note, "Concurrent Kernel Execution       : %s\n"    , DeviceProp.concurrentKernels ? "Yes" : "No" );
#  endif
#  if ( CUDART_VERSION >= 3010 )
   fprintf( Note, "Device has ECC Support Enabled    : %s\n"    , DeviceProp.ECCEnabled ? "Yes" : "No" );
#  endif

   fprintf( Note, "\n\n" );
   
   fprintf( Note, "***********************************************************************************\n" );

   fclose( Note );


// set POT_GPU_NPGROUP and GPU_NSTREAM to the default values if they are not set by the command-line inputs
   if ( GPU_NStream == NULL_INT )
   {
      if ( DeviceProp.deviceOverlap )
      {
#        if   ( GPU_ARCH == FERMI )
         GPU_NStream = 8;
#        elif ( GPU_ARCH == KEPLER )
         GPU_NStream = 48;             // optimized for K40
#        else
#        error : UNKNOWN GPU_ARCH !!
#        endif
      }

      else
         GPU_NStream = 1;
   }

   if ( Pot_GPU_NPGroup == NULL_INT )     
   {
#     if   ( GPU_ARCH == FERMI )
      Pot_GPU_NPGroup = 2*GPU_NStream*DeviceProp.multiProcessorCount;
#     elif ( GPU_ARCH == KEPLER )
      Pot_GPU_NPGroup = 32*DeviceProp.multiProcessorCount;  // optimized for K40 (for GPU_NStream=48, NPGroup=480)
#     else
#     error : UNKNOWN GPU_ARCH !!
#     endif
   }


// set the cache preference
#  if   ( POT_SCHEME == SOR )
#  ifdef USE_PSOLVER_10TO14
   CUDA_CHECK_ERROR( cudaFuncSetCacheConfig( CUPOT_PoissonSolver_SOR_10to14cube, cudaFuncCachePreferShared ) );
#  else
   CUDA_CHECK_ERROR( cudaFuncSetCacheConfig( CUPOT_PoissonSolver_SOR_16to18cube, cudaFuncCachePreferShared ) );
#  endif
#  elif ( POT_SCHEME == MG  )
   CUDA_CHECK_ERROR( cudaFuncSetCacheConfig( CUPOT_PoissonSolver_MG,             cudaFuncCachePreferShared ) );
#  endif // POT_SCHEME

   CUDA_CHECK_ERROR( cudaFuncSetCacheConfig( CUPOT_GravitySolver,                cudaFuncCachePreferShared ) );


// set the constant variables
   if ( CUPOT_PoissonSolver_SetConstMem() != 0 )
   Aux_Error( ERROR_INFO, "CUPOT_PoissonSolver_SetConstMem failed ...\n" );

} // FUNTION : CUAPI_Set_Diagnose_Device



//-------------------------------------------------------------------------------------------------------
// Function    :  Aux_GetCPUInfo
// Description :  Record the CPU information 
// 
// Parameter   :  FileName : Name of the output file
//-------------------------------------------------------------------------------------------------------
void Aux_GetCPUInfo( const char *FileName )
{

   FILE *Note = fopen( FileName, "a" );
   char *line = NULL;
   size_t len = 0;
   char String[2][100];


// 1. get the CPU info
   const char *CPUInfo_Path = "/proc/cpuinfo";
   FILE *CPUInfo = fopen( CPUInfo_Path, "r" );

   if ( CPUInfo == NULL )
   {
      fprintf( stderr, "WARNING : the CPU information file \"%s\" does not exist !!\n", CPUInfo_Path );
      return;
   }

   while ( getline(&line, &len, CPUInfo) != -1 ) 
   {
      sscanf( line, "%s%s", String[0], String[1] );

      if (  strcmp( String[0], "model" ) == 0  &&  strcmp( String[1], "name" ) == 0  )    
      {
         strncpy( line, "CPU Type  ", 10 );
         fprintf( Note, "%s", line );
      }

      if (  strcmp( String[0], "cpu" ) == 0  &&  strcmp( String[1], "MHz" ) == 0  )    
      {
         strncpy( line, "CPU MHz", 7 );
         fprintf( Note, "%s", line );
      }

      if (  strcmp( String[0], "cache" ) == 0  &&  strcmp( String[1], "size" ) == 0  )    
      {
         strncpy( line, "Cache Size", 10 );
         fprintf( Note, "%s", line );
      }

      if (  strcmp( String[0], "cpu" ) == 0  &&  strcmp( String[1], "cores" ) == 0  )    
      {
         strncpy( line, "CPU Cores", 9 );
         fprintf( Note, "%s", line );
         break;
      }
   }

   if ( line != NULL )  
   {
      free( line );
      line = NULL;
   }

   fclose( CPUInfo );


// 2. get the memory info
   const char *MemInfo_Path = "/proc/meminfo";
   FILE *MemInfo = fopen( MemInfo_Path, "r" );

   if ( MemInfo == NULL )
   {
      fprintf( stderr, "WARNING : the memory information file \"%s\" does not exist !!\n", MemInfo_Path );
      return;
   }

   while ( getline(&line, &len, MemInfo) != -1 ) 
   {
      sscanf( line, "%s%s", String[0], String[1] );

      if (  strncmp( String[0], "MemTotal", 8 ) == 0  )
      {
         fprintf( Note, "Total Memory    : %4.1f GB\n", atof( String[1] )/(1<<20) );
         break;
      }
   }

   if ( line != NULL )  
   {
      free( line );
      line = NULL;
   }

   fclose( MemInfo );
   fclose( Note );

} // FUNCTION : Aux_GetCPUInfo

