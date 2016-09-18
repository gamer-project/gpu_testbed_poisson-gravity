#ifndef __MACRO_H__
#define __MACRO_H__



// ########################################
// ### Symbolic Constants and Data Type ###
// ########################################

// Poisson solvers
#ifdef GRAVITY
#define SOR       1
#define MG        2
#endif


// GPU architecture
#define FERMI     1
#define KEPLER    2
#define MAXWELL   3


// single/double precision
#ifdef FLOAT8
typedef double real;
#else
typedef float  real;
#endif


// short names for unsigned type
typedef unsigned int       uint;
typedef unsigned long int  ulong;


// number of variables and patch size
#define NCOMP           5
#define PS1             ( 1*PATCH_SIZE )


// number of density ghost zones for the Poisson solver
#define RHO_GHOST_SIZE  ( POT_GHOST_SIZE-1 )

// number of coarse-grid cells for potential interpolation
#define POT_INT         ( (POT_GHOST_SIZE == 0) ? 0 : (POT_GHOST_SIZE+3)/2 ) 

// the size of arrays (in one dimension) sending into GPU for the Poisson solver
#define POT_NXT         ( PATCH_SIZE/2 + 2*POT_INT        )    
#define RHO_NXT         ( PATCH_SIZE   + 2*RHO_GHOST_SIZE )
#define GRA_NXT         ( PATCH_SIZE   + 2*GRA_GHOST_SIZE )


// extreme and NULL values
#ifndef __INT_MAX__
   #define __INT_MAX__     2147483647
#endif

#ifndef __FLT_MAX__
   #define __FLT_MAX__     3.40282347e+38F
#endif

#ifndef NULL_INT
   #define NULL_INT        __INT_MAX__
#endif



// #############
// ### Macro ###
// #############

// macro for the function "Aux_Error"
#define ERROR_INFO         __FILE__, __LINE__, __FUNCTION__

// single/double-precision mathematic functions
#ifdef FLOAT8
   #define FABS   fabs
#else
   #define FABS   fabsf
#endif



#endif  // #ifndef __MACRO_H__
