#include "GAMER.h"
#include <cstdarg>




//-------------------------------------------------------------------------------------------------------
// Function    :  Aux_Error
// Description :  Output the error messages and force the program to be terminated
//
// Note        :  Use the variable argument lists provided in "cstdarg" 
// 
// Parameter   :  File     : Name of the file where error occurs 
//                Line     : Line number where error occurs
//                Func     : Name of the function where error occurs
//                Format   : Output format
//                ...      : Arguments in vfprintf
//-------------------------------------------------------------------------------------------------------
void Aux_Error( const char *File, const int Line, const char *Func, const char *Format, ... )
{

// flush all previous messages
   fflush( stdout ); fflush( stdout ); fflush( stdout );
   fflush( stderr ); fflush( stderr ); fflush( stderr );


// output error messages
   va_list Arg;
   va_start( Arg, Format );

   fprintf ( stderr, "********************************************************************************\n" );
   fprintf ( stderr, "ERROR : " );
   vfprintf( stderr, Format, Arg );
// fprintf ( stderr, "        Rank <%d>, file <%s>, line <%d>, function <%s>\n", GAMER_RANK, File, Line, Func );
   fprintf ( stderr, "        Rank <%d>, file <%s>, line <%d>, function <%s>\n", 0, File, Line, Func );
   fprintf ( stderr, "********************************************************************************\n" );

   va_end( Arg );


// terminate the program   
   MPI_Exit();

} // FUNCTION : Aux_Error

