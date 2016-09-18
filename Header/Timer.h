#ifndef __TIMER_H__
#define __TIMER_H__



#include "sys/time.h"


//-------------------------------------------------------------------------------------------------------
// Structure   :  Timer_t 
// Description :  Data structure for measuring the elapsed time
//
// Data Member :  Status    : The status of each timer : (false / true) <--> (stop / ticking)
//                Time      : The variable recording the elapsed time (in microsecond)
//                WorkingID : The currently working ID of the array "Time"
//                NTimer    : The number of timers
//
// Method      :  Timer_t   : Constructor 
//               ~Timer_t   : Destructor
//                Start     : Start timing
//                Stop      : Stop timing
//                GetValue  : Get the elapsed time recorded in timer (in second)
//                Reset     : Reset timer
//-------------------------------------------------------------------------------------------------------
struct Timer_t
{

// data members
// ===================================================================================
   bool  *Status;
   ulong *Time;
   uint  WorkingID;
   uint  NTimer;



   //===================================================================================
   // Constructor :  Timer_t 
   // Description :  Constructor of the structure "Timer_t"
   //
   // Note        :  Allocate and initialize all data member 
   //                a. The WorkingID is initialized as "0"
   //                b. The Status is initialized as "false"
   //                c. The timer is initialized as "0"
   //
   // Parameter   :  N : The number of timers to be allocated
   //===================================================================================
   Timer_t( const uint N )
   {
      NTimer    = N;
      WorkingID = 0;
      Time      = new ulong [NTimer];
      Status    = new bool  [NTimer];

      for (uint t=0; t<NTimer; t++)     
      {
         Time  [t] = 0;
         Status[t] = false;
      }
   } 



   //===================================================================================
   // Destructor  :  ~Timer_t 
   // Description :  Destructor of the structure "Timer_t"
   //
   // Note        :  Release memory 
   //===================================================================================
   ~Timer_t()
   {
      delete [] Time;
      delete [] Status;
   } 



   //===================================================================================
   // Method      :  Start
   // Description :  Start timing and set status as "true"
   //===================================================================================
   void Start()
   {
      if ( WorkingID >= NTimer )
      {
         fprintf( stderr, "ERROR : timer is exhausted !!\n" );
         exit(-1);
      }

      if ( Status[WorkingID] )
         fprintf( stderr, "WARNING : the timer has already been started (WorkingID = %d) !!\n", WorkingID );

      timeval tv;
      gettimeofday( &tv, NULL );

      Time[WorkingID] = tv.tv_sec*1000000 + tv.tv_usec - Time[WorkingID];

      Status[WorkingID] = true;
   }



   //===================================================================================
   // Method      :  Stop
   // Description :  Stop timing and set status as "false" 
   //
   // Note        :  Increase the WorkingID if the input parameter "Next == true"
   //
   // Parameter   :  Next : (true / false) --> (increase / not increase) the WorkingID 
   //===================================================================================
   void Stop( const bool Next )
   {
      if ( WorkingID >= NTimer )
      {
         fprintf( stderr, "ERROR : timer is exhausted !!\n" );
         exit(-1);
      }

      if ( !Status[WorkingID] )
         fprintf( stderr, "WARNING : the timer has NOT been started (WorkingID = %d) !!\n", WorkingID );

      timeval tv;
      gettimeofday( &tv, NULL );

      Time[WorkingID] = tv.tv_sec*1000000 + tv.tv_usec - Time[WorkingID];

      Status[WorkingID] = false;

      if ( Next )    WorkingID++;
   }



   //===================================================================================
   // Method      :  GetValue
   // Description :  Get the elapsed time (in second) recorded in the timer "TargetID"
   //
   // Parameter   :  TargetID : The ID of the targeted timer
   //===================================================================================
   float GetValue( const uint TargetID )
   {
      if ( TargetID >= NTimer )
      {
         fprintf( stderr, "ERROR : the queried timer does NOT exist (TargetID = %d, NTimer = %d) !!\n",
                  TargetID, NTimer );
         exit(-1);
      }

      if ( Status[TargetID] )
         fprintf( stderr, "WARNING : the timer is still ticking (TargetID = %d) !!\n", TargetID );

      return Time[TargetID]*1.e-6f;
   }



   //===================================================================================
   // Method      :  Reset
   // Description :  Reset all timers and set WorkingID as "0" 
   //
   // Note        :  All timing information previously recorded will be lost after 
   //                invoking this function 
   //===================================================================================
   void Reset() 
   {
      for (uint t=0; t<NTimer; t++)
      {
         if ( Status[t] )
            fprintf( stderr, "WARNING : resetting a using timer (WorkingID = %d) !!\n", t );

         Time[t]   = 0;
         WorkingID = 0;
      }
   }


}; // struct Timer_t



#endif // #ifndef __TIMER_H__
