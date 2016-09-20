#include "Copyright.h"
#include "Macro.h"
#include "CUPOT.h"

#if ( defined GRAVITY  &&  defined GPU )

	
//-------------------------------------------------------------------------------------------------------
// Function    :  blockReduceSum 
// Description :  GPU reduction using register shuffling. Sums up the elements in one warp
//
// Note        :  a. increases reduction speed by using registers
//				  b. reference: https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
//
// Parameter   :  val   : the value that each thread holds that will be added together
//
// Return value:  Sum of the warp
//---------------------------------------------------------------------------------------------------
__inline__ __device__
real warpReduceSum(real val) 
{
  for (int offset = warpSize/2; offset > 0; offset /= 2) 
    val += __shfl_down(val, offset, warpSize);
  return val;
}

//-------------------------------------------------------------------------------------------------------
// Function    :  blockReduceSum 
// Description :  GPU reduction using register shuffling. Sums up the elements in one block.
//
// Note        :  a. increases reduction speed by using registers
//				  b. reference: https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
//
// Parameter   :  val   : the value that each thread holds that will be added together
//                ID    : index of each individual thread
//
// Return value:  sum of the block
//---------------------------------------------------------------------------------------------------
__inline__ __device__
real blockReduceSum(real val, int ID) 
{

  static __shared__ real shared[32]; // Shared mem for 32 partial sums
  int lane = ID % warpSize;
  int wid = ID / warpSize;

  if(ID < blockDim.x * blockDim.y * blockDim.z)
	val = warpReduceSum(val); 

  // Write reduced value to shared memory
  if (lane==0) shared[wid]=val; 

  // Wait for all partial reductions
  __syncthreads();              

  //read from shared memory only if that warp existed
  val = (ID < warpSize) ? shared[lane] : 0.0;

  //Final reduce within first warp
  if (wid==0) val = warpReduceSum(val); 

  return val;
}


#endif // #if ( defined GRAVITY  &&  defined GPU )
