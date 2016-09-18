#ifndef __GAMER_H__
#define __GAMER_H__



#include <iostream>
#include <cstdlib>
#include <cmath>
#include <stdio.h>
#include <unistd.h>
#include "Macro.h"
#include "Typedef.h"
#include "Prototype.h"

#ifdef OPENMP
   #include <omp.h>
#endif

#ifdef __CUDACC__
   #include "CUAPI.h"
#endif

using namespace std;



#endif // #ifndef __GAMER_H__
