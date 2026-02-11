// BaselineRem.h

#ifndef BaselineRem_h
#define BaselineRem_h

#include <Arduino.h>

class BaselineRem {
public:
    BaselineRem();  // Constructor
    ~BaselineRem(); // Destructor

    // Static method to remove the baseline from a dataset
    static void removeBaseline(float* data, int length);
};

#endif