#include "BaselineRem.h"
#include <numeric> // For std::accumulate

// Constructor
BaselineRem::BaselineRem() {}

// Destructor
BaselineRem::~BaselineRem() {}

// Static method to remove the baseline from a dataset
void BaselineRem::removeBaseline(float* data, int length) {
    // Calculate the mean of the data array
    float sum = std::accumulate(data, data + length, 0.0f);
    float mean = sum / static_cast<float>(length); // Ensure floating-point division

    // Subtract the mean from each element to remove the baseline
    for (int i = 0; i < length; ++i) {
        data[i] -= mean;
    }
}
