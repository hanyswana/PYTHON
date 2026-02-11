// NormEucLib.cpp

#include "NormEucLib.h"
#include <cmath> // Include cmath for sqrt and fabs

// Implementations of NormEucLib methods

float NormEucLib::euclidean_norm(float* arr, int length) {
    float norm = 0;
    for (int i = 0; i < length; i++) {
        norm += arr[i] * arr[i]; // Sum of squares
    }
    return sqrt(norm); // Square root of sum of squares
}

void NormEucLib::normalize_with_euclidean(float* arr, int length) {
    float norm = euclidean_norm(arr, length);
    if (norm == 0) return; // Avoid division by zero

    for (int i = 0; i < length; i++) {
        arr[i] /= norm; // Normalize each element
    }
}
