// NormManhLib.cpp

#include "NormManhLib.h"

float NormManhLib::manhattan_norm(float* arr, int length) {
    float norm = 0;
    for (int i = 0; i < length; i++) {
        norm += fabs(arr[i]); // Use fabs for absolute value
    }
    return norm;
}

void NormManhLib::normalize_with_manhattan(float* arr, int length) {
    float norm = manhattan_norm(arr, length);
    if (norm == 0) return; // Avoid division by zero

    for (int i = 0; i < length; i++) {
        arr[i] /= norm; // Normalize each element
    }
}