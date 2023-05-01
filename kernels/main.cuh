#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include "../src/cam_params.hpp"
#include "../src/constants.hpp"
#include <vector>

// This is the public interface of our cuda function, called directly in main.cpp
//void wrap_test_vectorA

float* compute_shared_cost_lauch(cam const ref, std::vector<cam> const& cam_vector, int window);
float* compute_naive_cost_lauch(cam const ref, std::vector<cam> const& cam_vector, int window);
float* compute_naive_cost_float2D_lauch(cam const ref, std::vector<cam> const& cam_vector, int window);
float* compute_shared_constante_(cam const ref, std::vector<cam> const& cam_vector, int window);