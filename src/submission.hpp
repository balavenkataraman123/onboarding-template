#pragma once

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <omp.h> 
#include <memory>

class Grid {
private:
  std::size_t rows_;
  std::size_t cols_;
  double* data_; 
  

public:
  Grid(std::size_t rows, std::size_t cols): rows_{rows}, cols_{cols}, data_{ static_cast<double*>(std::aligned_alloc(64, rows * cols * sizeof(double)))}{} // aligned to 64 byte cache line size.
 ~Grid() { std::free(data_); }

  double& operator()(std::size_t i, std::size_t j) {
    return data_[i * cols_ + j];
  }

  double operator()(std::size_t i, std::size_t j) const {
    return data_[i * cols_ + j];
  }

  double* data() { return data_; }
  const double* data() const { return data_; }

  int num_rows() const { return rows_; }
  int num_cols() const { return cols_; }
  bool boundary_copied = false; // flag to indicate if the boundary has been copied or not, so boundary is not copied multiple times.
};



void five_point_stencil_helper (const double* __restrict__ old_list, double* __restrict__ new_list, int num_cols, int num_rows){

  const int stride = 16;
  // with a stride length of 16, each block accesses 16 x 1024 x 2  x sizeof(double) = 0.25MB. My laptop CPU has 1.2MB L2 cache per core
  // at 2 threads a core, this is still within the limits

  #pragma omp parallel for
  for(int block = 0; block < (num_rows + stride - 1) / stride; ++block) { 
    for(int row = std::max(1, block * stride); row < std::min(block * stride + stride, num_rows - 1); ++row){ // max(1, ---) to avoid boundary rows
      const double* old_top = old_list + (row - 1) * num_cols + 1;
      const double* old_center = old_list + row * num_cols + 1;
      const double* old_bottom = old_list + (row + 1) * num_cols + 1;
      double* new_center = new_list + row * num_cols + 1;
      #pragma omp simd // as the output list elements are independent, this can be run with SIMD
      for (int i = 0; i < num_cols - 2; ++i) {
          new_center[i] = 0.5 * old_center[i] + 0.125 * (old_top[i] + old_bottom[i] + old_center[i-1] + old_center[i+1]);
      } 
    }
  }
}


void apply_stencil(const Grid& old_grid, Grid& new_grid) { // transfers boundaries if needed, calls on a helper function with only the aligned data pointers.
  const std::size_t rows = old_grid.num_rows();
  const std::size_t cols = old_grid.num_cols();
  // copies the boundary values from old_grid to new_grid if they haven't been copied yet
  if (old_grid.boundary_copied == false) {
    for (std::size_t row{}; row < rows; ++row) {
      new_grid(row, 0) = old_grid(row, 0);
      new_grid(row, cols - 1) = old_grid(row, cols - 1);
    }

    for (std::size_t col{}; col < cols; ++col) {
      new_grid(0, col) = old_grid(0, col);
      new_grid(rows - 1, col) = old_grid(rows - 1, col);
    }
  }
  new_grid.boundary_copied = true;
  five_point_stencil_helper(
    old_grid.data(),
    new_grid.data(),
    cols,
    rows
  ); 


}
