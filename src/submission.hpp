#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <new>

#include <omp.h>


class Grid {
public:
    static constexpr std::size_t Alignment = 64;

    Grid(std::size_t rows, std::size_t cols)
        : rows_(rows),
          cols_(cols),
          data_(allocate(rows, cols))
    {}

    ~Grid() {
        std::free(data_);
    }

    // Grid owns raw allocated memory, so copying must be disabled.
    Grid(const Grid&) = delete;
    Grid& operator=(const Grid&) = delete;

    // Moving is safe and useful if the harness/container requires it.
    Grid(Grid&& other) noexcept
        : rows_(other.rows_),
          cols_(other.cols_),
          data_(other.data_),
          boundary_initialized_(other.boundary_initialized_)
    {
        other.data_ = nullptr;
        other.boundary_initialized_ = false;
    }

    Grid& operator=(Grid&& other) noexcept {
        if (this != &other) {
            std::free(data_);

            rows_ = other.rows_;
            cols_ = other.cols_;
            data_ = other.data_;
            boundary_initialized_ = other.boundary_initialized_;

            other.data_ = nullptr;
            other.boundary_initialized_ = false;
        }

        return *this;
    }


    double& operator()(std::size_t row, std::size_t col) {
        return data_[row * cols_ + col];
    }

    double operator()(std::size_t row, std::size_t col) const {
        return data_[row * cols_ + col];
    }


    double* data() noexcept {
        return data_;
    }

    const double* data() const noexcept {
        return data_;
    }


    std::size_t rows() const noexcept {
        return rows_;
    }

    std::size_t cols() const noexcept {
        return cols_;
    }


    bool boundary_initialized() const noexcept {
        return boundary_initialized_;
    }

    void mark_boundary_initialized() noexcept {
        boundary_initialized_ = true;
    }


private:
    static double* allocate(
        std::size_t rows,
        std::size_t cols
    ) {
        const std::size_t bytes =
            rows * cols * sizeof(double);

        // std::aligned_alloc requires the requested size
        // to be a multiple of the alignment.
        const std::size_t aligned_bytes =
            ((bytes + Alignment - 1) / Alignment) * Alignment;

        void* memory =
            std::aligned_alloc(Alignment, aligned_bytes);

        if (memory == nullptr) {
            throw std::bad_alloc{};
        }

        return static_cast<double*>(memory);
    }


    std::size_t rows_;
    std::size_t cols_;
    double* data_;

    // The destination grid's boundary is copied only once.
    bool boundary_initialized_ = false;
};


// -----------------------------------------------------------------------------
// Five-point stencil
// -----------------------------------------------------------------------------

namespace stencil {

constexpr double CenterWeight   = 0.5;
constexpr double NeighborWeight = 0.125;

// Number of rows processed by each cache block.
// This should ideally be tuned experimentally for the target CPU.
constexpr std::size_t BlockRows = 16;


// Copy the fixed boundary from the old grid to the new grid.
//
// This is intentionally called only once for each destination buffer.
// Subsequent stencil iterations leave the boundary untouched.
inline void copy_boundary(
    const Grid& old_grid,
    Grid& new_grid
) {
    const std::size_t rows = old_grid.rows();
    const std::size_t cols = old_grid.cols();

    // Left and right boundaries.
    for (std::size_t row = 0; row < rows; ++row) {
        new_grid(row, 0)       = old_grid(row, 0);
        new_grid(row, cols - 1) = old_grid(row, cols - 1);
    }

    // Top and bottom boundaries.
    for (std::size_t col = 0; col < cols; ++col) {
        new_grid(0, col)        = old_grid(0, col);
        new_grid(rows - 1, col) = old_grid(rows - 1, col);
    }
}


// Compute the interior of the grid.
//
// The boundary is deliberately NOT written here. It is initialized
// separately by copy_boundary() and remains unchanged between steps.
inline void compute(
    const double* __restrict__ old_data,
    double* __restrict__ new_data,
    std::size_t rows,
    std::size_t cols
) {
    #pragma omp parallel for
    for (
        std::size_t block = 0;
        block < (rows + BlockRows - 1) / BlockRows;
        ++block
    ) {
        // Skip the boundary row at the top of the grid.
        const std::size_t first_row =
            std::max<std::size_t>(
                1,
                block * BlockRows
            );

        // Exclude the boundary row at the bottom of the grid.
        const std::size_t last_row =
            std::min(
                block * BlockRows + BlockRows,
                rows - 1
            );

        for (std::size_t row = first_row; row < last_row; ++row) {

            // Start at column 1 so that the SIMD loop only
            // operates on interior cells.
            const double* top =
                old_data + (row - 1) * cols + 1;

            const double* center =
                old_data + row * cols + 1;

            const double* bottom =
                old_data + (row + 1) * cols + 1;

            double* output =
                new_data + row * cols + 1;


            #pragma omp simd
            for (std::size_t col = 0; col < cols - 2; ++col) {

                output[col] =
                    CenterWeight * center[col]
                    + NeighborWeight * (
                        top[col]
                        + bottom[col]
                        + center[col - 1]
                        + center[col + 1]
                    );
            }
        }
    }
}

} // namespace stencil


// -----------------------------------------------------------------------------
// Public stencil operation
// -----------------------------------------------------------------------------

inline void apply_stencil(
    const Grid& old_grid,
    Grid& new_grid
) {
    // The harness provides an uninitialized destination buffer.
    // Copy its fixed boundary only the first time this buffer is used.
    if (!new_grid.boundary_initialized()) {
        stencil::copy_boundary(old_grid, new_grid);
        new_grid.mark_boundary_initialized();
    }

    // Compute only the interior. The boundary is never overwritten.
    stencil::compute(
        old_grid.data(),
        new_grid.data(),
        old_grid.rows(),
        old_grid.cols()
    );
}
