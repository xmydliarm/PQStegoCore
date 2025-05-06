#include "JPEGProcessor.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <jpeglib.h>
#include <csetjmp>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <vector>

/**
 * @brief Estimates the JPEG compression quality factor from the quantization matrix.
 *
 * Uses the standard JPEG Q50 matrix to reverse-engineer the quality factor
 * by comparing it with the image’s actual quantization table.
 *
 * @param[in] jpeg_file Pointer to the JPEGFile object containing the quantization matrix.
 * @return Estimated JPEG quality (integer in the range ~1–100).
 *
 * @note Uses the median of computed quality estimates over the 64 quantization values.
 */
int JPEGProcessor::EstimateQuality(const JPEGFile* jpeg_file) {
    const std::vector<std::vector<int> > Q50 = {
        {16, 11, 10, 16, 24, 40, 51, 61},
        {12, 12, 14, 19, 26, 58, 60, 55},
        {14, 13, 16, 24, 40, 57, 69, 56},
        {14, 17, 22, 29, 51, 87, 80, 62},
        {18, 22, 37, 56, 68, 109, 103, 77},
        {24, 35, 55, 64, 81, 104, 113, 92},
        {49, 64, 78, 87, 103, 121, 120, 101},
        {72, 92, 95, 98, 112, 100, 103, 99}
    };

    const auto& Qm = jpeg_file->getQM();

    const bool only_ones = std::ranges::all_of(Qm, [](const std::vector<int>& row) {
        return std::ranges::all_of(row, [](const int val) { return val == 1; });
    });

    // Qm has only ones
    if (only_ones) {
        return 100;
    }

    // Floor values
    auto results = std::vector<int>();
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            double x = 100 - (100.0 * Qm[i][j] - 50) / (2.0 * Q50[i][j]);
            int casted_x = static_cast<int>(x);
            results.emplace_back(casted_x);
        }
    }

    // Compute median
    std::ranges::stable_sort(results);
    int median = results[64 / 2 - 1];

    return median;
}

/**
 * @brief Calculates the DCT coefficients for an image and organizes them into blocks.
 *
 * This function computes the discrete cosine transform (DCT) coefficients for the input 2D image matrix `X`.
 * The image is divided into 8x8 blocks, and the DCT coefficients for each block are stored in a 3D array `D_raw`.
 * If the dimensions of the image are not multiples of 8, the image is cropped to the nearest 8-pixel boundary.
 *
 * @param[in] X Input 2D matrix representing the image (grayscale intensity values).
 * @param[out] D_raw 3D array to store the calculated DCT coefficients.
 *                  - The first two dimensions represent the block indices.
 *                  - The third dimension contains the 64 coefficients of the corresponding 8x8 DCT block, arranged column-major.
 */
void JPEGProcessor::ComputeDCTBlocks(const std::vector<std::vector<double>>& X, std::vector<std::vector<std::vector<double>>>& D_raw) {
    constexpr int B = 8; // Block size
    constexpr int B2 = B * B;

    const int M = (int)X.size(); // Number of rows
    const int N = (int)X[0].size(); // Number of columns

    const int MB = M / B; // Number of blocks along rows
    const int NB = N / B; // Number of blocks along columns

    // Resize the D matrix to hold the DCT coefficients
    D_raw.resize(MB, std::vector(NB, std::vector<double>(B2, 0)));

    // Loop over all blocks
    for (int i = 0; i < MB; ++i) {
        for (int j = 0; j < NB; ++j) {
            const int ib = i * B;
            const int jb = j * B;

            // Extract the 8x8 block from the original matrix X
            std::vector Block(B, std::vector<double>(B, 0));
            for (int m = 0; m < B; ++m) {
                for (int n = 0; n < B; ++n) {
                    Block[m][n] = static_cast<double>(X[ib + m][jb + n]);
                }
            }

            auto Qf = std::vector(B, std::vector<double>(B, 1));
            auto Dblock = std::vector(8, std::vector(8, 0.0));

            // Perform DCT
            ComputeQuantizedDCT(Block, Qf, Dblock);

            // Flatten the DCT coefficients and assign them to the D_raw matrix (correcting the indexing)
            int k = 0;
            for (int m = 0; m < B; ++m) {
                for (int n = 0; n < B; ++n) {
                    D_raw[i][j][k] = Dblock[n][m];
                    ++k;
                }
            }
        }
    }
}

/**
 * @brief Converts a spatial 8x8 block into quantized DCT coefficients.
 *
 * Performs a discrete cosine transform (DCT) on a spatial 8x8 block and quantizes the resulting coefficients using the provided quantization matrix.
 *
 * @param[in] Z Input 8x8 spatial block.
 * @param[in] QM Quantization matrix (8x8).
 * @param[out] QD Output 8x8 block of quantized DCT coefficients.
 * @return Quantized DCT coefficients as an 8x8 block.
 */
void JPEGProcessor::ComputeQuantizedDCT(const std::vector<std::vector<double>>& Z, const std::vector<std::vector<double>>& QM,
                         std::vector<std::vector<double>>& QD) {
    std::vector Z_shifted(8, std::vector(8, 0.0));

    // Shift matrix values by subtracting 128 from each element
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            Z_shifted[i][j] = Z[i][j] - 128;
        }
    }

    double out[64];
    double in[64];

    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
            in[8 * i + j] = Z_shifted[i][j];

    PerformSlowFDCT(out, in);

    // Column-first
    std::vector D(8, std::vector(8, 0.0));
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
            D[i][j] = out[8 * i + j];

    // Quantize the DCT coefficients
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            QD[i][j] = D[i][j] / (QM[i][j] * 8);
        }
    }
}

/**
 * @brief Performs the forward discrete cosine transform (FDCT) on a block of spatial data.
 *
 * This function computes the forward discrete cosine transform (FDCT) for an 8x8 block of spatial data using a slow implementation.
 * It takes the spatial domain values (image block) and applies the DCT, storing the resulting frequency domain coefficients.
 *
 * @param[out] out Output array to store the 8x8 DCT coefficients after the transformation.
 *                 The resulting coefficients are stored in row-major order.
 * @param[in] in Input array containing the 8x8 spatial block to be transformed.
 *               These are typically pixel values of a single block from the image.
 *
 * @note The input `in` is expected to contain the spatial domain pixel values in a flattened form (64 elements).
 * @note The function uses a "slow" implementation of DCT, which may be less efficient but provides a simple method for the transformation.
 */
void JPEGProcessor::PerformSlowFDCT(double out[], const double in[]) {
    INT32 tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
    INT32 tmp10, tmp11, tmp12, tmp13;
    INT32 z1, z2, z3, z4, z5;
    DCTELEM *dataptr, *data;
    int ctr, i;

    /* Pass 1: process rows. */
    /* Note results are scaled up by sqrt(8) compared to a true DCT; */
    /* furthermore, we scale the results by 2**PASS1_BITS. */
    data = (DCTELEM *) malloc(DCTSIZE * DCTSIZE * sizeof(DCTELEM));
    for (i = 0; i < DCTSIZE * DCTSIZE; i++) data[i] = (int) in[i];

    dataptr = data;
    for (ctr = DCTSIZE - 1; ctr >= 0; ctr--) {
        tmp0 = dataptr[0] + dataptr[7];
        tmp7 = dataptr[0] - dataptr[7];
        tmp1 = dataptr[1] + dataptr[6];
        tmp6 = dataptr[1] - dataptr[6];
        tmp2 = dataptr[2] + dataptr[5];
        tmp5 = dataptr[2] - dataptr[5];
        tmp3 = dataptr[3] + dataptr[4];
        tmp4 = dataptr[3] - dataptr[4];

        /* Even part per LL&M figure 1 --- note that published figure is faulty;
         * rotator "sqrt(2)*c1" should be "sqrt(2)*c6".
         */

        tmp10 = tmp0 + tmp3;
        tmp13 = tmp0 - tmp3;
        tmp11 = tmp1 + tmp2;
        tmp12 = tmp1 - tmp2;

        dataptr[0] = (DCTELEM) ((tmp10 + tmp11) << PASS1_BITS);
        dataptr[4] = (DCTELEM) ((tmp10 - tmp11) << PASS1_BITS);

        z1 = MULTIPLY(tmp12 + tmp13, FIX_0_541196100);
        dataptr[2] = (DCTELEM) DESCALE(z1 + MULTIPLY(tmp13, FIX_0_765366865),
                                       CONST_BITS-PASS1_BITS);
        dataptr[6] = (DCTELEM) DESCALE(z1 + MULTIPLY(tmp12, - FIX_1_847759065),
                                       CONST_BITS-PASS1_BITS);

        /* Odd part per figure 8 --- note paper omits factor of sqrt(2).
         * cK represents cos(K*pi/16).
         * i0..i3 in the paper are tmp4..tmp7 here.
         */

        z1 = tmp4 + tmp7;
        z2 = tmp5 + tmp6;
        z3 = tmp4 + tmp6;
        z4 = tmp5 + tmp7;
        z5 = MULTIPLY(z3 + z4, FIX_1_175875602); /* sqrt(2) * c3 */

        tmp4 = MULTIPLY(tmp4, FIX_0_298631336); /* sqrt(2) * (-c1+c3+c5-c7) */
        tmp5 = MULTIPLY(tmp5, FIX_2_053119869); /* sqrt(2) * ( c1+c3-c5+c7) */
        tmp6 = MULTIPLY(tmp6, FIX_3_072711026); /* sqrt(2) * ( c1+c3+c5-c7) */
        tmp7 = MULTIPLY(tmp7, FIX_1_501321110); /* sqrt(2) * ( c1+c3-c5-c7) */
        z1 = MULTIPLY(z1, - FIX_0_899976223); /* sqrt(2) * (c7-c3) */
        z2 = MULTIPLY(z2, - FIX_2_562915447); /* sqrt(2) * (-c1-c3) */
        z3 = MULTIPLY(z3, - FIX_1_961570560); /* sqrt(2) * (-c3-c5) */
        z4 = MULTIPLY(z4, - FIX_0_390180644); /* sqrt(2) * (c5-c3) */

        z3 += z5;
        z4 += z5;

        dataptr[7] = (DCTELEM) DESCALE(tmp4 + z1 + z3, CONST_BITS-PASS1_BITS);
        dataptr[5] = (DCTELEM) DESCALE(tmp5 + z2 + z4, CONST_BITS-PASS1_BITS);
        dataptr[3] = (DCTELEM) DESCALE(tmp6 + z2 + z3, CONST_BITS-PASS1_BITS);
        dataptr[1] = (DCTELEM) DESCALE(tmp7 + z1 + z4, CONST_BITS-PASS1_BITS);

        dataptr += DCTSIZE; /* advance pointer to next row */
    }

    /* Pass 2: process columns.
     * We remove the PASS1_BITS scaling, but leave the results scaled up
     * by an overall factor of 8.
     */

    dataptr = data;
    for (ctr = DCTSIZE - 1; ctr >= 0; ctr--) {
        tmp0 = dataptr[DCTSIZE * 0] + dataptr[DCTSIZE * 7];
        tmp7 = dataptr[DCTSIZE * 0] - dataptr[DCTSIZE * 7];
        tmp1 = dataptr[DCTSIZE * 1] + dataptr[DCTSIZE * 6];
        tmp6 = dataptr[DCTSIZE * 1] - dataptr[DCTSIZE * 6];
        tmp2 = dataptr[DCTSIZE * 2] + dataptr[DCTSIZE * 5];
        tmp5 = dataptr[DCTSIZE * 2] - dataptr[DCTSIZE * 5];
        tmp3 = dataptr[DCTSIZE * 3] + dataptr[DCTSIZE * 4];
        tmp4 = dataptr[DCTSIZE * 3] - dataptr[DCTSIZE * 4];

        /* Even part per LL&M figure 1 --- note that published figure is faulty;
         * rotator "sqrt(2)*c1" should be "sqrt(2)*c6".
         */

        tmp10 = tmp0 + tmp3;
        tmp13 = tmp0 - tmp3;
        tmp11 = tmp1 + tmp2;
        tmp12 = tmp1 - tmp2;

        dataptr[DCTSIZE * 0] = (DCTELEM) DESCALE(tmp10 + tmp11, PASS1_BITS);
        dataptr[DCTSIZE * 4] = (DCTELEM) DESCALE(tmp10 - tmp11, PASS1_BITS);

        z1 = MULTIPLY(tmp12 + tmp13, FIX_0_541196100);
        dataptr[DCTSIZE * 2] = (DCTELEM) DESCALE(z1 + MULTIPLY(tmp13, FIX_0_765366865),
                                                 CONST_BITS+PASS1_BITS);
        dataptr[DCTSIZE * 6] = (DCTELEM) DESCALE(z1 + MULTIPLY(tmp12, - FIX_1_847759065),
                                                 CONST_BITS+PASS1_BITS);

        /* Odd part per figure 8 --- note paper omits factor of sqrt(2).
         * cK represents cos(K*pi/16).
         * i0..i3 in the paper are tmp4..tmp7 here.
         */

        z1 = tmp4 + tmp7;
        z2 = tmp5 + tmp6;
        z3 = tmp4 + tmp6;
        z4 = tmp5 + tmp7;
        z5 = MULTIPLY(z3 + z4, FIX_1_175875602); /* sqrt(2) * c3 */

        tmp4 = MULTIPLY(tmp4, FIX_0_298631336); /* sqrt(2) * (-c1+c3+c5-c7) */
        tmp5 = MULTIPLY(tmp5, FIX_2_053119869); /* sqrt(2) * ( c1+c3-c5+c7) */
        tmp6 = MULTIPLY(tmp6, FIX_3_072711026); /* sqrt(2) * ( c1+c3+c5-c7) */
        tmp7 = MULTIPLY(tmp7, FIX_1_501321110); /* sqrt(2) * ( c1+c3-c5-c7) */
        z1 = MULTIPLY(z1, - FIX_0_899976223); /* sqrt(2) * (c7-c3) */
        z2 = MULTIPLY(z2, - FIX_2_562915447); /* sqrt(2) * (-c1-c3) */
        z3 = MULTIPLY(z3, - FIX_1_961570560); /* sqrt(2) * (-c3-c5) */
        z4 = MULTIPLY(z4, - FIX_0_390180644); /* sqrt(2) * (c5-c3) */

        z3 += z5;
        z4 += z5;

        dataptr[DCTSIZE * 7] = (DCTELEM) DESCALE(tmp4 + z1 + z3,
                                                 CONST_BITS+PASS1_BITS);
        dataptr[DCTSIZE * 5] = (DCTELEM) DESCALE(tmp5 + z2 + z4,
                                                 CONST_BITS+PASS1_BITS);
        dataptr[DCTSIZE * 3] = (DCTELEM) DESCALE(tmp6 + z2 + z3,
                                                 CONST_BITS+PASS1_BITS);
        dataptr[DCTSIZE * 1] = (DCTELEM) DESCALE(tmp7 + z1 + z4,
                                                 CONST_BITS+PASS1_BITS);

        dataptr++; /* advance pointer to next column */
    }
    for (i = 0; i < DCTSIZE * DCTSIZE; i++) out[i] = (double) data[i];
    free(data);
}

/**
 * @brief Decompresses a DCT-transformed image back into the spatial domain.
 *
 * This function takes quantized DCT coefficients organized into 8x8 blocks (`D`), applies
 * inverse quantization and inverse discrete cosine transform (IDCT) using the quantization
 * matrix (`Q`), and reconstructs the decompressed image (`X`) in the spatial domain.
 *
 * @param[in] D 3D vector containing quantized DCT coefficients.
 *              - Dimensions: [MD][ND][64], where MD and ND are the number of vertical and horizontal blocks.
 *              - Each block contains 64 coefficients in a flattened 8x8 structure, arranged column-major.
 * @param[in] QM 2D vector representing the quantization matrix (8x8).
 * @param[out] X 2D vector to store the decompressed image in the spatial domain.
 *               - Dimensions: [MD*8][ND*8], where each block contributes an 8x8 region.
 *
 * @note This function assumes the DCT coefficients and quantization matrices are valid.
 * @note Uses the `jpeg_idct_islow` function to perform IDCT with input quantized coefficients and quantization steps.
 */
void JPEGProcessor::DecompressImage(const std::vector<std::vector<std::vector<int>>>& D, const std::vector<std::vector<int>>& QM,
                     std::vector<std::vector<double>>& X) {
    const int MD = (int)D.size(); // Number of 8x8 blocks in the vertical direction
    const int ND = (int)D[0].size(); // Number of 8x8 blocks in the horizontal direction

    // Temporary block to store the spatial representation of a DCT block
    std::vector Dblock(8, std::vector<int>(8));

    // Loop over each block
    for (int i = 0; i < MD; i++) {
        for (int j = 0; j < ND; j++) {
            int B = 8;
            // Copy the DCT coefficients of the (i, j)-th block into Dblock
            for (int k = 0; k < B; k++) {
                for (int l = 0; l < B; l++) {
                    Dblock[k][l] = D[i][j][k * B + l]; // Convert flat 64 to 8x8 matrix
                }
            }

            // Calculate the starting pixel positions for the block in the decompressed image
            int ib = i * B;
            int jb = j * B;

            std::vector<double> out = std::vector<double>(64, 0);
            std::vector<double> quant = std::vector<double>(64, 0);
            std::vector<double> coef_block_linear = std::vector<double>(64, 0);

            int counter = 0;
            for (size_t pp = 0; pp < Dblock.size(); pp++) {
                for (size_t pq = 0; pq < Dblock[pp].size(); pq++) {
                    coef_block_linear[counter] = Dblock[pq][pp];
                    counter++;
                }
            }

            counter = 0;
            for (const auto & k : QM) {
                for (const int l : k) {
                    quant[counter] = l;
                    counter++;
                }
            }

            // All are row-first matrices
            double out_array[64];
            double quant_array[64];
            double coefs_array[64];

            for (int k = 0; k < 64; ++k) {
                out_array[k] = out[k];
                quant_array[k] = quant[k];
                coefs_array[k] = coef_block_linear[k];
            }

            PerformSlowIDCT(out_array, coefs_array, quant_array);

            // Place the spatial block into the decompressed image X
            counter = 0;
            for (int k = 0; k < B; k++) {
                for (int l = 0; l < B; l++) {
                    if (ib + k < (int)X.size() && jb + l < (int)X[0].size()) {
                        X[ib + k][jb + l] = out_array[counter];
                    }
                    counter++;
                }
            }
        }
    }
}

/**
 * @brief Converts a 3D DCT coefficient structure into a 2D image plane.
 *
 * Reconstructs a 2D matrix from a 3D structure where blocks are organized column-major.
 * Each block contributes an 8x8 region in the output 2D matrix.
 *
 * @param[in] cube 3D vector of DCT coefficients.
 * @param[in] MB Number of block rows.
 * @param[in] NB Number of block columns.
 * @param[out] plane 2D matrix to store the reconstructed image.
 */
void JPEGProcessor::VecToPlane(const std::vector<std::vector<std::vector<double>>>& cube, const size_t MB, const size_t NB,
                std::vector<std::vector<double>>& plane) {
    // Define the dimensions for rows and columns based on Cube size
    const size_t M = NB * 8; // Count of columns in Plane
    const size_t N = MB * 8; // Count of rows in Plane

    // Resize the Plane vector to match the required dimensions
    plane.resize(N, std::vector(M, 0.0));

    // Fill the Plane array with data from Cube in 8x8 blocks
    for (size_t i = 0; i < MB; ++i) {
        // Rows of 8x8 blocks
        for (size_t j = 0; j < NB; ++j) {
            // Columns of 8x8 blocks
            int idx = 0;
            // Copy values from Cube(i,j,:) into an 8x8 block in Plane
            for (int n = 0; n < 8; ++n) {
                // Rows within 8x8 block
                for (int m = 0; m < 8; ++m) {
                    // Columns within 8x8 block
                    plane[i * 8 + m][j * 8 + n] = cube[i][j][idx++];
                }
            }
        }
    }
}

/**
 * @brief Converts a 2D matrix into a 3D DCT coefficient structure.
 *
 * Reshapes a 2D image plane into a 3D vector structure where the first two dimensions represent
 * the block indices, and the third dimension stores DCT coefficients in column-major order.
 *
 * @param[in] plane 2D matrix representing the image.
 * @param[in] Y Height of the image in pixels.
 * @param[in] X Width of the image in pixels.
 * @param[out] D 3D vector to store the reshaped blocks of DCT coefficients.
 */
void JPEGProcessor::PlaneToVec(const std::vector<std::vector<double>>& plane, const size_t Y, const size_t X, std::vector<std::vector<std::vector<int>>>& D) {
    const size_t M = ceil((double)X / 8); // Count of columns
    const size_t N = ceil((double)Y / 8); // Count of rows

    // Resize the D1 vector to hold the reshaped blocks
    D.resize(N, std::vector(M, std::vector(64, 0)));

    // Fill the D1 array with reshaped blocks from the plane in column-major order
    for (size_t i = 0; i < N; ++i) {
        // Rows of blocks
        for (size_t j = 0; j < M; ++j) {
            // Columns of blocks
            int idx = 0;
            // Iterate in column-major order for an 8x8 block
            for (int n = 0; n < 8; ++n) {
                // Columns inside block (MATLAB-style: first move column)
                for (int m = 0; m < 8; ++m) {
                    // Rows inside block
                    if (i * 8 + m < plane.size() && j * 8 + n < plane[0].size()) {
                        D[i][j][idx++] = static_cast<int>(plane[i * 8 + m][j * 8 + n]);
                    }
                }
            }
        }
    }
}

/**
 * @brief Saves a perturbed (possibly stego) grayscale image as a JPEG file using libjpeg-turbo.
 *
 * This function overwrites the DCT coefficient arrays of a JPEG image with new values
 * from the given spatial-domain plane (usually containing embedded data) and writes
 * the result to a JPEG file with the original image's structure and metadata.
 *
 * @param[in] file_path Target path to save the new JPEG file.
 * @param[in] temp_file Pointer to a JPEGFile object representing the base structure for output.
 *                      Used for copying headers and quantization tables.
 * @param[in] plane 2D matrix representing the modified spatial-domain grayscale image.
 *
 * @throws std::runtime_error if the file cannot be opened or saved.
 *
 * @note This function assumes the image is grayscale and does not modify chroma components.
 */
void JPEGProcessor::SavePerturbedJPEG(const std::string& file_path, const JPEGFile* temp_file, std::vector<std::vector<double>>& plane) {
    // 1. Open original and get coefficient arrays
    jpeg_decompress_struct cinfo = temp_file->getCinfo();
    jvirt_barray_ptr* coef_arrays = jpeg_read_coefficients(&cinfo);

    // 2. Set up compressor for output JPEG
    jpeg_compress_struct cinfo_out{};
    jpeg_error_mgr jerr_out{};
    cinfo_out.err = jpeg_std_error(&jerr_out);
    jpeg_create_compress(&cinfo_out);

    FILE* outfile = fopen(file_path.c_str(), "wb");
    if (!outfile) {
        jpeg_destroy_compress(&cinfo_out);
        throw std::runtime_error("Could not create final image file");
    }
    jpeg_stdio_dest(&cinfo_out, outfile);

    // 3. Copy all critical parameters (sampling, quant, etc.) from original
    jpeg_copy_critical_parameters(&cinfo, &cinfo_out);

    // 4. Overwrite Y DCT blocks, leave CbCr untouched
    for (int ci = 0; ci < cinfo.num_components; ci++) {
        jpeg_component_info* compptr = &cinfo.comp_info[ci];
        int block_rows = compptr->height_in_blocks;
        int block_cols = compptr->width_in_blocks;

        for (int blk_y = 0; blk_y < block_rows; blk_y++) {
            // TRUE: writeable buffer for output, FALSE: read-only for input
            JBLOCKARRAY buffer = (cinfo.mem->access_virt_barray)
                (reinterpret_cast<j_common_ptr>(&cinfo), coef_arrays[ci], blk_y, 1, TRUE);

            for (int blk_x = 0; blk_x < block_cols; blk_x++) {
                if (ci == 0) { // Y channel: overwrite with plane data
                    for (int i = 0; i < DCTSIZE; i++)
                        for (int j = 0; j < DCTSIZE; j++) {
                            int row = blk_y * DCTSIZE + i;
                            int col = blk_x * DCTSIZE + j;
                            // Bounds check (may be needed for non-multiple-of-8 images)
                            if (row < static_cast<int>(plane.size()) && col < static_cast<int>(plane[0].size())) {
                                buffer[0][blk_x][i * DCTSIZE + j] = static_cast<JCOEF>(plane[row][col]);
                            }
                        }
                }
            }
        }
    }

    // 5. Write the modified coefficients out
    jpeg_write_coefficients(&cinfo_out, coef_arrays);

    jpeg_finish_compress(&cinfo_out);
    fclose(outfile);
    jpeg_destroy_compress(&cinfo_out);
}

/**
 * @brief Saves a 2D image matrix as a JPEG file.
 *
 * Compresses and writes a 2D image matrix into a JPEG file with a specified quality.
 *
 * @param[in] filename Path to the output JPEG file.
 * @param[in] original_file Pointer to a JPEGFile object representing the base structure for output.
 *                          Used for copying image size and coefficients.
 * @param[in] quality JPEG compression quality (1-100).
 */
void JPEGProcessor::SaveCoverJPEG(const char* filename, const JPEGFile* original_file, int quality) {
    // 1) DECOMPRESS the original JPEG into raw YCbCr
    jpeg_decompress_struct din{};
    jpeg_error_mgr         djerr{};
    din.err = jpeg_std_error(&djerr);
    jpeg_create_decompress(&din);

    FILE* orig_fp = fopen(original_file->getFilePath().c_str(), "rb");
    if (!orig_fp) {
        jpeg_destroy_decompress(&din);
        throw std::runtime_error("Can't reopen source JPEG");
    }
    jpeg_stdio_src(&din, orig_fp);

    jpeg_read_header(&din, TRUE);
    din.out_color_space = JCS_YCbCr;
    jpeg_start_decompress(&din);

    const int W          = din.output_width;
    const int H          = din.output_height;
    const int comps      = din.output_components;   // should be 3
    const int row_stride = W * comps;

    std::vector<JSAMPLE> rawbuf(H * row_stride);
    JSAMPROW             rowptr[1];
    for (int y = 0; y < H; ++y) {
        rowptr[0] = &rawbuf[y * row_stride];
        jpeg_read_scanlines(&din, rowptr, 1);
    }

    jpeg_finish_decompress(&din);
    fclose(orig_fp);

    // 2) PATCH only the Y channel
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            int idx = y * row_stride + x * 3;
            double v = original_file->getX()[y][x];
            v = std::min(std::max(v, 0.0), 255.0);
            rawbuf[idx + 0] = static_cast<JSAMPLE>(v);
        }
    }

    // 3) RE-COMPRESS as YCbCr, preserving all original parameters
    jpeg_compress_struct cout{};
    jpeg_error_mgr       cjerr{};
    cout.err = jpeg_std_error(&cjerr);
    jpeg_create_compress(&cout);

    FILE* out_fp = fopen(filename, "wb");
    if (!out_fp) {
        jpeg_destroy_compress(&cout);
        jpeg_destroy_decompress(&din);
        throw std::runtime_error("Could not create cover image file");
    }
    jpeg_stdio_dest(&cout, out_fp);

    // Copy the original's sampling factors, quant tables, markers, etc.
    jpeg_copy_critical_parameters(&din, &cout);

    // Set input color space and components
    cout.input_components = comps;
    cout.in_color_space   = JCS_YCbCr;

    // Rescale quantization tables to the desired quality
    jpeg_set_quality(&cout, quality, TRUE);

    jpeg_start_compress(&cout, TRUE);

    // Write back each scanline of {Y(patched), Cb, Cr}
    for (int y = 0; y < H; ++y) {
        rowptr[0] = &rawbuf[y * row_stride];
        jpeg_write_scanlines(&cout, rowptr, 1);
    }

    jpeg_finish_compress(&cout);
    fclose(out_fp);

    // CLEAN UP
    jpeg_destroy_compress(&cout);
    jpeg_destroy_decompress(&din);
}

/**
 * @brief Generates a unique temporary filename with a given prefix and extension.
 *
 * Creates a filename in the system's temporary directory with a "PQ" subfolder,
 * combining a timestamp and a random number to ensure uniqueness.
 *
 * @param[in] prefix Prefix string to use in the filename.
 * @param[in] extension File extension (e.g., "jpg", "tmp").
 * @return Full path to the generated temporary file.
 *
 * @note Ensures the directory `<temp>/PQ/` exists before generating the path.
 */
std::string JPEGProcessor::GetTempFilename(const std::string& prefix, const std::string& extension) {
    // Get temporary directory path
    std::filesystem::path temp_dir = std::filesystem::temp_directory_path() / "PQ";
    std::filesystem::create_directories(temp_dir);  // Create if doesn't exist

    // Generate unique filename using timestamp and random number
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()
    ).count();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 9999);

    return (temp_dir / 
           std::filesystem::path(prefix +
                   "_" + 
                   std::to_string(timestamp) + 
                   "_" + 
                   std::to_string(dis(gen)) + 
                   "." + extension))
           .string();
}

/**
 * @brief Computes the quantization matrix for a given JPEG quality factor.
 *
 * This function generates an 8x8 quantization matrix `Q` based on the specified quality factor.
 * It scales the default 50% quality quantization matrix (`Q50`) proportionally to the quality value,
 * where higher quality results in lower quantization values (less compression).
 *
 * @param[in] quality JPEG quality factor (range: 1-100).
 *                    - Values closer to 1 result in higher compression and lower image quality.
 *                    - Values closer to 100 result in lower compression and higher image quality.
 * @param[out] QM 8x8 quantization matrix to store the computed values.
 *               - Higher values increase compression by reducing precision in the frequency domain.
 *               - Lower values preserve more detail at the cost of higher file size.
 *
 * @note The input quality value is clamped to the range [1, 100] if it exceeds the bounds.
 * @note The generated matrix is based on scaling the default JPEG 50% quality matrix.
 */
void JPEGProcessor::ComputeQmatrix(int quality, std::vector<std::vector<int>>& QM) {
    // Default 50% quality quantization matrix
    const std::vector<std::vector<int> > Q50 = {
        {16, 11, 10, 16, 24, 40, 51, 61},
        {12, 12, 14, 19, 26, 58, 60, 55},
        {14, 13, 16, 24, 40, 57, 69, 56},
        {14, 17, 22, 29, 51, 87, 80, 62},
        {18, 22, 37, 56, 68, 109, 103, 77},
        {24, 35, 55, 64, 81, 104, 113, 92},
        {49, 64, 78, 87, 103, 121, 120, 101},
        {72, 92, 95, 98, 112, 100, 103, 99}
    };

    // Ensure quality is between 1 and 100
    if (quality < 1) quality = 1;
    if (quality > 100) quality = 100;

    // Compute quantization matrix based on quality
    if (quality >= 50) {
        // High quality case (quality >= 50)
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                QM[i][j] = (int)fmax(1, round(2 * Q50[i][j] * (1 - quality / 100.0)));
            }
        }
    } else {
        // Low quality case (quality < 50)
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                QM[i][j] = (int)fmin(255, round(Q50[i][j] * 50.0 / quality));
            }
        }
    }
}

/**
 * @brief Clamps an integer value to the JPEG valid range [0, 255].
 *
 * Adds 128 to the input and ensures the result stays within the range
 * of 0 (black) to 255 (white), suitable for grayscale JPEG pixel output.
 *
 * @param[in] x Input value.
 * @return Clamped output in the range [0, 255].
 */
int JPEGProcessor::Range_limit(int x) {
    x = x + 128;
    if (x < 0) return 0;
    if (x > 255) return 255;
    return x;
}

/**
 * @brief Performs the inverse discrete cosine transform (IDCT) on a block of DCT coefficients.
 *
 * This function computes the inverse DCT for an 8x8 block of DCT coefficients using a slow implementation.
 * It takes the quantized DCT coefficients, applies the inverse DCT transformation, and stores the resulting
 * spatial domain values in the output array.
 *
 * @param[out] out Output array to store the 8x8 spatial block after applying the inverse DCT.
 *                 The resulting values are stored in row-major order.
 * @param[in] coef_block Input array containing the quantized DCT coefficients for the block.
 *                      These are typically 64 DCT coefficients from a single 8x8 block.
 * @param[in] qt Quantization matrix used to scale the DCT coefficients back to their original values before IDCT.
 *              This matrix is used for the inverse quantization process before applying the IDCT.
 *
 * @note The input `coef_block` is expected to contain the quantized DCT coefficients in a flattened form (64 elements).
 * @note The function uses a "slow" implementation of IDCT, which may be less efficient but provides a simple method for inverse transformation.
 */
void JPEGProcessor::PerformSlowIDCT(double out[], const double coef_block[], const double qt[]) {
    INT32 tmp0, tmp1, tmp2, tmp3;
    INT32 tmp10, tmp11, tmp12, tmp13;
    INT32 z1, z2, z3, z4, z5;
    short *inptr, *intp;
    int *quantptr, *qtp;
    int *wsptr;
    short *outptr, *outbuf;
    int ctr, i;
    int workspace[DCTSIZE2];


    /* Pass 1: process columns from input, store into work array. */
    /* Note results are scaled up by sqrt(8) compared to a true IDCT; */
    /* furthermore, we scale the results by 2**PASS1_BITS. */

    /*inptr = coef_block;
    quantptr = (ISLOW_MULT_TYPE *) compptr->dct_table;*/
    inptr = (short *) malloc(DCTSIZE2 * sizeof(short));
    quantptr = (int *) malloc(DCTSIZE2 * sizeof(int));
    intp = inptr;
    qtp = quantptr;
    for (i = 0; i < DCTSIZE2; i++) inptr[i] = (short) coef_block[i];
    for (i = 0; i < DCTSIZE2; i++) quantptr[i] = (int) qt[i];

    wsptr = workspace;
    for (ctr = DCTSIZE; ctr > 0; ctr--) {
        /* Due to quantization, we will usually find that many of the input
         * coefficients are zero, especially the AC terms.  We can exploit this
         * by short-circuiting the IDCT calculation for any column in which all
         * the AC terms are zero.  In that case each output is equal to the
         * DC coefficient (with scale factor as needed).
         * With typical images and quantization tables, half or more of the
         * column DCT calculations can be simplified this way.
         */

        if (inptr[DCTSIZE * 1] == 0 && inptr[DCTSIZE * 2] == 0 &&
            inptr[DCTSIZE * 3] == 0 && inptr[DCTSIZE * 4] == 0 &&
            inptr[DCTSIZE * 5] == 0 && inptr[DCTSIZE * 6] == 0 &&
            inptr[DCTSIZE * 7] == 0) {
            /* AC terms all zero */
            int dcval = DEQUANTIZE(inptr[DCTSIZE*0], quantptr[DCTSIZE*0]) << PASS1_BITS;

            wsptr[DCTSIZE * 0] = dcval;
            wsptr[DCTSIZE * 1] = dcval;
            wsptr[DCTSIZE * 2] = dcval;
            wsptr[DCTSIZE * 3] = dcval;
            wsptr[DCTSIZE * 4] = dcval;
            wsptr[DCTSIZE * 5] = dcval;
            wsptr[DCTSIZE * 6] = dcval;
            wsptr[DCTSIZE * 7] = dcval;

            inptr++; /* advance pointers to next column */
            quantptr++;
            wsptr++;
            continue;
        }

        /* Even part: reverse the even part of the forward DCT. */
        /* The rotator is sqrt(2)*c(-6). */

        z2 = DEQUANTIZE(inptr[DCTSIZE*2], quantptr[DCTSIZE*2]);
        z3 = DEQUANTIZE(inptr[DCTSIZE*6], quantptr[DCTSIZE*6]);

        z1 = MULTIPLY(z2 + z3, FIX_0_541196100);
        tmp2 = z1 + MULTIPLY(z3, - FIX_1_847759065);
        tmp3 = z1 + MULTIPLY(z2, FIX_0_765366865);

        z2 = DEQUANTIZE(inptr[DCTSIZE*0], quantptr[DCTSIZE*0]);
        z3 = DEQUANTIZE(inptr[DCTSIZE*4], quantptr[DCTSIZE*4]);

        tmp0 = (z2 + z3) << CONST_BITS;
        tmp1 = (z2 - z3) << CONST_BITS;

        tmp10 = tmp0 + tmp3;
        tmp13 = tmp0 - tmp3;
        tmp11 = tmp1 + tmp2;
        tmp12 = tmp1 - tmp2;

        /* Odd part per figure 8; the matrix is unitary and hence its
         * transpose is its inverse.  i0  i3 are y7,y5,y3,y1 respectively.
         */

        tmp0 = DEQUANTIZE(inptr[DCTSIZE*7], quantptr[DCTSIZE*7]);
        tmp1 = DEQUANTIZE(inptr[DCTSIZE*5], quantptr[DCTSIZE*5]);
        tmp2 = DEQUANTIZE(inptr[DCTSIZE*3], quantptr[DCTSIZE*3]);
        tmp3 = DEQUANTIZE(inptr[DCTSIZE*1], quantptr[DCTSIZE*1]);

        z1 = tmp0 + tmp3;
        z2 = tmp1 + tmp2;
        z3 = tmp0 + tmp2;
        z4 = tmp1 + tmp3;
        z5 = MULTIPLY(z3 + z4, FIX_1_175875602); /* sqrt(2) * c3 */

        tmp0 = MULTIPLY(tmp0, FIX_0_298631336); /* sqrt(2) * (-c1+c3+c5-c7) */
        tmp1 = MULTIPLY(tmp1, FIX_2_053119869); /* sqrt(2) * ( c1+c3-c5+c7) */
        tmp2 = MULTIPLY(tmp2, FIX_3_072711026); /* sqrt(2) * ( c1+c3+c5-c7) */
        tmp3 = MULTIPLY(tmp3, FIX_1_501321110); /* sqrt(2) * ( c1+c3-c5-c7) */
        z1 = MULTIPLY(z1, - FIX_0_899976223); /* sqrt(2) * (c7-c3) */
        z2 = MULTIPLY(z2, - FIX_2_562915447); /* sqrt(2) * (-c1-c3) */
        z3 = MULTIPLY(z3, - FIX_1_961570560); /* sqrt(2) * (-c3-c5) */
        z4 = MULTIPLY(z4, - FIX_0_390180644); /* sqrt(2) * (c5-c3) */

        z3 += z5;
        z4 += z5;

        tmp0 += z1 + z3;
        tmp1 += z2 + z4;
        tmp2 += z2 + z3;
        tmp3 += z1 + z4;

        /* Final output stage: inputs are tmp10..tmp13, tmp0..tmp3 */

        wsptr[DCTSIZE * 0] = (int) DESCALE(tmp10 + tmp3, CONST_BITS-PASS1_BITS);
        wsptr[DCTSIZE * 7] = (int) DESCALE(tmp10 - tmp3, CONST_BITS-PASS1_BITS);
        wsptr[DCTSIZE * 1] = (int) DESCALE(tmp11 + tmp2, CONST_BITS-PASS1_BITS);
        wsptr[DCTSIZE * 6] = (int) DESCALE(tmp11 - tmp2, CONST_BITS-PASS1_BITS);
        wsptr[DCTSIZE * 2] = (int) DESCALE(tmp12 + tmp1, CONST_BITS-PASS1_BITS);
        wsptr[DCTSIZE * 5] = (int) DESCALE(tmp12 - tmp1, CONST_BITS-PASS1_BITS);
        wsptr[DCTSIZE * 3] = (int) DESCALE(tmp13 + tmp0, CONST_BITS-PASS1_BITS);
        wsptr[DCTSIZE * 4] = (int) DESCALE(tmp13 - tmp0, CONST_BITS-PASS1_BITS);

        inptr++; /* advance pointers to next column */
        quantptr++;
        wsptr++;
    }

    /* Pass 2: process rows from work array, store into output array. */
    /* Note that we must descale the results by a factor of 8 == 2**3, */
    /* and also undo the PASS1_BITS scaling. */
    outptr = (short *) malloc(DCTSIZE * sizeof(short));
    outbuf = (short *) malloc(DCTSIZE2 * sizeof(short));
    wsptr = workspace;
    for (ctr = 0; ctr < DCTSIZE; ctr++) {
        /*outptr = output_buf[ctr] + output_col;*/
        /* Rows of zeroes can be exploited in the same way as we did with columns.
         * However, the column calculation has created many nonzero AC terms, so
         * the simplification applies less often (typically 5% to 10% of the time).
         * On machines with very fast multiplication, it's possible that the
         * test takes more time than it's worth.  In that case this section
         * may be commented out.
         */

#ifndef NO_ZERO_ROW_TEST
        if (wsptr[1] == 0 && wsptr[2] == 0 && wsptr[3] == 0 && wsptr[4] == 0 &&
            wsptr[5] == 0 && wsptr[6] == 0 && wsptr[7] == 0) {
            /* AC terms all zero */
            JSAMPLE dcval = Range_limit((int) DESCALE((INT32) wsptr[0], PASS1_BITS+3)
            );

            outptr[0] = dcval;
            outptr[1] = dcval;
            outptr[2] = dcval;
            outptr[3] = dcval;
            outptr[4] = dcval;
            outptr[5] = dcval;
            outptr[6] = dcval;
            outptr[7] = dcval;
            memcpy(&outbuf[ctr * DCTSIZE], outptr, DCTSIZE * sizeof(short));
            wsptr += DCTSIZE; /* advance pointer to next row */
            continue;
        }
#endif

        /* Even part: reverse the even part of the forward DCT. */
        /* The rotator is sqrt(2)*c(-6). */

        z2 = (INT32) wsptr[2];
        z3 = (INT32) wsptr[6];

        z1 = MULTIPLY(z2 + z3, FIX_0_541196100);
        tmp2 = z1 + MULTIPLY(z3, - FIX_1_847759065);
        tmp3 = z1 + MULTIPLY(z2, FIX_0_765366865);

        tmp0 = ((INT32) wsptr[0] + (INT32) wsptr[4]) << CONST_BITS;
        tmp1 = ((INT32) wsptr[0] - (INT32) wsptr[4]) << CONST_BITS;

        tmp10 = tmp0 + tmp3;
        tmp13 = tmp0 - tmp3;
        tmp11 = tmp1 + tmp2;
        tmp12 = tmp1 - tmp2;

        /* Odd part per figure 8; the matrix is unitary and hence its
         * transpose is its inverse.  i0..i3 are y7,y5,y3,y1 respectively.
         */

        tmp0 = (INT32) wsptr[7];
        tmp1 = (INT32) wsptr[5];
        tmp2 = (INT32) wsptr[3];
        tmp3 = (INT32) wsptr[1];

        z1 = tmp0 + tmp3;
        z2 = tmp1 + tmp2;
        z3 = tmp0 + tmp2;
        z4 = tmp1 + tmp3;
        z5 = MULTIPLY(z3 + z4, FIX_1_175875602); /* sqrt(2) * c3 */

        tmp0 = MULTIPLY(tmp0, FIX_0_298631336); /* sqrt(2) * (-c1+c3+c5-c7) */
        tmp1 = MULTIPLY(tmp1, FIX_2_053119869); /* sqrt(2) * ( c1+c3-c5+c7) */
        tmp2 = MULTIPLY(tmp2, FIX_3_072711026); /* sqrt(2) * ( c1+c3+c5-c7) */
        tmp3 = MULTIPLY(tmp3, FIX_1_501321110); /* sqrt(2) * ( c1+c3-c5-c7) */
        z1 = MULTIPLY(z1, - FIX_0_899976223); /* sqrt(2) * (c7-c3) */
        z2 = MULTIPLY(z2, - FIX_2_562915447); /* sqrt(2) * (-c1-c3) */
        z3 = MULTIPLY(z3, - FIX_1_961570560); /* sqrt(2) * (-c3-c5) */
        z4 = MULTIPLY(z4, - FIX_0_390180644); /* sqrt(2) * (c5-c3) */

        z3 += z5;
        z4 += z5;

        tmp0 += z1 + z3;
        tmp1 += z2 + z4;
        tmp2 += z2 + z3;
        tmp3 += z1 + z4;

        /* Final output stage: inputs are tmp10 tmp13, tmp0 tmp3 */

        outptr[0] = (short)Range_limit((int) DESCALE(tmp10 + tmp3,
                                              CONST_BITS+PASS1_BITS+3)
        );
        outptr[7] = (short)Range_limit((int) DESCALE(tmp10 - tmp3,
                                              CONST_BITS+PASS1_BITS+3)
        );
        outptr[1] = (short)Range_limit((int) DESCALE(tmp11 + tmp2,
                                              CONST_BITS+PASS1_BITS+3)
        );
        outptr[6] = (short)Range_limit((int) DESCALE(tmp11 - tmp2,
                                              CONST_BITS+PASS1_BITS+3)
        );
        outptr[2] = (short)Range_limit((int) DESCALE(tmp12 + tmp1,
                                              CONST_BITS+PASS1_BITS+3)
        );
        outptr[5] = (short)Range_limit((int) DESCALE(tmp12 - tmp1,
                                              CONST_BITS+PASS1_BITS+3)
        );
        outptr[3] = (short)Range_limit((int) DESCALE(tmp13 + tmp0,
                                              CONST_BITS+PASS1_BITS+3)
        );
        outptr[4] = (short)Range_limit((int) DESCALE(tmp13 - tmp0,
                                              CONST_BITS+PASS1_BITS+3)
        );
        memcpy(&outbuf[ctr * DCTSIZE], outptr, DCTSIZE * sizeof(short));
        wsptr += DCTSIZE; /* advance pointer to next row */
    }
    for (i = 0; i < DCTSIZE2; i++) out[i] = (double) outbuf[i];
    free(intp);
    free(qtp);
    free(outptr);
    free(outbuf);
}