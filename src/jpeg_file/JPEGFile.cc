#include "JPEGFile.h"
#include <jpeglib.h>
#include <csetjmp>
#include <cstdio>
#include <stdexcept>

#include "../jpeg_processor/JPEGProcessor.h"

/**
 * @brief Constructs a JPEGFile object by reading a JPEG image and preparing its DCT data.
 *
 * This constructor opens the specified JPEG file, reads its header and DCT coefficients,
 * extracts the quantization matrix, estimates the image quality, and prepares both the
 * frequency (DCT) and spatial (pixel) representations of the image.
 *
 * @param[in] filePath Path to the input JPEG image file.
 *
 * @throws std::runtime_error if the file cannot be opened.
 *
 * @note The quantization matrix is modified internally (DC terms may be disabled manually if needed).
 */
JPEGFile::JPEGFile(const std::string& filePath)
    : filePath_(filePath),
    cinfo_() {

    FILE *infile;
    if ((infile = fopen(filePath.c_str(), "rb")) == nullptr) {
        throw std::runtime_error("Can't open JPEG file");
    }

    ReadJPEGInfo(infile);

    height_ = cinfo_.image_height;
    width_ = cinfo_.image_width;

    QM_= std::vector(8, std::vector<int>(8));
    ExtractQmatrix();

    quality_ = JPEGProcessor::EstimateQuality(this);

    JPEGProcessor::PlaneToVec(dct_coefficients_, height_, width_, D_);

    X_ = std::vector(height_, std::vector(width_, 0.0));
    JPEGProcessor::DecompressImage(D_, QM_, X_);

    fclose(infile);
}

// Destructor implementation
JPEGFile::~JPEGFile() {
    jpeg_destroy_decompress(&cinfo_);
}

/**
 * @brief Reads JPEG file information and extracts its DCT coefficients.
 *
 * This function initializes a JPEG decompression structure, reads the header information
 * of the input JPEG file, and extracts its quantized DCT coefficients into a 2D matrix.
 *
 * @param[in] infile Pointer to the opened JPEG file to be read.
 *
 * @note This function assumes the input file is a valid JPEG image.
 * @note Calls `ExtractDCTCoefficients` to process the coefficient arrays.
 */
void JPEGFile::ReadJPEGInfo(FILE* infile) {
    jpeg_error_mgr jerr{};

    cinfo_.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo_);
    jpeg_stdio_src(&cinfo_, infile);
    jpeg_read_header(&cinfo_, TRUE);

    ExtractDCTCoefficients();
}

/**
 * @brief Extracts DCT coefficients from a JPEG image.
 *
 * This function reads the DCT coefficients from a JPEG image file using libjpeg.
 * The coefficients are organized into a 2D matrix representing the entire image.
 *
 */
void JPEGFile::ExtractDCTCoefficients() {
    // Read DCT coefficient arrays
    jvirt_barray_ptr *coef_arrays = jpeg_read_coefficients(&cinfo_);

    // Determine the total image dimensions (height, width) to initialize the 2D matrix
    const size_t total_height = cinfo_.image_height;
    const size_t total_width = cinfo_.image_width;

    // Resize the dct_coefficients to hold the full image (total_height x total_width)
    dct_coefficients_.resize(total_height, std::vector(total_width, 0.0));

    // Loop over each component (Y, Cb, Cr)
    size_t current_height_offset = 0; // Keep track of vertical offset for each component
    for (int ci = 0; ci < cinfo_.num_components; ci++) {
        jpeg_component_info *compptr = &cinfo_.comp_info[ci];

        // Determine the height and width of the DCT block grid for the current component
        auto block_rows = compptr->height_in_blocks;
        auto block_cols = compptr->width_in_blocks;
        auto c_height = block_rows * DCTSIZE; // Full height in pixels

        // Loop through each block row (y direction)
        for (unsigned int blk_y = 0; blk_y < block_rows; blk_y++) {
            // Access the current row of blocks from the virtual array
            const JBLOCKARRAY buffer = (cinfo_.mem->access_virt_barray)(reinterpret_cast<j_common_ptr>(&cinfo_),
                                                                        coef_arrays[ci], blk_y, 1, FALSE);

            // Loop through each block column (x direction)
            for (unsigned int blk_x = 0; blk_x < block_cols; blk_x++) {
                // Get the current DCT block (8x8)
                const JBLOCK &block = buffer[0][blk_x];

                // Copy DCT coefficients from the block to the correct position in the 2D matrix
                for (int i = 0; i < DCTSIZE; i++) {
                    // For each row in the block
                    for (int j = 0; j < DCTSIZE; j++) {
                        // For each column in the block
                        // Place the coefficient into the correct spot in the 2D matrix
                        const size_t row = blk_y * DCTSIZE + i + current_height_offset;
                        const size_t col = blk_x * DCTSIZE + j;
                        if (row < total_height && col < total_width) {
                            dct_coefficients_[row][col] = static_cast<double>(block[i * DCTSIZE + j]);
                        }
                    }
                }
            }
        }

        // Increment the height offset for the next component
        current_height_offset += c_height;
    }
}

/**
 * @brief Extracts the quantization matrix from the JPEG decompression structure.
 *
 * Retrieves the first quantization table (usually for the luminance channel) and stores
 * it in a standard 8x8 matrix format.
 *
 * @note Assumes that `quant_tbl_ptrs[0]` is valid and populated.
 */
void JPEGFile::ExtractQmatrix() {
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            QM_[i][j] = cinfo_.quant_tbl_ptrs[0]->quantval[i * 8 + j];
        }
    }
}