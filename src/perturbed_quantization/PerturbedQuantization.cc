#include "PerturbedQuantization.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>

#include "../jpeg_processor/JPEGProcessor.h"

std::vector<std::vector<int> > PerturbedQuantization::E_;

/**
 * @brief Embeds a binary message into an image using perturbed quantization.
 *
 * This method modifies selected DCT coefficients in the second-compressed image (`D2n`) using
 * quantization-based embedding, such that the changes encode the given message.
 *
 * @param[in] input_file Pointer to the original JPEGFile object (first compression).
 * @param[in] temp_file Pointer to the JPEGFile created from the second compression.
 * @param[in] D2raw Raw DCT coefficients after second compression.
 * @param[in] contrib_multiples Precomputed vector of eligible DCT positions and their (q1, q2) pairs.
 * @param[in] message The message string to embed. Appends an end-marker (0xFF 0x00) automatically.
 *
 * @return 2D matrix (plane) of DCT coefficients after message embedding.
 *
 * @throws std::runtime_error if the message exceeds image embedding capacity.
 *
 * @note This function modifies only eligible DCT coefficients (non-DC) based on contributing pairs.
 */
std::vector<std::vector<double>> PerturbedQuantization::EmbedMessage(const JPEGFile* input_file, const JPEGFile* temp_file,
                                         const std::vector<std::vector<std::vector<double>>>& D2raw,
                                         const std::vector<ContribMultiple>& contrib_multiples,
                                         std::string message) {
    const auto& Qm2 = temp_file->getQM();
    const auto& D1 = input_file->getD();

    // Find the message end sequence 0xFF 0x00 (11111111 00000000)
    const std::string END_MARKER = "\xFF\x00";
    message = message.append(END_MARKER);

    size_t MessageLength = message.length() * 8;
    if (MessageLength > contrib_multiples.size()) {
        throw std::runtime_error("Message length exceeded capacity of image!");
    }

    const unsigned int MB = input_file->getX().size() / 8;
    const unsigned int NB = input_file->getX()[0].size() / 8;

    auto D2n = std::vector(MB, std::vector(NB, std::vector<double>(64)));

    int c = 0; // Message bit counter
    int changes = 0; // Number of changes against cover
    int NonZeros = 0, Zeros = 0; // Zero and non-zero counters

    for (int k = 0; k < 64; ++k) {
        int ii = k / 8; // row in 8x8 block
        int jj = k % 8; // column in 8x8 block

        // Non-contributing pair, round to closest multiple of q2
        for (size_t i = 0; i < MB; ++i) {
            for (size_t j = 0; j < NB; ++j) {
                D2n[i][j][k] = RoundToInt(D2raw[i][j][k] / Qm2[jj][ii]);
                if (D2n[i][j][k] > 0 && k > 0) NonZeros++;
                else Zeros++;
            }
        }
    }

    std::vector<int> Message = StringToBitVector(message);
    for (size_t m = 0; m < Message.size(); ++m) {
        const auto cm = contrib_multiples[m];

        D2n[cm.i][cm.j][cm.k] = (D1[cm.i][cm.j][cm.k] * cm.q1 + Sign(D1[cm.i][cm.j][cm.k]) * Message[c] * cm.q2 / 2) / cm.q2;

        int cover_value = RoundToInt(D2raw[cm.i][cm.j][cm.k] / cm.q2);
        if (std::abs(cm.q2 * cover_value - cm.q1 * RoundToInt(cm.q2 * cover_value / cm.q1)) == cm.q2 / 2) {
            cover_value += Sign(D2raw[cm.i][cm.j][cm.k] / cm.q2 - cover_value);
        }

        if (D2n[cm.i][cm.j][cm.k] != cover_value) {
            changes++;
        }

        c++;
    }

    std::vector<std::vector<double>> resultPlane;
    JPEGProcessor::VecToPlane(D2n, MB, NB, resultPlane);

    std::cout << "\n ---- Message embedded! (Changes: " << changes << ") ----" << std::endl;
    //std::cout << "Zeros: " << Zeros << ", NonZeros: " << NonZeros << "\n";

    return resultPlane;
}

/**
 * @brief Decodes a hidden message from a JPEG image that was embedded using perturbed quantization.
 *
 * Reconstructs the original compression (first compression) of the embedded image and extracts bits
 * from positions matching the contributing multiples criteria.
 *
 * @param[in] original_image Pointer to the original JPEGFile (used as reference).
 * @param[in] embedded_image Pointer to the JPEGFile with the embedded message.
 *
 * @return Extracted message string, truncated at the special end-marker (0xFF 0x00).
 *
 * @throws std::runtime_error if input images mismatch in size or quality order is incorrect.
 */
std::string PerturbedQuantization::DecodeMessage(JPEGFile *original_image, const JPEGFile *embedded_image) {
    if (original_image->getQuality() <= embedded_image->getQuality()) {
        throw std::runtime_error("Incorrect original image, higher quality expected");
    }

    if (original_image->getCinfo().image_width != embedded_image->getCinfo().image_width
        || original_image->getCinfo().image_height != embedded_image->getCinfo().image_height) {
        throw std::runtime_error("Size mismatch, incorrect images");
    }

    std::string temp_path = JPEGProcessor::GetTempFilename("pq_image", "jpg");
    JPEGProcessor::SaveCoverJPEG(temp_path.c_str(), original_image, original_image->getQuality() - 5);

    auto temp_cover_image = new JPEGFile(temp_path);

    auto D2raw = std::vector<std::vector<std::vector<double> > >();
    JPEGProcessor::ComputeDCTBlocks(original_image->getX(), D2raw);

    auto contrib_multiples = ComputeContributingMultiples(original_image, temp_cover_image, D2raw);

    auto result = ExtractMessageBits(embedded_image->getD(), original_image->getD(), contrib_multiples);

    std::string decodedMessage = BitVectorToString(result);

    // Find the message end sequence 0xFF 0x00 (11111111 00000000)
    const std::string END_MARKER = "\xFF\x00";
    
    size_t end_pos = decodedMessage.find(END_MARKER);

    decodedMessage = decodedMessage.substr(0, end_pos);

    return decodedMessage;
}

/**
 * @brief Extracts the embedded message bits from the given DCT coefficient structures.
 *
 * For each entry in `contrib_multiples`, the function compares the actual embedded value
 * to the theoretical embedding result for both message bits 0 and 1.
 *
 * @param[in] D2n DCT coefficients of the embedded image (after message embedding).
 * @param[in] D1 DCT coefficients of the original image.
 * @param[in] contrib_multiples Vector of positions and quantization pairs used for embedding.
 *
 * @return Vector of extracted message bits (0s and 1s).
 *
 * @throws std::runtime_error if bit value cannot be reliably determined at any position.
 */
std::vector<int> PerturbedQuantization::ExtractMessageBits(const std::vector<std::vector<std::vector<int>>>& D2n,
                                 const std::vector<std::vector<std::vector<int>>>& D1,
                                 const std::vector<ContribMultiple>& contrib_multiples) {
    std::vector<int> extracted_bits;

    for (const auto cm : contrib_multiples) {
        const int dct_embedded_zero = (D1[cm.i][cm.j][cm.k] * cm.q1 + Sign(D1[cm.i][cm.j][cm.k]) * 0 * cm.q2 / 2) / cm.q2;
        const int dct_embedded_one = (D1[cm.i][cm.j][cm.k] * cm.q1 + Sign(D1[cm.i][cm.j][cm.k]) * 1 * cm.q2 / 2) / cm.q2;

        if (dct_embedded_one == dct_embedded_zero) {
            throw std::runtime_error("Unable to decode message bit");
        }

        if (D2n[cm.i][cm.j][cm.k] == dct_embedded_zero) {
            extracted_bits.push_back(0);
        }
        else if (D2n[cm.i][cm.j][cm.k] == dct_embedded_one) {
            extracted_bits.push_back(1);
        }
    }

    return extracted_bits;
}

/**
 * @brief Converts a string into a vector of bits.
 *
 * Converts each character into 8 bits (MSB to LSB), resulting in a bit vector suitable for embedding.
 *
 * @param[in] message Input string to convert.
 * @return Vector of 0s and 1s representing the binary form of the message.
 */
std::vector<int> PerturbedQuantization::StringToBitVector(const std::string& message) {
    std::vector<int> bit_vector;
    for (char c: message) {
        // Extract bits from MSB (most significant bit) to LSB (least significant bit)
        for (int i = 0; i < 8; ++i) {
            int bit = (c & (0x80 >> i)) ? 1 : 0; // Mask each bit
            bit_vector.push_back(bit);
        }
    }
    return bit_vector;
}

/**
 * @brief Converts a vector of bits into a string.
 *
 * Groups the bit vector into 8-bit chunks (MSB to LSB) and converts each group to a character.
 *
 * @param[in] bits Vector of 0s and 1s representing the binary message.
 * @return Reconstructed string from binary data.
 */
std::string PerturbedQuantization::BitVectorToString(const std::vector<int>& bits) {
    std::string message;
    message.reserve(bits.size() / 8);

    for (size_t i = 0; i + 8 <= bits.size(); i += 8) {
        std::uint8_t byte = 0;
        for (size_t j = 0; j < 8; ++j) {
            byte |= static_cast<std::uint8_t>(bits[i + j]) << (7 - j);
        }
        message.push_back(static_cast<char>(byte));
    }
    return message;
}

/**
 * @brief Rounds a double to the nearest integer.
 *
 * @param[in] val Input value.
 * @return Rounded integer.
 */
int PerturbedQuantization::RoundToInt(const double val) {
    return static_cast<int>(std::round(val));
}

/**
 * @brief Returns the sign of an integer.
 *
 * @param[in] val Input value.
 * @return -1 for negative, 1 for positive, 0 for zero.
 */
int PerturbedQuantization::Sign(const int val) {
    return (val > 0) - (val < 0);
}

/**
 * @brief Populates the matrix E with all valid contributing quantization pairs (q1, q2).
 *
 * A contributing pair satisfies the condition that (j / gcd(i, j)) is even.
 * This enables robust embedding in PQ by ensuring bit-distinguishability.
 *
 * @param[out] E Matrix to store contributing pair flags and their divisor count.
 */
void PerturbedQuantization::InitializeContributingPairs(std::vector<std::vector<int>>& E) {
    // Calculate contributing pairs of quantization steps
    for (int i = 2; i < 121; i++) {
        for (int j = i + 1; j < 121; j++) {
            int g = std::gcd(i, j);
            if ((j / g) % 2 == 0) {
                E[i - 1][j - 1] = j / g;
            }
        }
    }
}

/**
 * @brief Computes the maximum message capacity for embedding based on available contributing coefficients.
 *
 * @param[in] contrib_multiples Precomputed list of embedding-eligible coefficient positions.
 * @return Number of usable bits (excluding 16 bits reserved for the end-marker).
 */
size_t PerturbedQuantization::ComputeCapacity(const std::vector<ContribMultiple>& contrib_multiples) {
    if (contrib_multiples.size() > 16) {
        return contrib_multiples.size() - 16;
    }

    return 0;
}

/**
 * @brief Identifies DCT coefficient positions that allow for reliable embedding via perturbed quantization.
 *
 * For each AC coefficient (non-DC), the method tests whether the condition for contributing multiples
 * is satisfied using the quantization tables and fills a sorted vector of embedding positions.
 *
 * @param[in] input_file Original JPEGFile (first compression).
 * @param[in] temp_file JPEGFile from recompression (second compression).
 * @param[in] D2raw Raw DCT coefficients after second compression.
 *
 * @return Sorted vector of ContribMultiple structures identifying embeddable positions.
 */
std::vector<PerturbedQuantization::ContribMultiple> PerturbedQuantization::ComputeContributingMultiples(
    const JPEGFile* input_file,
    const JPEGFile* temp_file,
    const std::vector<std::vector<std::vector<double>>>& D2raw) {

    const auto &Qm1 = input_file->getQM();
    const auto &Qm2 = temp_file->getQM();
    const auto &D1 = input_file->getD();

    const size_t MB = input_file->getX().size() / 8;
    const size_t NB = input_file->getX()[0].size() / 8;

    if (E_.empty()) {
        E_ = std::vector(121, std::vector(121, 0));
        InitializeContributingPairs(E_);
    }

    std::vector<ContribMultiple> contrib_multiples;
    size_t c = 0; // Message bit counter

    // Iterate over the 64 DCT modes
    for (size_t k = 1; k < 64; ++k) { // k = 1 -> SKIP DC COEF
        const int q1 = Qm1[k % 8][k / 8];
        const int q2 = Qm2[k % 8][k / 8];

        if (E_[q1 - 1][q2 - 1] != 0 && E_[q1 - 1][q2 - 1] % 2 == 0) {
            // Loop over all blocks
            for (size_t i = 0; i < MB; ++i) {
                for (size_t j = 0; j < NB; ++j) {
                    if (std::fmod(D1[i][j][k] - E_[q1 - 1][q2 - 1] * 0.5, E_[q1 - 1][q2 - 1]) == 0) {
                        // D1(i,j,k) is the contributing multiple
                        c++;
                        double D2_scaled = D2raw[i][j][k] / static_cast<double>(q2);
                        contrib_multiples.push_back({
                            i, j, k,
                            std::fabs(std::fabs(D2_scaled - std::round(D2_scaled)) - 0.5),
                            q1, q2
                        });
                    }
                }
            }
        }
    }

    // Sort by the value component (ascending order - most suitable first)
    auto comparator = [](const ContribMultiple &a, const ContribMultiple &b) {
        return a.value < b.value;
    };

    std::ranges::stable_sort(contrib_multiples, comparator);

    return contrib_multiples;
}
