#ifndef PERTURBEDQUANTIZATION_H
#define PERTURBEDQUANTIZATION_H

#include <vector>
#include <string>

#include "../jpeg_file/JPEGFile.h"

class PerturbedQuantization {
public:
    PerturbedQuantization() = delete;

    struct ContribMultiple {
        size_t i;
        size_t j;
        size_t k;
        double distance;
        int q1;
        int q2;
    };

    static int ComputeOptimalCompressionQuality(int original_quality);

    static size_t ComputeCapacity(const std::vector<ContribMultiple>& contrib_multiples);

    static std::vector<ContribMultiple> ComputeContributingMultiples(const JPEGFile* input_file,
                                                                     const JPEGFile* temp_file,
                                                                     const std::vector<std::vector<std::vector<double>>>& D2raw);

    static std::vector<std::vector<double>> EmbedMessage(const JPEGFile* input_file, const JPEGFile* temp_file,
                                                          const std::vector<std::vector<std::vector<double>>>& D2raw,
                                                          const std::vector<ContribMultiple>& contrib_multiples,
                                                          std::string message);

    static std::string DecodeMessage(const JPEGFile* original_image, const JPEGFile* embedded_image);

private:
    static std::vector<std::vector<int>> E_;

    static void InitializeContributingPairs(std::vector<std::vector<int>>& E);

    static int Sign(int val);

    static int RoundToInt(double val);

    static std::vector<int> StringToBitVector(const std::string& message);

    static std::string BitVectorToString(const std::vector<int>& bits);

    static std::vector<int> ExtractMessageBits(const std::vector<std::vector<std::vector<int>>>& D2n,
                                               const std::vector<std::vector<std::vector<int>>>& D1,
                                               const std::vector<ContribMultiple>& contrib_multiples);
};

#endif // PERTURBEDQUANTIZATION_H
