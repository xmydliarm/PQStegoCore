#ifndef JPEGPROCESSOR_H
#define JPEGPROCESSOR_H

#include <vector>
#include <string>

#include "../jpeg_file/JPEGFile.h"

#define DCTSIZE 8
#define DCTSIZE2 64
#define CONST_BITS 13
#define PASS1_BITS 2
#define INT32 long
#define JSAMPLE unsigned char
#define DCTELEM	int
#define ONE ((INT32)1)

#define FIX_0_298631336 ((INT32)2446)
#define FIX_0_390180644 ((INT32)3196)
#define FIX_0_541196100 ((INT32)4433)
#define FIX_0_765366865 ((INT32)6270)
#define FIX_0_899976223 ((INT32)7373)
#define FIX_1_175875602 ((INT32)9633)
#define FIX_1_501321110 ((INT32)12299)
#define FIX_1_847759065 ((INT32)15137)
#define FIX_1_961570560 ((INT32)16069)
#define FIX_2_053119869 ((INT32)16819)
#define FIX_2_562915447 ((INT32)20995)
#define FIX_3_072711026 ((INT32)25172)

#define MULTIPLY(var, const) ((var) * (const))
#define DEQUANTIZE(coef, quantval) (((int)(coef)) * (quantval))
#define RIGHT_SHIFT(x, shft) ((x) >> (shft))
#define DESCALE(x, n) RIGHT_SHIFT((x) + (ONE << ((n)-1)), n)

class JPEGProcessor {
public:
    JPEGProcessor() = delete;

    static void SaveCoverJPEG(const char* filename, const JPEGFile* original_file, int quality);

    static void ComputeDCTBlocks(const std::vector<std::vector<double>>& X,
                                 std::vector<std::vector<std::vector<double>>>& D_raw);

    static int EstimateQuality(const JPEGFile* jpeg_file);

    static void DecompressImage(const std::vector<std::vector<std::vector<int>>>& D,
                                const std::vector<std::vector<int>>& QM,
                                std::vector<std::vector<double>>& X
    );

    static void SavePerturbedJPEG(const std::string& file_path, const JPEGFile* temp_file,
                                  std::vector<std::vector<double>>& plane);

    static void VecToPlane(const std::vector<std::vector<std::vector<double>>>& cube, size_t MB, size_t NB,
                           std::vector<std::vector<double>>& plane);

    static void PlaneToVec(const std::vector<std::vector<double>>& plane, size_t Y, size_t X,
                           std::vector<std::vector<std::vector<int>>>& D);

    static std::string GetTempFilename(const std::string& prefix, const std::string& extension);

private:
    static void PerformSlowIDCT(double out[], const double coef_block[], const double qt[]);

    static void PerformSlowFDCT(double out[], const double in[]);

    static int Range_limit(int x);

    static void ComputeQmatrix(int quality, std::vector<std::vector<int>>& QM);

    static void ComputeQuantizedDCT(const std::vector<std::vector<double>>& Z,
                                    const std::vector<std::vector<double>>& QM,
                                    std::vector<std::vector<double>>& QD);
};

#endif // JPEGPROCESSOR_H
