#ifndef JPEGFILE_H
#define JPEGFILE_H

#include <string>
#include <vector>

#include "jpeglib.h"

class JPEGFile {
public:
    explicit JPEGFile(const std::string &filePath);

    ~JPEGFile();

    int getQuality() const { return quality_; }
    std::string getFilePath() const { return filePath_; }
    const jpeg_decompress_struct &getCinfo() const { return cinfo_; }
    const std::vector<std::vector<int>> &getQM() const { return QM_; }
    const std::vector<std::vector<std::vector<int>>> &getD() const { return D_; }
    const std::vector<std::vector<double>> &getX() const { return X_; }

private:
    int quality_;
    std::string filePath_;
    jpeg_decompress_struct cinfo_;
    std::vector<std::vector<int>> QM_;
    std::vector<std::vector<double>> dct_coefficients_;
    std::vector<std::vector<std::vector<int>>> D_;
    std::vector<std::vector<double>> X_;

    void ReadJPEGInfo(FILE *infile);

    void ExtractDCTCoefficients();

    void ExtractQmatrix();
};

#endif // JPEGFILE_H
