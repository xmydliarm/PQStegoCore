#include <iostream>

#include "src/jpeg_processor/JPEGProcessor.h"
#include "src/perturbed_quantization/PerturbedQuantization.h"

using namespace std;

void PrintUsage() {
    cerr << "Usage:\n"
         << "  Modes:\n"
         << "    -capacity <original_image>\n"
         << "    -encode <original_image> <output_image> <message>\n"
         << "    -decode <original_image> <embedded_image>\n";
}

int main(int argc, char* argv[]) {
    if(argc < 2) {
        PrintUsage();
        return 1;
    }

    string mode = argv[1];
    
    if(mode == "-capacity" && argc == 3) {
        // Capacity calculation mode
        string original_image_path = argv[2];

        auto original_image = new JPEGFile(original_image_path);

        const int cover_quality = PerturbedQuantization::ComputeOptimalCompressionQuality(original_image->getQuality());

        string temp_path = JPEGProcessor::GetTempFilename("pq_image", "jpg");
        JPEGProcessor::SaveCoverJPEG(temp_path.c_str(), original_image, cover_quality);
        auto temp_cover_image = new JPEGFile(temp_path);

        auto D2raw = vector<vector<vector<double>>>();
        JPEGProcessor::ComputeDCTBlocks(original_image->getX(), D2raw);

        auto contrib_multiples = PerturbedQuantization::ComputeContributingMultiples(original_image, temp_cover_image, D2raw);

        size_t capacity = PerturbedQuantization::ComputeCapacity(contrib_multiples);

        remove(temp_path.c_str());

        cout << "Estimated capacity: " << capacity << " bits\n";

    } else if(mode == "-encode" && argc == 5) {
        // Encoding mode
        string original_image_path = argv[2];
        string output_image_path = argv[3];
        string message = argv[4];

        auto original_image = new JPEGFile(original_image_path);

        const int cover_quality = PerturbedQuantization::ComputeOptimalCompressionQuality(original_image->getQuality());

        string temp_path = JPEGProcessor::GetTempFilename("pq_image", "jpg");
        JPEGProcessor::SaveCoverJPEG(temp_path.c_str(), original_image, cover_quality);
        auto temp_cover_image = new JPEGFile(temp_path);

        auto D2raw = vector<vector<vector<double>>>();
        JPEGProcessor::ComputeDCTBlocks(original_image->getX(), D2raw);

        auto contrib_multiples = PerturbedQuantization::ComputeContributingMultiples(original_image, temp_cover_image, D2raw);

        auto image_X2_embedded = PerturbedQuantization::EmbedMessage(original_image, temp_cover_image, D2raw, contrib_multiples, message);

        JPEGProcessor::SavePerturbedJPEG(output_image_path, temp_cover_image, image_X2_embedded);

        remove(temp_path.c_str());

        std::cout << "Perturbed JPEG saved successfully!" << std::endl;

    } else if(mode == "-decode" && argc == 4) {
        // Decoding mode
        string original_image_path = argv[2];
        string embedded_image_path = argv[3];

        auto original_file = new JPEGFile(original_image_path);
        auto embedded_file = new JPEGFile(embedded_image_path);

        string decoded_message = PerturbedQuantization::DecodeMessage(original_file, embedded_file);

        cout << "Decoded message: " << decoded_message << endl;

    } else {
        PrintUsage();
        return 1;
    }

    return 0;
}
