#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <iostream>
#include <fstream>

using namespace nvinfer1;
using namespace nvonnxparser;

class Logger : public ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} logger;

int main(int argc, char* argv[]) {

    if (argc < 3) { // We need at least 3 arguments: program name + model path + engine path
        std::cerr << "Usage: " << argv[0] << " <path_to.onnx_model> <path_to_engine_file>" << std::endl;
        return 1;
    }

    const char* modelFilePath = argv[1];
    const char* engineFilePath = argv[2];
    std::cout << "Parsing ONNX model: " << modelFilePath << std::endl;

    IBuilder* builder = createInferBuilder(logger);
    if (!builder) {
        std::cerr << "ERROR: Failed to create TensorRT builder!" << std::endl;
        return 1;
    }

    uint32_t flags = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition* network = builder->createNetworkV2(flags);
    if (!network) {
        std::cerr << "ERROR: Failed to create TensorRT network!" << std::endl;
        delete builder;
        return 1;
    }

    IParser* parser = createParser(*network, logger);
    if (!parser) {
        std::cerr << "ERROR: Failed to create ONNX parser!" << std::endl;
        delete network;
        delete builder;
        return 1;
    }

    parser->parseFromFile(modelFilePath,
        static_cast<int32_t>(ILogger::Severity::kWARNING));
    for (int32_t i = 0; i < parser->getNbErrors(); ++i)
    {
        std::cout << parser->getError(i)->desc() << std::endl;
    }

    if (parser->getNbErrors() > 0) {
        std::cerr << "ERROR: Failed to parse ONNX model!" << std::endl;
        delete parser;
        delete network;
        delete builder;
        return 1;
    }
    std::cout << "Successfully parsed ONNX model!" << std::endl;

    IBuilderConfig* config = builder->createBuilderConfig();

    IHostMemory* serializedModel = builder->buildSerializedNetwork(*network, *config);
    if (!serializedModel) {
        std::cerr << "ERROR: Failed to build serialized network!" << std::endl;
        delete parser;
        delete network;
        delete config;
        delete builder;
        delete serializedModel;
        return 1;
    }
    std::cout << "TensorRT engine built successfully." << std::endl;

    // Serialized model now contains the necessary copies of the weights so we can delete objects
    delete parser;
    delete network;
    delete config;
    delete builder;

    std::cout << "Saving engine to: " << engineFilePath << std::endl;
    if (serializedModel) {
        std::ofstream engineFileStream(engineFilePath, std::ios::binary);
        if (!engineFileStream) {
            std::cerr << "ERROR: Failed to open engine file for writing!" << std::endl;
            delete serializedModel;
            return 1;
        }
        engineFileStream.write(static_cast<const char*>(serializedModel->data()), serializedModel->size());
        engineFileStream.close();
        delete serializedModel;
        std::cout << "Successfully saved TensorRT engine to: " << engineFilePath << std::endl;
    } else {
        std::cerr << "ERROR: Failed to build serialized network!" << std::endl;
        delete serializedModel;
        return 1;
    }

}