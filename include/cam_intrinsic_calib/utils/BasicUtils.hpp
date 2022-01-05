/**
 * @file    BasicUtils.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace _cv
{
inline std::vector<std::string> splitByDelim(const std::string& s, const char delimiter)
{
    std::stringstream ss(s);
    std::string token;
    std::vector<std::string> tokens;
    while (std::getline(ss, token, delimiter)) {
        tokens.emplace_back(token);
    }
    return tokens;
}

inline std::vector<std::string> parseMetaDataFile(const std::string& metaDataFilePath)
{
    std::ifstream inFile;
    inFile.open(metaDataFilePath);

    if (!inFile) {
        throw std::runtime_error("unable to open " + metaDataFilePath);
    }

    std::stringstream buffer;
    buffer << inFile.rdbuf();

    return splitByDelim(buffer.str(), '\n');
}
}  // namespace _cv
