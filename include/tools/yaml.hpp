#ifndef TOOLS__YAML_HPP
#define TOOLS__YAML_HPP

#include <yaml-cpp/yaml.h>

#include "tools/logger.hpp"


inline YAML::Node yaml_load(const std::string & path)
{
    try {
        return YAML::LoadFile(path);
    } catch (const YAML::BadFile & e) {
        logger()->error("[YAML] Failed to load file: {}", e.what());
        exit(1);
    } catch (const YAML::ParserException & e) {
        logger()->error("[YAML] Parser error: {}", e.what());
        exit(1);
    }
}

template <typename T>
inline T yaml_read(const YAML::Node & yaml, const std::string & key)
{
    if (yaml[key]) return yaml[key].as<T>();
    logger()->error("[YAML] {} not found!", key);
    exit(1);
}


#endif  // TOOLS__YAML_HPP