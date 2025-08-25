/**
 * @file    gpu_file_manager.cu
 * @brief   GPU File Manager Implementation - GPU Virtual File System Layer
 *
 * @details
 * This implementation provides a GPU file management system that mirrors
 * the NVMe file manager patterns. It manages virtual GPU files that map
 * to physical NVMe storage through a persistent log-based system.
 *
 * Key features:
 * 1. Persistent GPU file metadata storage with bitmap allocation
 * 2. Compact NVMe mapping for efficient GPU-to-storage bridging
 * 3. Thread-safe operations with mutex protection
 * 4. Automatic persistence with configurable thresholds
 * 5. 64-byte aligned structures for optimal performance
 *
 * @author      GeminiFS Team
 * @version     1.0.0
 * @date        2025-01-27
 */

#include "include/gpu_file_manager.cuh"
#include "ops.h"  // 包含geminifs_debug宏定义
#include <cstring>
#include <chrono>
#include <iostream>
#include <algorithm>

// 构造函数
GPUFileManager::GPUFileManager(const std::string& log_path, size_t persistence_threshold)
    : log_file_path_(log_path)
    , log_file_handle_(nullptr)
    , persistence_threshold_(persistence_threshold)
    , pending_writes_count_(0) {
    
    geminifs_debug("Initializing GPUFileManager with log path: %s\n", log_path.c_str());
    
    // 初始化dirty bitmap
    dirty_bitmap_.resize(GPU_MAX_RECORDS, false);
    
    // 初始化日志文件
    initializeLogFile();
    
    // 从文件加载现有数据
    loadFromFile();
    
    geminifs_debug("GPUFileManager initialized with %zu active files\n", 
                   file_id_to_desc_map_.size());
}

// 析构函数
GPUFileManager::~GPUFileManager() {
    geminifs_debug("Destroying GPUFileManager, forcing final persistence\n");
    
    // 强制持久化所有挂起的更改
    forcePersist();
    
    // 关闭文件句柄
    if (log_file_handle_) {
        fclose(log_file_handle_);
        log_file_handle_ = nullptr;
    }
    
    geminifs_debug("GPUFileManager destroyed\n");
}

// 创建GPU文件
bool GPUFileManager::createGPUFile(const std::vector<uint32_t>& nvme_file_ids,
                                   const std::vector<size_t>& tensor_shape, GPUFileDesc& out_desc, 
                                   size_t total_size, size_t block_size) {
    std::lock_guard<std::mutex> lock(mtx_);
    
    // 查找下一个可用的slot，slot_index就是file_id
    long slot_index = findNextFreeSlot();
    if (slot_index < 0) {
        geminifs_debug("No free slots available for new GPU file\n");
        return false;
    }
    
    GPUFileId file_id = static_cast<GPUFileId>(slot_index);
    
    // 检查文件ID是否已存在
    if (file_id_to_desc_map_.find(file_id) != file_id_to_desc_map_.end()) {
        geminifs_debug("GPU file with ID %u already exists\n", file_id);
        return false;
    }
    
    // 检查NVMe文件ID数量
    if (nvme_file_ids.size() > 4) {
        geminifs_debug("Too many NVMe file IDs provided (%zu), maximum is 4\n", nvme_file_ids.size());
        return false;
    }
    
    // 检查tensor_shape参数
    if (tensor_shape.size() != 3) {
        geminifs_debug("tensor_shape must have exactly 3 dimensions, got %zu\n", tensor_shape.size());
        return false;
    }
    
    // 创建GPU文件描述符
    GPUFileDesc desc = {};
    desc.file_id = file_id;
    desc.total_file_size = total_size;
    desc.block_size = block_size;
    
    // 设置tensor形状
    desc.tensor_shape[0] = tensor_shape[0];  // 第一维度tensor数量
    desc.tensor_shape[1] = tensor_shape[1];  // 第二维度tensor数量  
    desc.tensor_shape[2] = tensor_shape[2];  // tensor内存对象大小(应该等于block_size)
    
    // 设置NVMe映射
    desc.nvme_mapping = CompactNVMeMapping(); // 默认构造函数初始化
    desc.nvme_mapping.set_valid(true);
    
    for (size_t i = 0; i < nvme_file_ids.size() && i < 4; ++i) {
        desc.nvme_mapping.nvme_file_ids[i] = nvme_file_ids[i];
        desc.nvme_mapping.set_nvme_controller_exists(i, true);
    }
    
    // 写入到文件
    if (!writeRecordToSlot(desc, slot_index)) {
        geminifs_debug("Failed to write GPU file record to slot %ld\n", slot_index);
        return false;
    }
    
    // 添加到内存映射
    file_id_to_desc_map_[file_id] = desc;
    
    // 标记为dirty并增加挂起写入计数
    dirty_bitmap_[slot_index] = true;
    pending_writes_count_++;
    
    // 检查是否需要持久化
    if (pending_writes_count_ >= persistence_threshold_) {
        persistBitmap();
    }
    
    out_desc = desc;
    
    geminifs_debug("Successfully created GPU file with ID %u at slot %ld\n", file_id, slot_index);
    return true;
}

// 删除GPU文件
bool GPUFileManager::deleteGPUFile(GPUFileId file_id) {
    std::lock_guard<std::mutex> lock(mtx_);
    
    geminifs_debug("Deleting GPU file with ID: %u\n", file_id);
    
    auto it = file_id_to_desc_map_.find(file_id);
    if (it == file_id_to_desc_map_.end()) {
        geminifs_debug("GPU file with ID %u not found\n", file_id);
        return false;
    }
    
    uint32_t slot_index = file_id;  // file_id就是slot_index
    
    // 从内存映射中移除
    file_id_to_desc_map_.erase(it);
    
    // 标记slot为可用
    dirty_bitmap_[slot_index] = false;
    
    // 更新活跃记录计数
    header_.active_record_count--;
    
    // 将空记录写入文件
    GPUFileDesc empty_desc = {};
    writeRecordToSlot(empty_desc, slot_index);
    
    pending_writes_count_++;
    
    // 检查是否需要持久化
    if (pending_writes_count_ >= persistence_threshold_) {
        persistBitmap();
    }
    
    geminifs_debug("Successfully deleted GPU file with ID %u from slot %u\n", file_id, slot_index);
    return true;
}

// 根据ID获取GPU文件
bool GPUFileManager::getGPUFileById(GPUFileId file_id, GPUFileDesc& out_desc) const {
    std::lock_guard<std::mutex> lock(mtx_);
    
    auto it = file_id_to_desc_map_.find(file_id);
    if (it == file_id_to_desc_map_.end()) {
        geminifs_debug("GPU file with ID %u not found\n", file_id);
        return false;
    }
    
    out_desc = it->second;
    return true;
}

// 获取所有GPU文件ID
std::vector<GPUFileId> GPUFileManager::getAllGPUFileIds() const {
    std::lock_guard<std::mutex> lock(mtx_);
    
    std::vector<GPUFileId> file_ids;
    file_ids.reserve(file_id_to_desc_map_.size());
    
    for (const auto& pair : file_id_to_desc_map_) {
        file_ids.push_back(pair.first);
    }
    
    return file_ids;
}

// 强制持久化
void GPUFileManager::forcePersist() {
    std::lock_guard<std::mutex> lock(mtx_);
    persistBitmap();
}

// 初始化日志文件
void GPUFileManager::initializeLogFile() {
    // 尝试打开现有文件
    log_file_handle_ = fopen(log_file_path_.c_str(), "r+b");
    
    if (!log_file_handle_) {
        // 文件不存在，创建新文件
        geminifs_debug("Creating new GPU log file: %s\n", log_file_path_.c_str());
        log_file_handle_ = fopen(log_file_path_.c_str(), "w+b");
        
        if (!log_file_handle_) {
            throw std::runtime_error("Failed to create GPU log file: " + log_file_path_);
        }
        
        // 初始化头部
        header_ = {};
        header_.magic_num = MAGIC_NUMBER;
        header_.header_size = sizeof(GPULogHeader);
        header_.bitmap_offset = header_.header_size;
        header_.bitmap_size = GPU_BITMAP_SIZE_BYTES;
        header_.records_offset = header_.bitmap_offset + header_.bitmap_size;
        header_.record_size = sizeof(GPUFileDesc);
        header_.total_record_capacity = GPU_MAX_RECORDS;
        header_.active_record_count = 0;
        header_.log_file_size = header_.records_offset + (header_.total_record_capacity * header_.record_size);
        
        // 写入头部
        fwrite(&header_, sizeof(GPULogHeader), 1, log_file_handle_);
        
        // 写入空的bitmap
        std::vector<uint8_t> empty_bitmap(GPU_BITMAP_SIZE_BYTES, 0);
        fwrite(empty_bitmap.data(), 1, GPU_BITMAP_SIZE_BYTES, log_file_handle_);
        
        // 强制刷新到磁盘
        fflush(log_file_handle_);
        
        geminifs_debug("Initialized new GPU log file with %zu max records\n", GPU_MAX_RECORDS);
    }
}

// 从文件加载数据
void GPUFileManager::loadFromFile() {
    if (!log_file_handle_) return;
    
    // 定位到文件开头
    fseek(log_file_handle_, 0, SEEK_SET);
    
    // 读取头部
    if (fread(&header_, sizeof(GPULogHeader), 1, log_file_handle_) != 1) {
        geminifs_debug("Failed to read GPU log header\n");
        return;
    }
    
    // 验证魔数
    if (header_.magic_num != MAGIC_NUMBER) {
        geminifs_debug("Invalid magic number in GPU log file\n");
        return;
    }
    
    // 读取bitmap
    fseek(log_file_handle_, header_.bitmap_offset, SEEK_SET);
    std::vector<uint8_t> bitmap_data(GPU_BITMAP_SIZE_BYTES);
    if (fread(bitmap_data.data(), 1, GPU_BITMAP_SIZE_BYTES, log_file_handle_) != GPU_BITMAP_SIZE_BYTES) {
        geminifs_debug("Failed to read GPU bitmap data\n");
        return;
    }
    
    // 转换bitmap到vector<bool>
    for (size_t i = 0; i < GPU_MAX_RECORDS; ++i) {
        size_t byte_index = i / 8;
        size_t bit_index = i % 8;
        dirty_bitmap_[i] = (bitmap_data[byte_index] & (1 << bit_index)) != 0;
    }
    
    // 读取记录
    fseek(log_file_handle_, header_.records_offset, SEEK_SET);
    for (size_t i = 0; i < GPU_MAX_RECORDS; ++i) {
        if (dirty_bitmap_[i]) {
            GPUFileDesc desc;
            if (fread(&desc, sizeof(GPUFileDesc), 1, log_file_handle_) == 1) {
                if (desc.nvme_mapping.is_valid()) {
                    file_id_to_desc_map_[desc.file_id] = desc;
                }
            }
        } else {
            // 跳过空记录
            fseek(log_file_handle_, sizeof(GPUFileDesc), SEEK_CUR);
        }
    }
    
    geminifs_debug("Loaded %zu GPU files from log file\n", file_id_to_desc_map_.size());
}

// 持久化bitmap
void GPUFileManager::persistBitmap() {
    if (!log_file_handle_) return;
    
    // 更新头部的活跃记录计数
    header_.active_record_count = file_id_to_desc_map_.size();
    
    // 写入头部
    fseek(log_file_handle_, 0, SEEK_SET);
    fwrite(&header_, sizeof(GPULogHeader), 1, log_file_handle_);
    
    // 转换vector<bool>到字节数组
    std::vector<uint8_t> bitmap_data(GPU_BITMAP_SIZE_BYTES, 0);
    for (size_t i = 0; i < GPU_MAX_RECORDS; ++i) {
        if (dirty_bitmap_[i]) {
            size_t byte_index = i / 8;
            size_t bit_index = i % 8;
            bitmap_data[byte_index] |= (1 << bit_index);
        }
    }
    
    // 写入bitmap
    fseek(log_file_handle_, header_.bitmap_offset, SEEK_SET);
    fwrite(bitmap_data.data(), 1, GPU_BITMAP_SIZE_BYTES, log_file_handle_);
    
    // 强制刷新到磁盘
    fflush(log_file_handle_);
    
    pending_writes_count_ = 0;
    
    geminifs_debug("Persisted GPU bitmap with %zu active records\n", header_.active_record_count);
}

// 查找下一个空闲slot
long GPUFileManager::findNextFreeSlot() {
    for (size_t i = 0; i < GPU_MAX_RECORDS; ++i) {
        if (!dirty_bitmap_[i]) {
            return static_cast<long>(i);
        }
    }
    return -1; // 没有可用slot
}

// 将记录写入指定slot
bool GPUFileManager::writeRecordToSlot(const GPUFileDesc& desc, uint64_t slot_index) {
    if (!log_file_handle_ || slot_index >= GPU_MAX_RECORDS) {
        return false;
    }
    
    long file_offset = header_.records_offset + (slot_index * header_.record_size);
    fseek(log_file_handle_, file_offset, SEEK_SET);
    
    return fwrite(&desc, sizeof(GPUFileDesc), 1, log_file_handle_) == 1;
}

