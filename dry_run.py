import sys
import os

# Add src folder to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import config

# Override config parameters for extremely fast dry-run
config.DEBUG_CONFIG["enabled"] = True
config.DEBUG_CONFIG["num_samples"] = 40       # Dữ liệu train cực nhỏ
config.DEBUG_CONFIG["eval_samples"] = 10      # Dữ liệu eval cực nhỏ

config.TRAIN_CONFIG["batch_size_debug"] = 4   # Batch size 4 để phù hợp 40 samples (10 batches)
config.TRAIN_CONFIG["epochs_stage0"] = 1
config.TRAIN_CONFIG["epochs_stage1"] = 1
config.TRAIN_CONFIG["epochs_stage2"] = 1
config.TRAIN_CONFIG["save_every_epoch"] = True

import train

if __name__ == "__main__":
    print("==================================================================")
    print("🚀 BẮT ĐẦU DRY-RUN TEST TRÊN APPLE SILICON (MPS) 🚀")
    print("==================================================================")
    print("Mục đích: Ép toàn bộ pipeline chạy 1 vòng cực ngắn để bắt lỗi (nếu có).")
    print("Dữ liệu: Vô cùng nhỏ (40 samples/stage). Sẽ hoàn thành trong ~1-2 phút.")
    print(f"Device: {config.get_device()}")
    print("==================================================================\n")
    
    train.main()
