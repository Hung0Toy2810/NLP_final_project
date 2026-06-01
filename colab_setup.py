import os
import sys
import subprocess

def main():
    print("============================================================")
    print(" GOOGLE COLAB SETUP — SWFT Training Pipeline")
    print("============================================================")

    # LƯU Ý CHO USER: Vì không dùng Google Drive, toàn bộ dữ liệu 
    # và Checkpoints sẽ nằm trên ổ đĩa tạm thời của Colab. 
    # Mọi thứ sẽ mất đi khi bạn tắt tab Colab. Nhớ tải Checkpoints về máy nhé!

    # 1. Thiết lập Biến môi trường trực tiếp trên ổ cứng Colab
    # Cache lưu ở ổ cứng của Colab (SSD)
    os.environ["SWFT_CACHE_DIR"] = "/content/data_cache"
    
    # Checkpoint lưu trực tiếp trong thư mục dự án hiện tại
    os.environ["SWFT_CHECKPOINT_DIR"] = "./checkpoints"
    os.environ["SWFT_DEBUG"] = "0"  # Tắt debug để train full data

    os.makedirs(os.environ["SWFT_CHECKPOINT_DIR"], exist_ok=True)
    
    print("\n[Cấu hình Thư mục]")
    print(f" - Cache Dir      : {os.environ['SWFT_CACHE_DIR']}")
    print(f" - Checkpoint Dir : {os.environ['SWFT_CHECKPOINT_DIR']} (Nhớ download về trước khi thoát!)")

    # 2. Chạy chuẩn bị dữ liệu
    print("\n============================================================")
    print(" BƯỚC 1/2: CHUẨN BỊ DỮ LIỆU (PREPARE DATA)")
    print("============================================================")
    print("Lưu ý: Quá trình tokenize 20GB text sẽ mất một chút thời gian (tùy thuộc vào số cores của Colab).")
    try:
        subprocess.run([sys.executable, "src/prepare_data.py"], check=True)
    except subprocess.CalledProcessError:
        print(" Lỗi xảy ra trong quá trình chuẩn bị dữ liệu!")
        sys.exit(1)

    # 3. Chạy Training
    print("\n============================================================")
    print(" BƯỚC 2/2: BẮT ĐẦU TRAINING (TRAIN)")
    print("============================================================")
    try:
        subprocess.run([sys.executable, "src/train.py"], check=True)
    except subprocess.CalledProcessError:
        print(" Lỗi xảy ra trong quá trình huấn luyện!")
        sys.exit(1)

    print("\n HOÀN TẤT! Mô hình đã train xong.")
    print(" CẢNH BÁO TỐI QUAN TRỌNG: Hãy click chuột phải vào thư mục './checkpoints'")
    print(" và chọn 'Download' để tải file trọng số (.pt) về máy Mac của bạn TRƯỚC KHI TẮT TAB COLAB!")

if __name__ == "__main__":
    main()
