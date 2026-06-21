# SFT-BE: Document Similarity bằng Teacher-Student Distillation

## 1. Mục tiêu

Đồ án xây dựng một mô hình sentence/document embedding nhỏ hơn BERT-base để phục vụ bài toán tìm văn bản tương tự. Mô hình được thiết kế theo hướng bi-encoder: mỗi câu hoặc đoạn văn được encode độc lập thành vector, sau đó so sánh bằng cosine similarity.

Phạm vi source code hiện tại chỉ giữ pipeline cuối:

- Huấn luyện Stage 0 bằng supervised teacher-student distillation.
- Dữ liệu huấn luyện: Wikipedia sentence-level.
- Teacher: `sentence-transformers/all-mpnet-base-v2`.
- Student: SFT-BE, Transformer encoder tự cài đặt bằng PyTorch.
- Đánh giá phụ: STS-B Spearman correlation.

Các hướng Stage 1/2 như NLI, PAWS, contrastive learning và BM25 baseline đã bị loại khỏi source chính vì không còn phù hợp với kết quả thực nghiệm cuối.

## 2. Cơ sở khoa học

Các thành phần chính dựa trên các công trình đã qua bình duyệt:

| Thành phần | Nguồn tham khảo | Vai trò trong đồ án |
|:--|:--|:--|
| Transformer encoder | Vaswani et al., NeurIPS 2017 | Self-attention, multi-head attention, FFN |
| BERT tokenizer/config tham khảo | Devlin et al., NAACL 2019 | WordPiece tokenizer, hidden size 768 |
| Siamese bi-encoder | Reimers & Gurevych, EMNLP-IJCNLP 2019 | Encode hai văn bản độc lập rồi so sánh cosine |
| Factorized embedding | Lan et al., ICLR 2020 | Giảm tham số embedding bằng 128 -> 768 |
| Pre-LayerNorm | Xiong et al., ICML 2020 | Ổn định gradient khi train Transformer |
| AdamW | Loshchilov & Hutter, ICLR 2019 | Optimizer chính |
| Cosine annealing | Loshchilov & Hutter, ICLR 2017 | Learning-rate schedule |
| Sentence embedding distillation | Reimers & Gurevych, EMNLP 2020 | Student học mimic embedding space của teacher |
| STS Benchmark | Cer et al., SemEval@ACL 2017 | Đánh giá tương quan semantic similarity |

## 3. Kiến trúc mô hình

SFT-BE là một Transformer encoder nông hơn BERT-base nhưng giữ chiều ẩn 768 để tương thích trực tiếp với teacher embedding.

| Thành phần | Giá trị |
|:--|:--|
| Vocabulary | 30,522 WordPiece tokens |
| Embedding dimension | 128 |
| Hidden size | 768 |
| Transformer layers | 6 |
| Attention heads | 12 |
| FFN hidden size | 3072 |
| Position encoding | Sinusoidal |
| Normalization | Pre-LN |
| Pooling | Mean pooling có attention mask |

Ý tưởng chính là giảm độ sâu từ 12 layer xuống 6 layer, đồng thời dùng factorized embedding 128 -> 768 để giảm số tham số ở tầng embedding. Phần encoder vẫn giữ hidden size 768 để tận dụng tốt phép nhân ma trận trên GPU và để distillation trực tiếp với teacher 768 chiều.

## 4. Hàm loss

Teacher và student đều xuất vector 768 chiều. Teacher chạy ở chế độ `eval/no_grad`, chỉ student nhận gradient.

Với mỗi câu đầu vào:

```text
teacher = normalize(Teacher(sentence))
student = normalize(Student(sentence))
loss = 1 - cosine(student, teacher)
```

Trong code, loss nằm ở:

```text
src/losses.py
Stage0TeacherDistillationLoss
```

Mục tiêu của Stage 0 không phải học nhãn đúng/sai chi tiết, mà là đưa student vào cùng không gian embedding với teacher đã được kiểm chứng. Vì vậy mô hình phù hợp nhất cho bước lọc/ranking ngữ nghĩa ban đầu trong hệ thống document similarity.

## 5. Dữ liệu

Pipeline chuẩn bị dữ liệu nằm ở:

```text
src/prepare_data.py
```

Script tạo hai cache:

| Cache | Mục đích |
|:--|:--|
| `wikipedia_tokenized` | Dữ liệu train Stage 0 |
| `stsb_tokenized` | Dữ liệu đánh giá STS-B |

Wikipedia được tách từ article-level thành sentence-level, lọc câu quá ngắn hoặc quá dài, sau đó tokenize bằng `bert-base-uncased`.

STS-B chỉ dùng để đánh giá, không dùng để train.

## 6. Pipeline huấn luyện

Pipeline chính nằm ở:

```text
src/train.py
```

Quy trình:

1. Load tokenizer `bert-base-uncased`.
2. Load student SFT-BE.
3. Load teacher `all-mpnet-base-v2`.
4. Load Wikipedia cache.
5. Chia train/validation bằng split quyết định, không trùng dữ liệu.
6. Train student bằng cosine distillation loss.
7. Sau mỗi epoch, đánh giá:
   - Held-out Wikipedia distillation loss.
   - STS-B Spearman correlation.
8. Lưu checkpoint:
   - `stage0_latest.pt`
   - `stage0_final.pt`

Lệnh chạy:

```bash
python3 src/prepare_data.py --cache-dir ./data_cache
python3 src/train.py
python3 src/evaluate.py
```

## 7. Vai trò trong hệ thống document similarity

Mô hình SFT-BE đóng vai trò tầng embedding/ranking ban đầu:

```text
document text -> SFT-BE encoder -> vector database -> top-k candidates
```

Sau đó hệ thống có thể dùng một tầng mạnh hơn, ví dụ LLM hoặc verifier, để đọc các đoạn top-k và trả lời người dùng. Đây là cách chia việc hợp lý:

- SFT-BE: truy xuất nhanh, rẻ, phù hợp với lượng tài liệu lớn.
- LLM/verifier: kiểm tra ý nghĩa chi tiết, suy luận, loại kết quả sai.

## 8. Giới hạn

Mô hình 46M tham số không nên được trình bày như một mô hình hiểu ngữ nghĩa sâu ngang LLM. Thực nghiệm cho thấy mô hình mạnh hơn ở phân tách chủ đề và tìm vùng tài liệu liên quan, nhưng yếu ở các quan hệ cần suy luận chính xác như:

- đúng/sai,
- có/không,
- đảo vai chủ thể và đối tượng,
- nhập nhằng nghĩa của từ theo ngữ cảnh rất hẹp.

Vì vậy kết luận kỹ thuật hợp lý là: SFT-BE là tầng truy xuất embedding nhẹ cho document similarity, không phải tầng reasoning cuối cùng.

## 9. Cấu trúc source hiện tại

```text
src/
  config.py          cấu hình model và training
  prepare_data.py    tạo cache Wikipedia và STS-B
  dataset.py         dataset và dataloader
  losses.py          Stage0TeacherDistillationLoss
  train.py           huấn luyện Stage 0
  evaluate.py        đánh giá stage0_final.pt trên STS-B
  model/
    embedding.py
    attention.py
    encoder_blocks.py
    encoder.py
```

Source hiện tại đã được rút gọn để phục vụ báo cáo: không còn budget logic, debug script, monitor gateway, BM25 baseline, Stage 1 hoặc Stage 2.
