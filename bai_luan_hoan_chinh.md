**BỘ GIÁO DỤC VÀ ĐÀO TẠO**

**TRƯỜNG ĐẠI HỌC CÔNG NGHỆ KỸ THUẬT TP. HỒ CHÍ MINH**

**KHOA CÔNG NGHỆ THÔNG TIN**

<br>

**TIỂU LUẬN KẾT THÚC HỌC PHẦN**

**HỌC KỲ II/2025-2026**

**XỬ LÝ NGÔN NGỮ TỰ NHIÊN**

<br>

**XÂY DỰNG CÔNG CỤ TÌM KIẾM VĂN BẢN TƯƠNG TỰ**

<br>

**GVHD:** ThS. Đoàn Minh Trí  
**MÃ HP:** 252NLPR431585_01CLC  
**NHÓM SINH VIÊN THỰC HIỆN:** Nhóm 7  

<br>

**TP. Hồ Chí Minh, tháng 6 năm 2026**

---

**DANH SÁCH CÁC THÀNH VIÊN THAM GIA TIỂU LUẬN**

**Học kỳ II năm học 2025-2026**  
**Chủ đề:** Xây dựng công cụ tìm kiếm văn bản tương tự

| STT | Họ tên sinh viên | MSSV | Kết quả |
|:--:|:--|:--|:--:|
| 1 | Phạm Đăng Quang | 23110143 | 100% |
| 2 | Trần Đức Trường | 23110164 | 100% |
| 3 | Trần Minh Huy | 23110 | 100% |
| 4 | Phạm Công Trường | 231101 | 100% |
| 5 | Lương Nguyễn Thành Hưng |  | 100% |

**Ghi chú:** Tỉ lệ phần trăm thể hiện mức độ tham gia vào quá trình tìm hiểu lý thuyết, xây dựng mô hình, kiểm thử hệ thống và hoàn thiện báo cáo.

---

**ĐÁNH GIÁ, NHẬN XÉT CỦA GIẢNG VIÊN CHẤM ĐIỂM**

................................................................................................................  
................................................................................................................  
................................................................................................................  
................................................................................................................  

Ngày ...... tháng ...... năm ......

Giảng viên chấm điểm

---

**LỜI CẢM ƠN**

Trong quá trình thực hiện tiểu luận kết thúc học phần Xử lý ngôn ngữ tự nhiên, nhóm chúng em đã có cơ hội tiếp cận một bài toán vừa có nền tảng lý thuyết rõ ràng, vừa có khả năng ứng dụng thực tế: tìm kiếm và so sánh văn bản dựa trên ý nghĩa. Đề tài này giúp nhóm hiểu rằng một hệ thống NLP hoàn chỉnh không chỉ dừng lại ở việc gọi một mô hình có sẵn, mà còn cần một chuỗi xử lý gồm chuẩn bị dữ liệu, thiết kế kiến trúc, huấn luyện, đánh giá, lập chỉ mục, truy xuất, xếp hạng lại và trình bày bằng chứng cho người dùng.

Nhóm chúng em xin gửi lời cảm ơn chân thành đến ThS. Đoàn Minh Trí, giảng viên phụ trách học phần, đã truyền đạt các kiến thức nền tảng về biểu diễn văn bản, mô hình ngôn ngữ, Transformer, embedding và các bài toán truy xuất thông tin. Những kiến thức này là cơ sở để nhóm lựa chọn hướng tiếp cận Bi-Encoder cho tầng truy xuất nhanh, đồng thời kết hợp thêm Cross-Encoder và cơ chế verifier để cải thiện chất lượng xếp hạng trong hệ thống tìm kiếm tài liệu.

Trong suốt quá trình làm bài, nhóm nhận thấy sự khác biệt lớn giữa việc tạo ra một mô hình chạy được và việc xây dựng một hệ thống có thể giải thích được kết quả. Vì vậy, ngoài việc xây dựng mô hình SFT-BE, nhóm cũng chú trọng đến metadata, citation, context window và cách đánh giá trung thực dựa trên những số liệu thực sự có trong checkpoint và log huấn luyện. Do thời gian thực hiện còn hạn chế, báo cáo và hệ thống chắc chắn vẫn còn những điểm có thể cải thiện. Nhóm rất mong nhận được góp ý từ thầy để tiếp tục hoàn thiện trong các môn học và đồ án sau.

Nhóm chúng em xin chân thành cảm ơn thầy.

---

**MỤC LỤC**

Mục lục tự động được tạo trong file DOCX.

---

# CHƯƠNG 1. TỔNG QUAN BÀI TOÁN

## 1.1. Lý do chọn đề tài

Trong học tập, nghiên cứu và công việc hằng ngày, người dùng thường xuyên phải làm việc với nhiều loại tài liệu văn bản như giáo trình, báo cáo, bài báo khoa học, tài liệu kỹ thuật, hợp đồng, ghi chú hoặc các tập tin PDF được tải từ Internet. Khi số lượng tài liệu còn nhỏ, người dùng có thể mở từng file và dùng chức năng tìm kiếm từ khóa như Ctrl + F. Tuy nhiên, khi số lượng tài liệu tăng lên, cách làm này nhanh chóng bộc lộ hạn chế. Người dùng không phải lúc nào cũng nhớ chính xác cụm từ xuất hiện trong tài liệu, trong khi cùng một ý nghĩa có thể được diễn đạt bằng nhiều cách khác nhau.

Ví dụ, một người dùng muốn tìm nội dung liên quan đến "ảnh hưởng của mạng xã hội đến hành vi chính trị". Trong tài liệu, nội dung tương ứng có thể được viết là "political engagement on online platforms", "partisan news exposure in social media" hoặc "digital platforms shape political participation". Những cụm từ này không trùng khớp hoàn toàn về mặt từ vựng, nhưng lại gần nhau về ý nghĩa. Một hệ thống tìm kiếm thuần từ khóa có thể bỏ sót các kết quả quan trọng, vì nó chỉ đo sự trùng lặp bề mặt giữa truy vấn và văn bản.

Từ hạn chế đó, tìm kiếm ngữ nghĩa trở thành một hướng tiếp cận phù hợp hơn. Thay vì biểu diễn văn bản bằng các vector thưa dựa trên số lần xuất hiện của từ, hệ thống dùng mô hình neural encoder để ánh xạ câu, đoạn văn hoặc document chunk thành vector dense trong một không gian embedding. Khi hai nội dung gần nhau về nghĩa, vector của chúng có xu hướng cùng hướng hoặc nằm gần nhau hơn. Độ gần này có thể đo bằng cosine similarity, từ đó cho phép hệ thống tìm ra các đoạn liên quan ngay cả khi không trùng nhiều từ khóa.

Đề tài "Xây dựng công cụ tìm kiếm văn bản tương tự" được chọn vì có tính thực tế và phù hợp với nội dung môn Xử lý ngôn ngữ tự nhiên. Về mặt lý thuyết, đề tài liên quan trực tiếp đến tiền xử lý văn bản, tokenization, sentence embedding, Transformer, Bi-Encoder, Cross-Encoder và các chỉ số đánh giá semantic similarity. Về mặt triển khai, đề tài yêu cầu xây dựng một pipeline có thể chạy end-to-end: đọc tài liệu, chia nhỏ nội dung, encode chunk thành vector, lưu chỉ mục, truy vấn theo vector, xếp hạng lại kết quả và hiển thị citation cho người dùng.

Điểm quan trọng của đề tài là không xem mô hình embedding như một hộp đen duy nhất. Trong hệ thống thực tế, mỗi loại mô hình có vai trò riêng. Bi-Encoder phù hợp với tầng truy xuất đầu tiên vì có thể encode tài liệu trước và tìm kiếm nhanh. Cross-Encoder phù hợp với reranking vì đọc đồng thời query và passage, cho chất lượng đánh giá từng cặp tốt hơn nhưng chi phí cao hơn. LLM verifier có thể dùng như một tầng kiểm tra tùy chọn để đánh giá lại bằng chứng, đặc biệt trong các truy vấn có phủ định, con số, địa danh hoặc quan hệ chủ thể - đối tượng. Vì vậy, đề tài hướng đến một kiến trúc nhiều tầng thay vì chỉ dùng một phép cosine similarity đơn lẻ.

## 1.2. Mục tiêu đề tài

Mục tiêu tổng quát của đề tài là xây dựng một hệ thống tìm kiếm văn bản tương tự dựa trên ngữ nghĩa, có khả năng lập chỉ mục một tập tài liệu cục bộ và trả về các đoạn văn bản liên quan nhất với truy vấn của người dùng. Hệ thống cần vừa thể hiện được kiến thức mô hình học sâu, vừa có khả năng chạy thử trong môi trường local.

Các mục tiêu cụ thể gồm:

- Xây dựng mô hình SFT-BE (*Shallow Factorized Transformer Bi-Encoder*) bằng PyTorch, trong đó mỗi văn bản được encode độc lập thành embedding 768 chiều.
- Thiết kế kiến trúc Transformer encoder gọn hơn BERT-base, sử dụng factorized embedding 128 -> 768, sinusoidal positional encoding, 6 encoder block Pre-LN, 12 attention heads và mean pooling có attention mask.
- Huấn luyện mô hình bằng teacher-student distillation, sử dụng teacher `sentence-transformers/all-mpnet-base-v2` để tạo embedding mục tiêu cho student.
- Chuẩn bị dữ liệu huấn luyện từ Wikipedia sentence-level và dữ liệu đánh giá từ STS-B.
- Xây dựng hệ thống retrieval phía trên SFT-BE, gồm chunking tài liệu, encoding, lưu vector local, tìm kiếm top-k, lexical boost, Cross-Encoder reranking và LLM verification tùy chọn.
- Thiết kế luồng truy vấn để người dùng có thể nhập câu hỏi tự nhiên, xem điểm xếp hạng, citation và ngữ cảnh của kết quả.
- Đánh giá trung thực dựa trên số liệu thực nghiệm: training loss, cosine với teacher trên 10MB cuối Wikipedia, số lượng chunk trong chỉ mục paper, shape ma trận vector và kiểm thử logic retrieval pipeline.

## 1.3. Đối tượng và phạm vi nghiên cứu

Đối tượng nghiên cứu của đề tài là các phương pháp biểu diễn văn bản và truy xuất thông tin dựa trên ngữ nghĩa. Trọng tâm của đề tài là mô hình Bi-Encoder tự cài đặt và cách sử dụng nó trong một hệ thống document retrieval. Ngoài ra, đề tài cũng xem xét vai trò của Cross-Encoder và LLM verifier như các tầng tăng chất lượng sau bước truy xuất ban đầu.

Về phạm vi dữ liệu, mô hình được huấn luyện trên Wikipedia tiếng Anh ở mức câu. Tập STS-B được dùng để đánh giá semantic textual similarity. Phần kiểm tra truy xuất sử dụng 30 file PDF thuộc 6 nhóm chủ đề gồm culture, linguistics, literature, politics, psychology và sociology. Đây là các tài liệu dạng paper, phù hợp để kiểm tra khả năng truy xuất theo nội dung học thuật.

Về phạm vi định dạng tài liệu, retrieval system hiện hỗ trợ `.txt`, `.md`, `.rst` và `.pdf` có thể trích xuất text trực tiếp. Hệ thống chưa xử lý trực tiếp file DOCX trong pipeline retrieval, mặc dù ở mức thao tác ngoài hệ thống có thể dùng công cụ chuyển đổi để đọc DOCX. Đối với PDF scan không có text layer, hệ thống chưa tích hợp OCR, vì vậy không thể đảm bảo trích xuất nội dung đúng.

Về phạm vi ứng dụng, hệ thống được triển khai ở mức thử nghiệm cục bộ. Chỉ mục vector được thiết kế cho quy mô nhỏ đến trung bình nhằm chứng minh pipeline truy xuất ngữ nghĩa, chưa hướng đến triển khai nhiều người dùng đồng thời, chưa có authentication, chưa có quản lý chỉ mục theo user và chưa dùng vector database chuyên dụng như FAISS, Chroma hoặc Milvus.

## 1.4. Đóng góp chính của đề tài

Đề tài có bốn đóng góp chính.

Thứ nhất, nhóm xây dựng mô hình SFT-BE bằng PyTorch thay vì chỉ gọi trực tiếp một embedding model có sẵn. Mô hình có 46,536,960 tham số, sử dụng kiến trúc encoder-only, output embedding 768 chiều và được thiết kế để đóng vai trò retriever.

Thứ hai, nhóm triển khai pipeline huấn luyện teacher-student distillation. Teacher là `sentence-transformers/all-mpnet-base-v2`, student là SFT-BE. Loss được định nghĩa dựa trên cosine similarity giữa embedding student và embedding teacher sau L2 normalization. Cách huấn luyện này cho phép student học lại cấu trúc embedding space của teacher mà không cần nhãn thủ công cho từng cặp câu.

Thứ ba, nhóm xây dựng hệ thống retrieval nhiều tầng. Sau khi SFT-BE lấy top-k ứng viên bằng vector similarity, hệ thống bổ sung lexical boost, Cross-Encoder reranking và LLM verifier tùy chọn. Thiết kế này phản ánh đúng cách các hệ thống tìm kiếm ngữ nghĩa hiện đại thường hoạt động: dùng mô hình nhanh để lọc rộng, sau đó dùng mô hình chính xác hơn để xếp hạng nhóm nhỏ.

Thứ tư, nhóm lưu và hiển thị citation cho từng kết quả. Mỗi chunk trong chỉ mục có metadata như paper, section, page và vị trí chunk. Điều này giúp người dùng kiểm tra được nguồn của kết quả, đồng thời làm nền tảng cho answer generation có trích dẫn.

---

# CHƯƠNG 2. CƠ SỞ LÝ THUYẾT VÀ CÔNG NGHỆ NỀN TẢNG

## 2.1. Tổng quan về xử lý ngôn ngữ tự nhiên

Xử lý ngôn ngữ tự nhiên, hay Natural Language Processing, là lĩnh vực nghiên cứu các phương pháp giúp máy tính xử lý, phân tích và khai thác thông tin từ ngôn ngữ của con người. Dữ liệu ngôn ngữ tự nhiên thường xuất hiện dưới dạng câu, đoạn văn, tài liệu, hội thoại, bình luận hoặc bài báo. Khác với dữ liệu dạng bảng, văn bản không có cấu trúc cố định, có nhiều nhập nhằng và phụ thuộc mạnh vào ngữ cảnh.

Một đặc điểm quan trọng của ngôn ngữ là cùng một ý nghĩa có thể được diễn đạt bằng nhiều cách khác nhau. Ví dụ, "increase model accuracy", "improve prediction performance" và "enhance classification quality" đều có thể nói về việc cải thiện chất lượng mô hình, nhưng mức độ trùng từ giữa các câu không cao. Ngược lại, hai câu có thể trùng nhiều từ nhưng khác nghĩa do phủ định hoặc đảo vai chủ thể - đối tượng, ví dụ "the dog bites the cat" và "the cat bites the dog". Điều này khiến các phương pháp dựa hoàn toàn trên từ khóa gặp nhiều hạn chế.

Trong các hệ thống NLP hiện đại, một bước quan trọng là biểu diễn văn bản thành vector. Vector này có thể là sparse vector dựa trên từ vựng, hoặc dense vector do mô hình học sâu sinh ra. Khi văn bản đã được biểu diễn thành vector, máy tính có thể tính toán độ gần, phân loại, gom cụm, truy xuất hoặc xếp hạng.

Trong đề tài này, NLP được ứng dụng vào bài toán tìm kiếm văn bản tương tự. Hệ thống không cố gắng sinh văn bản tự do ngay từ đầu, mà trước hết tập trung vào việc đưa query và document chunk vào cùng một không gian embedding. Sau đó, hệ thống dùng cosine similarity, reranking và verifier để tìm các đoạn văn bản có khả năng trả lời nhu cầu thông tin của người dùng.

## 2.2. Bài toán truy xuất thông tin trong văn bản

Truy xuất thông tin (*Information Retrieval*) là bài toán tìm những tài liệu hoặc đoạn văn bản liên quan nhất đến một truy vấn. Một hệ thống retrieval thường gồm hai giai đoạn lớn: lập chỉ mục và truy vấn.

Ở giai đoạn lập chỉ mục, hệ thống xử lý trước corpus. Tài liệu được đọc, làm sạch, chia thành đơn vị nhỏ hơn, sau đó được biểu diễn dưới dạng chỉ mục. Chỉ mục có thể là inverted index trong các hệ thống keyword search hoặc vector index trong các hệ thống semantic search. Với semantic search, mỗi chunk thường được lưu cùng embedding và metadata để phục vụ truy xuất, xếp hạng và trích dẫn nguồn.

Ở giai đoạn searching, người dùng nhập query. Hệ thống biểu diễn query theo cùng cách với tài liệu, tính điểm phù hợp giữa query và từng đơn vị trong corpus, sau đó trả về top-k kết quả. Nếu corpus lớn, hệ thống thường dùng approximate nearest neighbor để tăng tốc. Trong phạm vi thử nghiệm, số chunk là 3,402 nên có thể tính trực tiếp tích vô hướng giữa query vector và toàn bộ ma trận vector.

Một điểm cần phân biệt là retrieval không nhất thiết phải trả lời trực tiếp câu hỏi. Nhiệm vụ đầu tiên của retrieval là tìm đúng vùng tài liệu có khả năng chứa thông tin. Nếu cần câu trả lời tự nhiên, hệ thống có thể dùng thêm một tầng đọc hiểu hoặc LLM phía sau. Vì vậy, trong đề tài này, SFT-BE được xem là tầng truy xuất ban đầu, còn Cross-Encoder và LLM verifier là các tầng tăng chất lượng sau đó.

## 2.3. Tìm kiếm ngữ nghĩa

Tìm kiếm ngữ nghĩa (*semantic search*) là hướng tiếp cận trong đó hệ thống tìm kết quả dựa trên sự gần nhau về ý nghĩa thay vì chỉ dựa trên sự trùng khớp bề mặt của từ khóa. Ý tưởng chính là ánh xạ query và văn bản vào một không gian vector sao cho các nội dung gần nghĩa nằm gần nhau.

Gọi $q$ là truy vấn và $d_i$ là document chunk thứ $i$. Một encoder $f_\theta$ ánh xạ văn bản thành vector:

$$z_q = f_\theta(q), \qquad z_i = f_\theta(d_i)$$

Độ tương đồng giữa query và chunk được tính bằng cosine similarity:

$$\operatorname{sim}(q,d_i)=\operatorname{cos}(z_q,z_i)=\frac{z_q^\top z_i}{\|z_q\|_2\|z_i\|_2}$$

Nếu cả hai vector đã được L2-normalize:

$$\hat{z}_q=\frac{z_q}{\|z_q\|_2}, \qquad \hat{z}_i=\frac{z_i}{\|z_i\|_2}$$

thì cosine similarity trở thành dot product:

$$\operatorname{cos}(z_q,z_i)=\hat{z}_q^\top \hat{z}_i$$

Trong hệ thống đề xuất, SFT-BE sinh embedding cho query và chunk, sau đó embedding được normalize. Khi tìm kiếm, hệ thống chỉ cần tính:

$$s = Z\hat{z}_q$$

trong đó $Z \in \mathbb{R}^{N \times 768}$ là ma trận embedding của toàn bộ chunk đã normalize.

Ưu điểm của semantic search là khả năng tìm các đoạn liên quan dù không trùng từ khóa. Tuy nhiên, semantic search không phải lúc nào cũng đủ chính xác cho các truy vấn chi tiết. Vì embedding thường nén cả câu hoặc đoạn thành một vector duy nhất, một số khác biệt nhỏ như phủ định, con số hoặc đảo vai chủ thể có thể bị làm mờ. Đây là lý do hệ thống cần thêm Cross-Encoder reranking.

## 2.4. Độ tương đồng văn bản và cosine similarity

Độ tương đồng văn bản là mức độ giống nhau hoặc liên quan giữa hai đơn vị văn bản. Trong bài toán document similarity, hai đơn vị này có thể là hai câu, hai đoạn văn, hai tài liệu hoặc một query và một document chunk. Tùy mục tiêu, "tương đồng" có thể được hiểu theo nhiều cách: trùng nội dung, cùng chủ đề, entailment, paraphrase hoặc cùng trả lời một nhu cầu thông tin.

Cosine similarity là một độ đo phổ biến khi làm việc với embedding. Nó đo góc giữa hai vector thay vì độ dài tuyệt đối. Nếu hai vector cùng hướng, cosine gần 1. Nếu vuông góc, cosine gần 0. Nếu ngược hướng, cosine có thể âm. Trong sentence embedding, hướng vector thường quan trọng hơn độ lớn, vì độ lớn có thể phụ thuộc vào phân phối activation của mô hình.

Với hai embedding $u$ và $v$:

$$\operatorname{cos}(u,v)=\frac{\sum_{k=1}^{d}u_kv_k}{\sqrt{\sum_{k=1}^{d}u_k^2}\sqrt{\sum_{k=1}^{d}v_k^2}}$$

Trong SFT-BE, $d=768$. Khi dùng cosine cho retrieval, ta có thể xếp hạng tất cả chunk theo:

$$\operatorname{rank}(d_i|q)=\operatorname{argsort}_i(-\operatorname{cos}(f(q),f(d_i)))$$

Tuy nhiên, cosine similarity chỉ phản ánh mức độ gần trong embedding space. Nếu embedding model chưa học tốt một loại quan hệ nào đó, score có thể cao nhưng kết quả chưa thật sự đúng. Ví dụ, hai đoạn cùng nhắc đến một paper và cùng chủ đề "political engagement" có thể gần nhau, nhưng một đoạn nói về nguyên nhân còn đoạn kia nói về hạn chế. Vì vậy, cosine thường được dùng để lấy candidate rộng trước khi rerank.

## 2.5. Biểu diễn văn bản bằng sparse vector và dense vector

Các phương pháp biểu diễn văn bản truyền thống như Bag-of-Words, TF-IDF và BM25 tạo ra vector thưa. Mỗi chiều thường tương ứng với một token hoặc term trong vocabulary. Nếu một từ không xuất hiện trong văn bản, giá trị ở chiều đó bằng 0. TF-IDF gán trọng số cao cho những từ xuất hiện nhiều trong một văn bản nhưng không quá phổ biến trong toàn bộ corpus:

$$\operatorname{TFIDF}(t,d)=\operatorname{TF}(t,d)\times \operatorname{IDF}(t)$$

Trong đó:

$$\operatorname{IDF}(t)=\log\frac{N}{1+\operatorname{df}(t)}$$

Sparse vector có ưu điểm là dễ giải thích và nhanh. Nếu query chứa một thuật ngữ hiếm như tên paper, tên tác giả hoặc mã tài liệu, lexical matching có thể rất hiệu quả. Tuy nhiên, sparse vector không hiểu tốt đồng nghĩa và ngữ cảnh.

Dense vector, hay embedding, là vector có số chiều cố định và hầu hết giá trị khác 0. Vector này được học bởi mô hình neural network. Thay vì mỗi chiều tương ứng trực tiếp với một từ, embedding mã hóa nhiều đặc trưng ngữ nghĩa ẩn. Trong đề tài, SFT-BE sinh embedding 768 chiều cho mỗi câu hoặc chunk. Embedding này không dễ giải thích từng chiều, nhưng hữu ích cho semantic search.

Hệ thống trong đề tài kết hợp cả hai hướng. Tầng chính là dense vector retrieval bằng SFT-BE. Sau đó, pipeline áp dụng lexical boost với trọng số 0.18 để giữ lại một phần tín hiệu từ khóa. Công thức kết hợp điểm được dùng là:

$$\operatorname{final\_score}=0.82\times \operatorname{vector\_score}+0.18\times \operatorname{lexical\_overlap}$$

Thiết kế này hợp lý vì semantic embedding giúp tìm nội dung gần nghĩa, còn lexical overlap giúp hệ thống không bỏ qua các từ khóa đặc biệt như tên riêng, mã paper, thuật ngữ hiếm hoặc con số.

## 2.6. Transformer Encoder trong mô hình ngôn ngữ

Transformer Encoder là kiến trúc nền tảng của nhiều mô hình NLP hiện đại. Thay vì xử lý chuỗi tuần tự như RNN, encoder nhận toàn bộ dãy token cùng lúc, gắn thêm thông tin vị trí và cập nhật representation qua nhiều block. Mỗi block thường gồm hai thành phần: cơ chế chú ý để các token trao đổi thông tin ngữ cảnh, và mạng feed-forward để biến đổi representation tại từng vị trí.

Trong bài toán biểu diễn văn bản, Transformer Encoder phù hợp vì nó tạo ra hidden state theo ngữ cảnh. Một token không còn được biểu diễn như một embedding tĩnh, mà thay đổi theo các token xung quanh. Ví dụ, từ "bank" trong "river bank" và "investment bank" có thể nhận representation khác nhau vì ngữ cảnh hai câu khác nhau.

Ở mức tổng quát, nếu input sau embedding là $X \in \mathbb{R}^{B\times L\times H}$, encoder biến đổi tuần tự qua $L_e$ lớp:

$$H^{(0)}=X,\qquad H^{(\ell+1)}=\operatorname{EncoderBlock}_{\ell}(H^{(\ell)})$$

Sau lớp cuối, mô hình có thể lấy hidden state của từng token hoặc gom chúng thành một vector câu. Trong SFT-BE, phần gom vector dùng mean pooling có attention mask; trong Cross-Encoder, vector ở vị trí `[CLS]` được dùng cho classification head. Vì phần kiến trúc chi tiết của SFT-BE được trình bày ở Chương 4, mục này chỉ nêu vai trò nền tảng của Transformer Encoder để tránh lặp lại công thức.

## 2.7. Bi-Encoder trong bài toán document similarity

Bi-Encoder là kiến trúc trong đó hai văn bản được encode độc lập thành hai vector, sau đó so sánh hai vector bằng một độ đo như cosine similarity. Với hai văn bản $x_a$ và $x_b$:

$$z_a=f_\theta(x_a), \qquad z_b=f_\theta(x_b)$$

Độ tương đồng:

$$\operatorname{sim}(x_a,x_b)=\operatorname{cos}(z_a,z_b)$$

Nếu hai nhánh encoder chia sẻ cùng tham số, kiến trúc này được gọi là Siamese Bi-Encoder. Việc chia sẻ tham số buộc mọi văn bản đi vào cùng một embedding space. Khi đó, phép so sánh cosine giữa query và document mới có ý nghĩa nhất quán.

Ưu điểm lớn nhất của Bi-Encoder là khả năng precompute document embedding. Với corpus:

$$\mathcal{D}=\{d_1,d_2,\ldots,d_N\}$$

ta có thể tính trước:

$$Z_\mathcal{D}=\{f_\theta(d_1),f_\theta(d_2),\ldots,f_\theta(d_N)\}$$

Khi có query mới $q$, hệ thống chỉ cần encode query một lần:

$$z_q=f_\theta(q)$$

rồi tìm:

$$\operatorname{TopK}(q)=\underset{d_i\in\mathcal{D}}{\operatorname{arg\,topK}}\ \operatorname{cos}(z_q,f_\theta(d_i))$$

Nhược điểm của Bi-Encoder là query và document không tương tác trực tiếp ở mức token trong quá trình encode. Vì vậy, Bi-Encoder có thể bỏ sót các quan hệ tinh vi. Trong hệ thống của đề tài, Bi-Encoder được dùng đúng với vai trò retrieval layer: nhanh, có thể mở rộng và chọn ra nhóm ứng viên trước khi chuyển sang Cross-Encoder.

## 2.8. Cross-Encoder và reranking

Cross-Encoder nhận query và passage trong cùng một input, thường ở dạng `[CLS] query [SEP] passage [SEP]`.

Do query và passage cùng đi qua Transformer, token của query có thể attention trực tiếp tới token của passage. Điều này giúp Cross-Encoder đánh giá mối quan hệ giữa hai văn bản chi tiết hơn Bi-Encoder. Tuy nhiên, Cross-Encoder không thể precompute embedding cho passage độc lập, vì output phụ thuộc vào cả query. Với một query và $N$ chunk, Cross-Encoder phải chạy $N$ lần:

$$\operatorname{cost}_{cross}=O(N\times \operatorname{forward}_{transformer})$$

Trong khi Bi-Encoder chỉ cần encode query một lần nếu document embedding đã có:

$$\operatorname{cost}_{bi}=O(\operatorname{forward}_{query}+N\times d)$$

Do đó, Cross-Encoder thường được dùng ở bước reranking. Hệ thống trước hết dùng Bi-Encoder lấy top 100 candidate, sau đó dùng Cross-Encoder rerank các candidate này và giữ top 10. Model Cross-Encoder được chọn trong đề tài là `cross-encoder/ms-marco-MiniLM-L6-v2`.

Raw score của Cross-Encoder được đưa qua sigmoid:

$$\sigma(x)=\frac{1}{1+e^{-x}}$$

rồi gán thành `cross_score`. Sau reranking, `final_score` được đặt bằng `cross_score`.

## 2.9. Teacher-student distillation

Teacher-student distillation là phương pháp huấn luyện trong đó một mô hình nhỏ hơn học bắt chước đầu ra của một mô hình mạnh hơn. Trong đề tài, teacher là `sentence-transformers/all-mpnet-base-v2`, còn student là SFT-BE. Teacher đã được huấn luyện trước để tạo sentence embedding chất lượng tốt. Student học cách sinh embedding gần với teacher.

Với câu đầu vào $x_i$, teacher sinh:

$$t_i=T(x_i)$$

Student sinh:

$$s_i=S_\theta(x_i)$$

Cả hai vector đều có 768 chiều. Trước khi tính loss, hai vector được normalize:

$$\hat{s}_i=\frac{s_i}{\|s_i\|_2}, \qquad \hat{t}_i=\frac{t_i}{\|t_i\|_2}$$

Loss cho một batch:

$$\mathcal{L}_{batch}=1-\frac{1}{B}\sum_{i=1}^{B}\hat{s}_i^\top \hat{t}_i$$

Tối thiểu hóa loss này tương đương tối đa hóa cosine similarity giữa student và teacher:

$$\min_\theta \mathcal{L}_{batch}\Longleftrightarrow \max_\theta \frac{1}{B}\sum_{i=1}^{B}\operatorname{cos}(S_\theta(x_i),T(x_i))$$

Điểm quan trọng là teacher không được update. Teacher chỉ sinh embedding mục tiêu, còn student nhận gradient và được cập nhật tham số. Nhờ vậy, quá trình huấn luyện tập trung vào việc đưa không gian embedding của student đến gần teacher.

## 2.10. Các độ đo đánh giá

Đề tài sử dụng các nhóm đánh giá sau.

Thứ nhất là cosine similarity giữa student và teacher trong distillation. Vì loss được định nghĩa là:

$$\mathcal{L}=1-\operatorname{cos}(student,teacher)$$

nên loss càng thấp đồng nghĩa student càng gần teacher trong embedding space.

Thứ hai là Spearman correlation trên STS-B. Với mỗi cặp câu, mô hình tính cosine similarity giữa hai embedding. Spearman đo tương quan thứ hạng giữa score dự đoán và score do con người gán. Đây là chỉ số phù hợp vì trong semantic similarity, thứ tự tương đối giữa các cặp câu thường quan trọng hơn giá trị tuyệt đối.

Thứ ba là các chỉ số retrieval như Precision@k, Recall@k, MRR và nDCG. Tuy nhiên, để tính các chỉ số này cần có tập query và nhãn chunk liên quan. Đề tài hiện chưa có bộ nhãn retrieval end-to-end cho 30 paper PDF. Vì vậy, báo cáo không tự bịa Precision@k hay MRR cho chỉ mục paper, mà chỉ trình bày những số liệu thực sự có: training loss, test cosine trên 10MB cuối Wikipedia, số chunk và kiểm thử logic pipeline.

---

# CHƯƠNG 3. MÔ TẢ BÀI TOÁN VÀ DATASET

## 3.1. Bài toán đặt ra

Bài toán của đề tài là xây dựng một hệ thống có thể nhận một tập tài liệu cục bộ và trả về các đoạn văn bản liên quan nhất với truy vấn ngôn ngữ tự nhiên của người dùng. Tài liệu đầu vào được chia thành các chunk. Mỗi chunk được encode thành vector 768 chiều bằng SFT-BE và lưu vào vector store. Khi có query, hệ thống encode query, tính similarity với toàn bộ chunk, lấy top-k candidate, sau đó rerank.

Gọi tập tài liệu là:

$$\mathcal{D}=\{D_1,D_2,\ldots,D_n\}$$

Sau khi chia nhỏ, ta có tập chunk:

$$\mathcal{C}=\{c_1,c_2,\ldots,c_m\}$$

Mỗi chunk được encode:

$$z_i=f_\theta(c_i), \qquad z_i\in\mathbb{R}^{768}$$

Với query $q$:

$$z_q=f_\theta(q)$$

Bước retrieval đầu tiên:

$$\mathcal{R}_0=\operatorname{TopK}_{c_i\in\mathcal{C}}\operatorname{cos}(z_q,z_i)$$

Sau đó, các candidate trong $\mathcal{R}_0$ đi qua lexical boost, Cross-Encoder reranking và LLM verifier tùy chọn để tạo tập kết quả cuối:

$$\mathcal{R}_{final}=\operatorname{Rerank}(\mathcal{R}_0,q)$$

## 3.2. Input và output của hệ thống

Input của hệ thống gồm hai loại.

Loại thứ nhất là input cho quá trình lập chỉ mục. Người dùng cung cấp một hoặc nhiều tài liệu. Hệ thống trích xuất văn bản, chia chunk, tạo embedding và lưu chỉ mục. Các định dạng được xét trong phạm vi đề tài là `.txt`, `.md`, `.rst` và `.pdf`.

Loại thứ hai là input cho quá trình truy vấn. Người dùng nhập một câu hỏi hoặc câu truy vấn. Ngoài query, người dùng có thể cấu hình:

- `vector_top_k`: số candidate lấy từ vector search, mặc định 100.
- `cross_rerank_k`: số candidate giữ sau Cross-Encoder, mặc định 10.
- `llm_rerank_k`: số candidate đưa vào LLM verifier, mặc định 8.
- `final_k`: số kết quả cuối cùng, mặc định 5.
- `use_llm`: có dùng LLM verifier hay không.
- `answer`: có sinh câu trả lời từ context hay không.
- `show_context`: có hiển thị context window hay không.

Output của hệ thống là danh sách kết quả đã xếp hạng. Mỗi kết quả gồm text chunk, score tổng hợp, các score thành phần nếu có, citation, page, section và vị trí chunk. Nếu người dùng yêu cầu answer, hệ thống trả thêm câu trả lời được sinh từ các context đã truy xuất.

## 3.3. Ví dụ minh họa bài toán

Giả sử trong corpus có đoạn: "Political engagement on online platforms is influenced by exposure to partisan news." Người dùng nhập: "How does social media affect political behavior?"

Tìm kiếm từ khóa có thể không đánh giá cao đoạn này vì query dùng "social media" và "political behavior", còn tài liệu dùng "online platforms" và "political engagement". Tuy nhiên, nếu embedding model học được rằng các cụm này gần nhau về ý nghĩa, cosine similarity giữa query và chunk vẫn có thể cao.

Một ví dụ khác là từ "Python". Nếu query là "programming language for machine learning", hệ thống cần ưu tiên đoạn nói về Python như một ngôn ngữ lập trình, không phải đoạn nói về loài trăn trong sinh học. Ví dụ này cho thấy hệ thống phải dựa vào ngữ cảnh chứ không chỉ dựa vào chuỗi ký tự.

## 3.4. Dữ liệu huấn luyện Wikipedia sentence-level

Dữ liệu huấn luyện chính của Stage 0 được lấy từ HuggingFace dataset `wikimedia/wikipedia, 20231101.en`. Mỗi article được tách thành câu, sau đó lọc các câu có độ dài từ 20 đến 500 ký tự:

$$20 \leq \operatorname{len}(sentence) \leq 500$$

Các câu quá ngắn thường thiếu ngữ cảnh và không hữu ích cho sentence embedding. Các câu quá dài dễ bị truncate mạnh khi tokenizer giới hạn 128 token. Vì vậy, rule lọc độ dài này giúp dữ liệu ổn định hơn.

Sau khi lọc, câu được tokenize bằng `bert-base-uncased`, padding/truncation về độ dài tối đa 128 token. Output gồm `input_ids` và `attention_mask`. Cách xử lý này làm cho dữ liệu huấn luyện có độ dài ổn định và phù hợp với kiến trúc SFT-BE.

Theo training log, Stage 0 có:

| Chỉ số | Giá trị |
|:--|--:|
| Tổng số sentence sau xử lý | 90,250,685 |
| Train samples | 88,445,671 |
| Validation samples | 1,805,014 |
| Validation ratio | 0.02 |

Quy mô này đủ lớn để student học cấu trúc embedding space tổng quát từ teacher.

## 3.5. Dữ liệu đánh giá STS-B

STS-B (*Semantic Textual Similarity Benchmark*) là tập dữ liệu gồm các cặp câu được gán điểm tương đồng bởi con người. Trong đề tài, STS-B được dùng như một hướng đánh giá semantic similarity ở mức cặp câu.

Mỗi mẫu gồm `sentence1`, `sentence2` và `score`. Score gốc được chia cho 5.0 để đưa về khoảng 0 đến 1:

$$score_{norm}=\frac{score}{5}$$

Khi đánh giá, mô hình encode hai câu:

$$z_a=f_\theta(x_a), \qquad z_b=f_\theta(x_b)$$

Sau đó tính:

$$\hat{y}=\operatorname{cos}(z_a,z_b)$$

Cuối cùng, Spearman correlation giữa $\hat{y}$ và score thật được tính. Tuy nhiên, do kết quả checkpoint Stage 0 trong đề tài chủ yếu được báo cáo bằng distillation loss và test tail Wikipedia, STS-B được xem là hướng đánh giá bổ sung hơn là số liệu chính nếu không có output cụ thể từ lần chạy cuối.

## 3.6. Tập paper dùng cho đánh giá truy xuất

Tập tài liệu dùng để kiểm tra retrieval gồm 30 paper thuộc 6 nhóm chủ đề:

| Chủ đề | Số PDF |
|:--|--:|
| culture | 5 |
| linguistics | 5 |
| literature | 5 |
| politics | 5 |
| psychology | 5 |
| sociology | 5 |

Tập paper này không được dùng để train SFT-BE. Nó được dùng để kiểm tra hệ thống retrieval trên tài liệu thực tế. Khi lập chỉ mục, mỗi PDF được trích xuất text từng trang, chuẩn hóa, suy luận page/section nếu có thể, sau đó chia thành chunk.

Index đã xây dựng trong thử nghiệm có:

| Thuộc tính | Giá trị |
|:--|--:|
| Số chunk | 3,402 |
| Embedding dimension | 768 |
| Shape ma trận vector | (3402, 768) |
| Kiểu dữ liệu vector | float32 |
| Chunk max chars | 900 |
| Chunk overlap chars | 120 |
| Thời gian lập chỉ mục | 46.58 giây |

## 3.7. Tiền xử lý và chia dữ liệu

Đề tài có hai pipeline tiền xử lý khác nhau.

Pipeline cho training xử lý Wikipedia ở mức câu. Mục tiêu là tạo nhiều mẫu sentence-level để student học embedding space của teacher. Câu được tokenize và padding/truncation về 128 token. Cách này phù hợp với teacher-student distillation vì teacher cũng sinh sentence embedding.

Pipeline cho retrieval xử lý tài liệu ở mức chunk. Hệ thống chuẩn hóa xuống dòng, loại ký tự lỗi, rút gọn khoảng trắng và chia văn bản theo paragraph. Nếu paragraph quá dài, hệ thống tách tiếp theo câu. Chunk có độ dài tối đa 900 ký tự và overlap 120 ký tự.

Overlap có vai trò quan trọng. Nếu một luận điểm bị cắt ở ranh giới giữa hai chunk, phần overlap giúp chunk sau vẫn giữ được một phần ngữ cảnh của chunk trước. Điều này đặc biệt hữu ích với paper PDF, nơi một ý có thể kéo dài qua nhiều câu hoặc nhiều dòng.

---

# CHƯƠNG 4. GIẢI PHÁP THỰC HIỆN VÀ MÔ HÌNH ĐỀ XUẤT

## 4.1. Tổng quan giải pháp

Giải pháp của đề tài được thiết kế theo hướng nhiều tầng. Tầng đầu tiên là mô hình embedding SFT-BE, chịu trách nhiệm biến câu, đoạn văn hoặc document chunk thành vector 768 chiều. Tầng thứ hai là tầng truy xuất vector, dùng cosine similarity để lấy nhanh một tập ứng viên có khả năng liên quan. Tầng thứ ba là reranking, trong đó Cross-Encoder đọc đồng thời query và passage để đánh giá lại các ứng viên ở mức chi tiết hơn. Tầng cuối cùng là verifier và answer generation tùy chọn, chỉ được dùng khi cần kiểm tra bằng chứng hoặc tạo câu trả lời tự nhiên có citation.

Điểm cốt lõi của thiết kế này là tách retrieval thành hai mục tiêu khác nhau: recall ở bước đầu và precision ở bước sau. Bi-Encoder có lợi thế tốc độ vì document embedding có thể được tính trước. Cross-Encoder có lợi thế chất lượng vì nó cho phép tương tác trực tiếp giữa token của query và token của passage. Vì vậy, hệ thống không bắt một mô hình duy nhất giải quyết toàn bộ bài toán, mà phân vai cho từng thành phần theo đúng ưu điểm của chúng.

Luồng tổng thể có thể mô tả như sau: tài liệu được trích xuất text, chia chunk, encode bằng SFT-BE và lưu vào vector store; query được encode bằng cùng mô hình, tìm kiếm bằng vector similarity, cộng thêm lexical boost, rerank bằng Cross-Encoder, sau đó tùy chọn đi qua LLM verifier để tạo kết quả cuối có citation/context/answer.

Thiết kế này cũng giúp hệ thống dễ mở rộng về mặt nghiên cứu. Nếu sau này thay SFT-BE bằng một embedding model tốt hơn, các bước chunking, lập chỉ mục và reranking vẫn giữ nguyên về mặt ý tưởng. Nếu corpus lớn hơn, vector search có thể chuyển sang approximate nearest neighbor mà không làm thay đổi bản chất của mô hình embedding. Nếu cần chất lượng ranking cao hơn, có thể fine-tune Cross-Encoder hoặc thay verifier, nhưng luồng hai giai đoạn vẫn giữ vai trò nền tảng.

## 4.2. Mô hình Bi-Encoder trong bài toán Document Similarity

### 4.2.1. Kiến trúc Bi-Encoder cho biểu diễn ngữ nghĩa văn bản

Trong bài toán document similarity, mục tiêu chính không chỉ là kiểm tra hai văn bản có trùng từ khóa hay không, mà là xác định chúng có gần nhau về mặt ý nghĩa hay không. Vì vậy, thay vì biểu diễn văn bản bằng các vector thưa như Bag-of-Words hoặc TF-IDF, hướng tiếp cận hiện đại là dùng neural encoder để ánh xạ mỗi câu, đoạn văn hoặc document chunk thành một vector dense trong không gian embedding. Khi đó, hai văn bản có nội dung gần nhau sẽ nằm gần nhau trong không gian vector, và độ tương đồng có thể được đo bằng cosine similarity.

Mô hình được sử dụng trong đồ án là **SFT-BE** (*Shallow Factorized Transformer Bi-Encoder*). Đây là một kiến trúc bi-encoder dạng Siamese: hai văn bản được encode độc lập bằng cùng một encoder, sau đó so sánh hai embedding đầu ra. Luồng xử lý tổng thể có thể tóm tắt như sau:

$$\text{Text} \longrightarrow \text{Tokenizer} \longrightarrow \text{Embedding} \longrightarrow \text{Transformer Encoder} \longrightarrow \text{Mean Pooling} \longrightarrow z \in \mathbb{R}^{768}$$

Với hai văn bản $x_a$ và $x_b$, mô hình sinh ra hai vector:

$$z_a = f_\theta(x_a), \qquad z_b = f_\theta(x_b)$$

Trong đó $f_\theta$ là encoder có tham số $\theta$. Độ tương đồng giữa hai văn bản được tính bằng:

$$\operatorname{sim}(x_a,x_b) = \operatorname{cos}(z_a,z_b) = \frac{z_a^\top z_b}{\|z_a\|_2\|z_b\|_2}$$

Điểm quan trọng của bi-encoder là hai văn bản không cần đi qua mô hình cùng lúc. Điều này khác với cross-encoder: cross-encoder đưa cả hai văn bản vào mô hình trong cùng một lần xử lý, nên các token của văn bản thứ nhất có thể attention trực tiếp tới các token của văn bản thứ hai. Cách này thường cho chất lượng ranking tốt hơn, nhưng đổi lại chi phí inference rất lớn: với một query và $N$ document, mô hình phải chạy forward $N$ lần.

Bi-encoder giải quyết vấn đề này theo hướng thực tế hơn cho retrieval. Toàn bộ document trong corpus có thể được encode trước và lưu vào vector database:

$$Z_{\mathcal{D}} = \{f_\theta(d_1), f_\theta(d_2), \ldots, f_\theta(d_N)\}$$

Khi có query mới $q$, hệ thống chỉ cần encode query một lần:

$$z_q = f_\theta(q)$$

Sau đó tìm các document gần nhất bằng cosine similarity:

$$\operatorname{TopK}(q) = \underset{d_i \in \mathcal{D}}{\operatorname{arg\,topK}}\; \operatorname{cos}(z_q, f_\theta(d_i))$$

Nhờ cơ chế này, SFT-BE phù hợp với vai trò retrieval layer: nhanh, nhẹ, có thể pre-compute embedding cho corpus lớn, và đủ tốt để chọn ra nhóm tài liệu ứng viên trước khi chuyển sang các bước reranking hoặc verification phức tạp hơn.

**Siamese Bi-Encoder.** Từ "Siamese" trong kiến trúc này có nghĩa là hai nhánh encoder chia sẻ cùng một bộ trọng số. Nếu input là hai câu khác nhau, mô hình không tạo hai encoder riêng biệt mà dùng chung một encoder $f_\theta$. Cách làm này buộc mọi văn bản phải được đặt vào cùng một embedding space, nhờ đó cosine similarity giữa các vector mới có ý nghĩa nhất quán.

$$x_a \rightarrow f_\theta \rightarrow z_a, \qquad x_b \rightarrow f_\theta \rightarrow z_b$$

Nếu dùng hai encoder khác nhau, embedding của query và document có thể nằm trong hai không gian không đồng nhất, làm phép so sánh trực tiếp trở nên kém ổn định. Việc shared weights giúp mọi văn bản được biểu diễn theo cùng một quy tắc: các văn bản gần nghĩa có xu hướng cùng hướng trong embedding space, còn các văn bản ít liên quan có xu hướng nằm xa hơn theo cấu trúc mà mô hình học được.

---

### 4.2.2. Xây dựng mô hình SFT-BE

SFT-BE được thiết kế như một Transformer encoder nhỏ hơn BERT-base nhưng vẫn giữ hidden size 768. Mô hình không dùng decoder vì nhiệm vụ ở đây không phải sinh chuỗi văn bản mới, mà là đọc input và nén nó thành một embedding vector để so sánh. Lý do giữ chiều 768 là teacher model `sentence-transformers/all-mpnet-base-v2` cũng sinh embedding 768 chiều, nên student có thể học trực tiếp từ teacher embedding mà không cần thêm projection layer trung gian.

Cấu hình chính của mô hình như sau:

| Thành phần | Giá trị |
|:--|:--|
| Tokenizer | `bert-base-uncased` |
| Vocabulary size | 30,522 |
| Max sequence length | 128 |
| Factorized embedding dim | 128 |
| Hidden size | 768 |
| Transformer layers | 6 |
| Attention heads | 12 |
| Head dimension | 64 |
| FFN hidden size | 3072 |
| Dropout | 0.1 |
| Pooling | Mean pooling with attention mask |
| Output dimension | 768 |

Luồng xử lý đầy đủ của một input có thể viết gọn:

$$\text{input\_ids}, \text{attention\_mask} \rightarrow E(x) \rightarrow H^{(0)} \rightarrow H^{(1)} \rightarrow \cdots \rightarrow H^{(6)} \rightarrow \operatorname{MeanPool}(H^{(6)}) \rightarrow z$$

Trong đó:

- `input_ids` là dãy id token sau khi tokenize.
- `attention_mask` đánh dấu đâu là token thật, đâu là padding.
- $H^{(\ell)}$ là hidden states sau layer thứ $\ell$.
- $z \in \mathbb{R}^{768}$ là embedding cuối cùng của văn bản.

**Tokenization và attention mask.** Với một câu đầu vào $x$, tokenizer chuyển câu thành dãy token id:

$$x = (w_1,w_2,\ldots,w_L) \longrightarrow (t_1,t_2,\ldots,t_L)$$

Vì mỗi batch cần có cùng độ dài, các câu ngắn được padding tới `max_seq_length = 128`. Attention mask được dùng để mô hình biết vị trí nào là token thật:

$$m_i = \begin{cases}1 & \text{nếu vị trí } i \text{ là token thật}\\0 & \text{nếu vị trí } i \text{ là padding}\end{cases}$$

Mask này xuất hiện ở hai nơi quan trọng: trong self-attention để không attention vào padding token, và trong mean pooling để padding không ảnh hưởng tới sentence embedding.

---

### 4.2.3. Factorized Embedding và Positional Encoding

**Factorized Embedding.** Nếu dùng embedding trực tiếp như BERT-base, mỗi token được ánh xạ thẳng từ vocabulary sang vector 768 chiều. Ma trận embedding khi đó có kích thước:

$$W_{\text{direct}} \in \mathbb{R}^{V \times H}$$

Với $V = 30522$ và $H = 768$, số tham số là:

$$V \times H = 30522 \times 768 \approx 23.4\text{M}$$

SFT-BE dùng factorized embedding để giảm số tham số ở tầng đầu vào. Thay vì ánh xạ trực tiếp $V \rightarrow H$, mô hình tách thành hai bước: token id trước hết được ánh xạ sang vector nhỏ hơn 128 chiều, sau đó mới project lên 768 chiều:

$$t_i \rightarrow e_i \in \mathbb{R}^{128} \rightarrow h_i^{(0)} \in \mathbb{R}^{768}$$

Công thức:

$$e_i = W_E[t_i], \qquad h_i^{(0)} = e_i W_P$$

Trong đó:

$$W_E \in \mathbb{R}^{30522 \times 128}, \qquad W_P \in \mathbb{R}^{128 \times 768}$$

Tổng số tham số của factorized embedding là:

$$30522 \times 128 + 128 \times 768 \approx 4.0\text{M}$$

Như vậy, tầng embedding giảm từ khoảng 23.4M xuống khoảng 4.0M tham số. Ý tưởng này hợp lý vì token embedding ban đầu chủ yếu là representation độc lập ngữ cảnh (*context-independent*), không nhất thiết phải có cùng số chiều lớn như hidden states bên trong Transformer. Phần biểu diễn ngữ cảnh phức tạp hơn sẽ được học ở các layer phía sau.

**Sinusoidal Positional Encoding.** Self-attention không tự biết thứ tự của token. Nếu không thêm positional encoding, câu "dog bites man" và "man bites dog" có thể bị nhìn như cùng một tập token. Vì vậy, sau factorized embedding, mô hình cộng thêm positional encoding để đưa thông tin vị trí vào input.

Với vị trí $\text{pos}$ và chiều $i$, sinusoidal positional encoding được tính như sau:

$$PE(\text{pos},2i) = \sin\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right)$$

$$PE(\text{pos},2i+1) = \cos\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right)$$

Embedding sau khi cộng positional encoding:

$$\tilde{h}_i^{(0)} = h_i^{(0)} + PE(i)$$

Sau đó mô hình áp dụng LayerNorm và Dropout:

$$x_i^{(0)} = \operatorname{Dropout}(\operatorname{LayerNorm}(\tilde{h}_i^{(0)}))$$

Việc dùng sinusoidal positional encoding có ưu điểm là không thêm tham số học được. Các giá trị vị trí được xác định bằng hàm sin và cos, giúp mô hình có thông tin về vị trí token mà vẫn giữ kiến trúc gọn.

---

### 4.2.4. Transformer Encoder trong SFT-BE

Sau embedding layer, input đi qua 6 Transformer encoder block. Mỗi block gồm hai phần chính: Multi-Head Self-Attention và Feed-Forward Network. Mô hình dùng Pre-LN, tức LayerNorm được đặt trước mỗi sub-layer thay vì đặt sau residual như BERT gốc. Thiết kế này giúp gradient ổn định hơn khi train từ đầu.

Luồng xử lý của một encoder block:

$$x \rightarrow \text{LayerNorm} \rightarrow \text{Multi-Head Self-Attention} \rightarrow \text{Residual} \rightarrow \text{LayerNorm} \rightarrow \text{FFN} \rightarrow \text{Residual} \rightarrow x'$$

**Scaled Dot-Product Attention.** Với hidden states $X \in \mathbb{R}^{B \times L \times H}$, mô hình tạo ra Query, Key và Value:

$$Q = XW_Q, \qquad K = XW_K, \qquad V = XW_V$$

Trong đó $W_Q$, $W_K$, $W_V$ là các ma trận projection. Attention score giữa các token được tính bằng dot product giữa query và key:

$$A = \frac{QK^\top}{\sqrt{d_k}}$$

Sau đó softmax biến score thành trọng số attention:

$$P = \operatorname{softmax}(A)$$

Output của attention:

$$\operatorname{Attention}(Q,K,V) = \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

Hệ số $\sqrt{d_k}$ có vai trò scale dot product. Nếu không chia cho $\sqrt{d_k}$, khi $d_k$ lớn, các score có thể quá lớn làm softmax bị bão hòa, gradient trở nên nhỏ và quá trình học kém ổn định.

**Attention mask.** Trong batch có padding token, mô hình không được để token thật attention vào padding. Vì vậy, trước softmax, các vị trí padding được gán score bằng $-\infty$:

$$A_{ij} = \begin{cases}\dfrac{q_i^\top k_j}{\sqrt{d_k}} & m_j = 1\\-\infty & m_j = 0\end{cases}$$

Sau softmax, trọng số attention tại padding token xấp xỉ bằng 0. Nhờ đó, padding chỉ đóng vai trò lấp đầy tensor cho đúng shape, không tham gia vào representation thật của câu.

**Multi-Head Self-Attention.** SFT-BE dùng 12 attention heads. Vì hidden size là 768, mỗi head có chiều:

$$d_k = \frac{768}{12} = 64$$

Mỗi head học một kiểu quan hệ khác nhau giữa các token. Một head có thể nhạy với quan hệ cú pháp, head khác có thể tập trung vào từ khóa nội dung, head khác nữa có thể học các pattern vị trí. Output của các head được concat lại rồi project về hidden size ban đầu:

$$\operatorname{head}_r = \operatorname{Attention}(XW_Q^{(r)}, XW_K^{(r)}, XW_V^{(r)})$$

$$\operatorname{MultiHead}(X) = \operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_{12})W_O$$

**Feed-Forward Network.** Sau self-attention, mỗi token đi qua một FFN giống nhau về tham số nhưng xử lý độc lập theo từng vị trí. FFN mở rộng vector từ 768 lên 3072 chiều, áp dụng GELU, rồi project về lại 768 chiều:

$$\operatorname{FFN}(x) = \operatorname{GELU}(xW_1 + b_1)W_2 + b_2$$

Trong đó:

$$W_1 \in \mathbb{R}^{768 \times 3072}, \qquad W_2 \in \mathbb{R}^{3072 \times 768}$$

GELU được dùng thay vì ReLU vì hàm này mượt hơn và thường hoạt động tốt trong Transformer:

$$\operatorname{GELU}(x) = x\Phi(x)$$

Với $\Phi(x)$ là CDF của phân phối chuẩn. Có thể hiểu FFN là nơi mô hình biến đổi representation của từng token sau khi token đó đã thu thập ngữ cảnh từ self-attention.

**Pre-LN Encoder Block.** Với input layer thứ $\ell$ là $X^{(\ell)}$, block Pre-LN được viết:

$$\hat{X}^{(\ell)} = \operatorname{LayerNorm}(X^{(\ell)})$$

$$Y^{(\ell)} = X^{(\ell)} + \operatorname{Dropout}(\operatorname{MultiHead}(\hat{X}^{(\ell)}))$$

$$\hat{Y}^{(\ell)} = \operatorname{LayerNorm}(Y^{(\ell)})$$

$$X^{(\ell+1)} = Y^{(\ell)} + \operatorname{Dropout}(\operatorname{FFN}(\hat{Y}^{(\ell)}))$$

Sau 6 block, mô hình áp dụng thêm final LayerNorm:

$$H = \operatorname{LayerNorm}(X^{(6)})$$

Trong đó:

$$H \in \mathbb{R}^{B \times L \times 768}$$

Final LayerNorm đặc biệt hữu ích với Pre-LN vì output của block cuối chưa được normalize sau residual cuối cùng.

---

### 4.2.5. Mean Pooling và Sentence Embedding

Transformer encoder trả về hidden state cho từng token, nhưng bài toán similarity cần một vector duy nhất cho toàn bộ câu hoặc đoạn văn. SFT-BE sử dụng mean pooling có attention mask. Với hidden states:

$$H = (h_1,h_2,\ldots,h_L), \qquad h_i \in \mathbb{R}^{768}$$

Sentence embedding được tính bằng trung bình các token thật:

$$z = \frac{\sum_{i=1}^{L} m_i h_i}{\sum_{i=1}^{L} m_i}$$

Trong đó $m_i$ là attention mask. Nếu $m_i = 0$, token đó là padding và không được tính vào trung bình. Nếu viết theo từng chiều $k$:

$$z_k = \frac{\sum_{i=1}^{L} m_i h_{i,k}}{\sum_{i=1}^{L} m_i}, \qquad k = 1,\ldots,768$$

Mean pooling có một ưu điểm thực tế: nó tận dụng thông tin của toàn bộ câu thay vì chỉ phụ thuộc vào một token đại diện duy nhất ở đầu input. Với các mô hình sentence embedding, mean pooling thường ổn định vì nhiều token trong câu cùng đóng góp vào representation cuối cùng. Attention mask giúp phép trung bình này không bị lệch bởi padding, đặc biệt khi các câu trong batch có độ dài khác nhau.

Sau mean pooling, mô hình thu được:

$$z = f_\theta(x) \in \mathbb{R}^{768}$$

Vector này chính là representation dùng cho similarity search, retrieval hoặc downstream ranking.

---

### 4.2.6. Cơ chế huấn luyện bằng Teacher-Student Distillation

SFT-BE được train bằng teacher-student distillation. Thay vì yêu cầu dataset có nhãn cho từng cặp câu, mô hình dùng một teacher sentence embedding model mạnh hơn để tạo embedding mục tiêu. Student học cách bắt chước embedding space của teacher.

Teacher được dùng trong đồ án là:

$$T(\cdot) = \texttt{sentence-transformers/all-mpnet-base-v2}$$

Student là SFT-BE:

$$S_\theta(\cdot) = f_\theta(\cdot)$$

Với mỗi câu $x_i$ trong batch, teacher sinh embedding:

$$t_i = T(x_i)$$

Student sinh embedding:

$$s_i = S_\theta(x_i)$$

Cả teacher và student đều có output 768 chiều:

$$t_i, s_i \in \mathbb{R}^{768}$$

Teacher chạy ở chế độ `eval` và `no_grad`, tức là teacher chỉ đóng vai trò tạo target embedding, không được update trong quá trình train:

$$\nabla_{\phi}\mathcal{L} = 0, \qquad \nabla_{\theta}\mathcal{L} \neq 0$$

Trong đó $\phi$ là tham số của teacher và $\theta$ là tham số của student.

**L2 normalization.** Trước khi tính loss, cả teacher embedding và student embedding đều được normalize:

$$\hat{s}_i = \frac{s_i}{\|s_i\|_2}, \qquad \hat{t}_i = \frac{t_i}{\|t_i\|_2}$$

Sau khi normalize, cosine similarity giữa hai vector trở thành dot product:

$$\operatorname{cos}(s_i,t_i) = \hat{s}_i^\top \hat{t}_i$$

Vì:

$$\|\hat{s}_i\|_2 = 1, \qquad \|\hat{t}_i\|_2 = 1$$

Điều này giúp loss tập trung vào hướng của vector, tức cấu trúc ngữ nghĩa trong embedding space, thay vì bị ảnh hưởng quá nhiều bởi độ lớn vector.

**Cosine Distillation Loss.** Loss cho một mẫu được định nghĩa:

$$\mathcal{L}_i = 1 - \operatorname{cos}(s_i,t_i)$$

Với batch size $B$, loss trung bình là:

$$\mathcal{L}_{\text{batch}} = 1 - \frac{1}{B}\sum_{i=1}^{B}\hat{s}_i^\top \hat{t}_i$$

Tối thiểu hóa loss này tương đương với tối đa hóa độ gần giữa student embedding và teacher embedding:

$$\min_\theta \mathcal{L}_{\text{batch}} \Longleftrightarrow \max_\theta \frac{1}{B}\sum_{i=1}^{B}\operatorname{cos}(S_\theta(x_i),T(x_i))$$

Nói cách khác, student học không phải bằng nhãn class, mà bằng hình dạng của embedding space do teacher tạo ra. Nếu teacher đặt hai câu cùng chủ đề ở gần nhau, student được khuyến khích tạo ra một không gian có cấu trúc tương tự. Đây là lý do distillation phù hợp với giai đoạn pretraining cho sentence/document embedding.

---

### 4.2.7. Dữ liệu huấn luyện và pipeline xử lý

Dữ liệu train chính là các sentence được trích từ Wikipedia tiếng Anh. Pipeline chuẩn bị dữ liệu gồm các bước: tải Wikipedia, tách article thành sentence, lọc các câu quá ngắn hoặc quá dài, tokenize bằng `bert-base-uncased`, padding/truncate về 128 token và lưu cache để train nhiều lần mà không phải xử lý lại.

Với mỗi câu $x$, tokenizer tạo:

$$\text{input\_ids}(x) = (t_1,t_2,\ldots,t_L)$$

$$\text{attention\_mask}(x) = (m_1,m_2,\ldots,m_L)$$

Trong lần chạy thực nghiệm của đồ án, quá trình huấn luyện bị dừng sớm trước khi hoàn tất một epoch, nên các bước đánh giá cuối epoch như validation loss hoặc STS-B không được dùng làm kết quả báo cáo chính thức. Vì vậy, phần kết quả bên dưới chỉ dựa trên training log và một bài test bổ sung trên 10MB dữ liệu Wikipedia lấy từ cuối dataset.

---

### 4.2.8. Tối ưu hóa mô hình

**AdamW optimizer.** Mô hình được tối ưu bằng AdamW với learning rate ban đầu $5\times10^{-4}$, weight decay $0.01$, $\beta_1=0.9$, $\beta_2=0.999$ và $\epsilon=10^{-8}$. Với gradient tại bước $t$ là $g_t$, Adam tính moving average bậc một và bậc hai:

$$m_t = \beta_1m_{t-1} + (1-\beta_1)g_t$$

$$v_t = \beta_2v_{t-1} + (1-\beta_2)g_t^2$$

Sau đó hiệu chỉnh bias:

$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$

Với AdamW, weight decay được tách khỏi gradient update:

$$\theta_t = \theta_{t-1} - \eta\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon} + \lambda\theta_{t-1}\right)$$

AdamW phù hợp trong bối cảnh này vì Transformer thường nhạy với learning rate, còn AdamW vừa có adaptive learning rate vừa có regularization rõ ràng thông qua weight decay.

**Warmup và Cosine Annealing.** Learning rate không giữ cố định trong toàn bộ quá trình train. 10% số update đầu tiên được dùng cho warmup, sau đó learning rate giảm dần theo cosine schedule. Nếu $T$ là tổng số update, $T_w = 0.1T$ là số bước warmup, learning rate tại bước $t$ là:

$$\eta_t = \eta_{\max}\frac{t}{T_w}, \qquad 1 \le t \le T_w$$

Sau warmup:

$$\eta_t = \frac{1}{2}\eta_{\max}\left(1+\cos\left(\pi\frac{t-T_w}{T-T_w}\right)\right), \qquad T_w < t \le T$$

Warmup giúp tránh cập nhật quá mạnh ở giai đoạn đầu khi trọng số còn ngẫu nhiên. Cosine annealing giúp learning rate giảm mượt về cuối, cho phép mô hình tinh chỉnh embedding space ổn định hơn.

**Gradient Accumulation.** Gradient accumulation cho phép mô hình tích lũy gradient qua nhiều mini-batch rồi mới update một lần. Nếu batch size là $B$ và số bước accumulation là $K$, effective batch size được tính:

$$B_{\text{eff}} = B \times K$$

Nếu loss ở mỗi mini-batch là $\mathcal{L}_j$, mô hình chia loss cho số bước accumulation $K$:

$$\tilde{\mathcal{L}}_j = \frac{\mathcal{L}_j}{K}$$

Gradient tích lũy:

$$g = \sum_{j=1}^{K}\nabla_\theta \tilde{\mathcal{L}}_j = \nabla_\theta\left(\frac{1}{K}\sum_{j=1}^{K}\mathcal{L}_j\right)$$

Cách này mô phỏng batch lớn hơn mà không cần tăng VRAM tương ứng. Với mô hình Transformer, batch lớn hơn thường giúp cosine distillation ổn định hơn vì gradient ít nhiễu hơn.

**Mixed Precision.** Khi chạy trên CUDA, pipeline hỗ trợ AMP (*Automatic Mixed Precision*). Một số phép toán được tính bằng precision thấp hơn để giảm bộ nhớ và tăng tốc, trong khi `GradScaler` được dùng để tránh underflow khi backpropagation. Đây là kỹ thuật thực tế giúp train nhanh hơn mà vẫn giữ độ ổn định số học ở mức chấp nhận được.

---

### 4.2.9. Theo dõi huấn luyện và test thực nghiệm

Trong quá trình huấn luyện Stage 0, mô hình được theo dõi bằng loss distillation theo thời gian. Vì quá trình train chỉ chạy được khoảng nửa epoch rồi dừng, phần đánh giá xu hướng loss không sử dụng `avg_loss_cumulative`. Thay vào đó, báo cáo chỉ dùng `interval_loss_reconstructed`, tức loss được ước lượng lại theo từng khoảng log liên tiếp. Cách nhìn này phản ánh tốt hơn trạng thái học tại từng giai đoạn, không bị kéo bởi các batch rất xấu ở đầu training.

Ở lần chạy tạo checkpoint, cấu hình thực tế được ghi trong log như sau:

| Chỉ số | Giá trị |
|:--|:--|
| Stage | Stage 0 - teacher-student distillation |
| Teacher | `sentence-transformers/all-mpnet-base-v2` |
| Tổng số sentence sau xử lý | 90,250,685 |
| Train samples | 88,445,671 |
| Batch size | 16 |
| Gradient accumulation | 8 |
| Effective batch size | 128 |
| Total batch steps cho 1 epoch | 5,527,854 |
| Step cuối ghi nhận | 2,712,000 |
| Tỷ lệ epoch đã chạy | 49.06% |
| Thời gian huấn luyện ghi nhận | 42.39 giờ |

Do đó, checkpoint cuối cùng nên được hiểu là kết quả sau khoảng $0.49$ epoch, chưa phải mô hình đã hội tụ hoàn toàn sau một epoch đầy đủ.

**Interval loss theo thời gian.** Loss được theo dõi từ file `stage0_loss_reconstructed.csv`, chỉ lấy cột `interval_loss_reconstructed`. Một số mốc chính:

| Mốc | Global step | Thời gian | Learning rate | Interval loss | Cosine xấp xỉ |
|:--|--:|--:|--:|--:|--:|
| Đầu train | 500 | 0.02 giờ | $6.36\times10^{-7}$ | 0.8575 | 0.1425 |
| 25% log | 678,500 | 10.28 giờ | $4.92\times10^{-4}$ | 0.1334 | 0.8666 |
| 50% log | 1,356,500 | 20.71 giờ | $4.12\times10^{-4}$ | 0.0950 | 0.9050 |
| 75% log | 2,034,000 | 32.14 giờ | $2.75\times10^{-4}$ | 0.0819 | 0.9181 |
| Cuối train | 2,712,000 | 42.39 giờ | $1.29\times10^{-4}$ | 0.0736 | 0.9264 |

![Stage 0 interval loss theo thời gian](checkpoints/stage0_interval_loss_reconstructed.png)

Đường loss giảm rất mạnh trong các giờ đầu: từ 0.8575 xuống 0.1334 sau khoảng 10.28 giờ. Sau đó loss tiếp tục giảm chậm hơn, từ 0.0950 ở giữa log xuống 0.0736 ở cuối lần chạy. Điều này phù hợp với đặc trưng của distillation: ở giai đoạn đầu student còn rất xa teacher embedding space nên loss cao; sau khi student học được hướng vector cơ bản của teacher, loss giảm chậm dần vì các cập nhật còn lại chủ yếu là tinh chỉnh representation.

Vì loss được định nghĩa:

$$\mathcal{L} = 1 - \operatorname{cos}(s,t)$$

nên interval loss 0.0736 ở cuối train tương ứng cosine trung bình xấp xỉ:

$$1 - 0.0736 = 0.9264$$

Điều này cho thấy student đã học được hướng embedding khá gần với teacher trên các batch huấn luyện ở giai đoạn cuối, dù quá trình train mới đi được khoảng nửa epoch.

**Test bổ sung trên 10MB cuối Wikipedia.** Vì early stopping làm các bước đánh giá chính thức ở cuối epoch bị bỏ qua, mô hình được test bổ sung trên một phần dữ liệu Wikipedia để có số liệu báo cáo. Cách lấy test set là duyệt dataset `wikimedia/wikipedia/20231101.en` từ cuối lên đầu cho tới khi đủ khoảng 10MB raw text. Sau đó áp dụng cùng rule tách câu như pipeline chuẩn bị dữ liệu: split theo `. ` và giữ các câu có độ dài từ 20 đến 500 ký tự.

Thông tin test set:

| Chỉ số | Giá trị |
|:--|--:|
| Dataset | `wikimedia/wikipedia/20231101.en` |
| Split | train |
| Hướng lấy dữ liệu | từ cuối dataset lên đầu |
| Kích thước raw text | 10.003 MB |
| Số article được lấy | 2,453 |
| Index đầu tiên được lấy | 6,405,361 |
| Index cuối cùng | 6,407,813 |
| Số câu sau lọc | 47,959 |
| Checkpoint test | `checkpoints/stage0_final.pt` |
| Batch size test | 64 |

Trên tập test này, mô hình đạt:

| Chỉ số | Giá trị |
|:--|--:|
| Test loss | 0.06567 |
| Test cosine trung bình | 0.93433 |
| Cosine nhỏ nhất | 0.55251 |
| Cosine lớn nhất | 0.98880 |

Test loss được tính cùng công thức với train loss:

$$\mathcal{L}_{\text{test}} = 1 - \frac{1}{N}\sum_{i=1}^{N}\operatorname{cos}(S_\theta(x_i),T(x_i))$$

Với $N = 47{,}959$, giá trị loss 0.06567 tương ứng cosine trung bình 0.93433. Kết quả này cho thấy trên phần dữ liệu Wikipedia lấy từ cuối dataset, student vẫn giữ được độ gần cao với teacher embedding. Tuy nhiên, vì test set này vẫn lấy từ cùng nguồn Wikipedia dùng cho Stage 0, nó nên được xem là kiểm tra bổ sung chất lượng distillation trên một tail-slice của dữ liệu, không phải benchmark độc lập hoàn toàn. Đồng thời, đây vẫn là đánh giá theo teacher-student similarity, không phải đánh giá reasoning hay đúng/sai logic; nó chỉ đo mức độ student bắt chước không gian embedding của teacher.

---

### 4.2.10. Vai trò của SFT-BE trong hệ thống retrieval

Sau khi train xong, SFT-BE có thể được dùng như embedding model cho hệ thống document retrieval. Corpus được chia thành các chunk, mỗi chunk được encode thành vector 768 chiều rồi lưu vào vector database. Khi người dùng nhập query, hệ thống encode query thành vector và tìm top-$K$ chunk gần nhất.

Nếu toàn bộ document embedding đã được L2-normalize và xếp thành ma trận:

$$Z = \begin{bmatrix}z_1^\top\\z_2^\top\\\vdots\\z_N^\top\end{bmatrix} \in \mathbb{R}^{N \times 768}$$

Với query embedding đã normalize là $z_q$, điểm similarity với toàn bộ corpus có thể tính bằng một phép nhân ma trận:

$$s = Zz_q$$

Trong đó $s_i$ chính là cosine similarity giữa query và document chunk thứ $i$. Đây là lợi thế rất lớn của bi-encoder: sau khi pre-compute document embedding, inference chủ yếu chỉ còn là encode query và nearest neighbor search.

Tuy vậy, cần nhìn đúng vai trò của mô hình. SFT-BE mạnh ở bước retrieval ban đầu, tức chọn ra vùng tài liệu có khả năng liên quan. Mô hình không phải là reasoning model và cũng không thay thế hoàn toàn cross-encoder. Với các trường hợp cần phân biệt phủ định, đảo vai chủ thể - đối tượng, hoặc những khác biệt ngữ nghĩa rất nhỏ, có thể cần thêm reranker hoặc verifier phía sau.

---

### 4.2.11. Kết luận

SFT-BE là một mô hình bi-encoder được thiết kế cho bài toán document similarity và semantic retrieval. Mô hình encode mỗi văn bản độc lập thành vector 768 chiều, sau đó dùng cosine similarity để đo độ gần về mặt ngữ nghĩa. Thiết kế bi-encoder giúp document embedding có thể được pre-compute, nhờ đó hệ thống truy hồi nhanh hơn nhiều so với cách dùng cross-encoder cho mọi cặp query-document.

Về kiến trúc, SFT-BE kết hợp factorized embedding 128 -> 768, sinusoidal positional encoding, 6 layer Pre-LN Transformer encoder, multi-head self-attention 12 heads, FFN 3072 chiều và mean pooling có attention mask. Các lựa chọn này giúp mô hình giữ được năng lực biểu diễn của Transformer nhưng giảm số tham số và chi phí so với một encoder lớn hơn như BERT-base.

Về huấn luyện, mô hình dùng teacher-student distillation với teacher `sentence-transformers/all-mpnet-base-v2`. Student được tối ưu bằng cosine distillation loss:

$$\mathcal{L}_{\text{batch}} = 1 - \frac{1}{B}\sum_{i=1}^{B}\operatorname{cos}(S_\theta(x_i),T(x_i))$$

Cách train này cho phép student học lại embedding space giàu ngữ nghĩa của teacher mà không cần nhãn cặp câu thủ công. Kết quả là SFT-BE trở thành một retrieval encoder gọn, nhanh và phù hợp để làm tầng tìm kiếm ban đầu trong hệ thống document similarity.

## 4.3. Mô hình Cross-Encoder dùng cho reranking

Khác với SFT-BE, Cross-Encoder trong đồ án không được nhóm huấn luyện lại từ đầu. Hệ thống sử dụng pretrained model `cross-encoder/ms-marco-MiniLM-L6-v2` thông qua thư viện `sentence-transformers`. Model này được huấn luyện trước cho bài toán MS MARCO Passage Ranking, tức nhận một cặp `(query, passage)` và sinh ra một điểm relevance cho biết passage có phù hợp với query hay không.

Về mặt kiến trúc, model này là một mô hình `BertForSequenceClassification` dùng backbone MiniLM/BERT encoder. Cấu hình chính của model như sau:

| Thành phần | Giá trị |
|:--|:--|
| Model name | `cross-encoder/ms-marco-MiniLM-L6-v2` |
| Kiến trúc HuggingFace | `BertForSequenceClassification` |
| Vocabulary size | 30,522 |
| Hidden size | 384 |
| Transformer layers | 6 |
| Attention heads | 12 |
| Head dimension | 32 |
| FFN intermediate size | 1536 |
| Max position embeddings | 512 |
| Segment/type vocab size | 2 |
| Hidden activation | GELU |
| Attention dropout | 0.1 |
| Hidden dropout | 0.1 |
| LayerNorm eps | 1e-12 |
| Output label | 1 relevance logit |

Điểm quan trọng nhất của Cross-Encoder là query và passage không được encode độc lập. Thay vào đó, cả hai được ghép thành một sequence duy nhất:

$$x = [\text{CLS}], q_1, q_2, \ldots, q_m, [\text{SEP}], p_1, p_2, \ldots, p_n, [\text{SEP}]$$

Trong đó $q_i$ là token của query và $p_j$ là token của passage. Sau tokenization, input của model gồm ba thành phần chính:

$$\text{input\_ids} = (t_1,t_2,\ldots,t_L)$$

$$\text{token\_type\_ids} = (s_1,s_2,\ldots,s_L)$$

$$\text{attention\_mask} = (m_1,m_2,\ldots,m_L)$$

Với BERT-style input, `token_type_ids` thường dùng giá trị 0 cho phần query và 1 cho phần passage:

$$s_i = \begin{cases}0 & \text{nếu token thuộc query hoặc special token trước passage}\\1 & \text{nếu token thuộc passage}\end{cases}$$

`attention_mask` đánh dấu token thật và padding:

$$m_i = \begin{cases}1 & \text{nếu vị trí } i \text{ là token thật}\\0 & \text{nếu vị trí } i \text{ là padding}\end{cases}$$

Embedding đầu vào của Cross-Encoder là tổng của ba loại embedding: token embedding, position embedding và segment embedding:

$$h_i^{(0)} = E_{tok}(t_i) + E_{pos}(i) + E_{seg}(s_i)$$

Trong đó:

$$E_{tok}\in\mathbb{R}^{30522\times384}$$

$$E_{pos}\in\mathbb{R}^{512\times384}$$

$$E_{seg}\in\mathbb{R}^{2\times384}$$

So với SFT-BE, Cross-Encoder dùng hidden size 384 thay vì 768. Tuy nhiên, nó lại có lợi thế là query và passage cùng xuất hiện trong một chuỗi, nên cơ chế attention có thể học tương tác trực tiếp giữa token query và token passage.

Với hidden states tại layer $\ell$:

$$H^{(\ell)}\in\mathbb{R}^{B\times L\times384}$$

mỗi Transformer layer tạo Query, Key và Value:

$$Q = H^{(\ell)}W_Q,\qquad K = H^{(\ell)}W_K,\qquad V = H^{(\ell)}W_V$$

Vì hidden size là 384 và số attention heads là 12, mỗi head có kích thước:

$$d_k = \frac{384}{12}=32$$

Scaled dot-product attention của mỗi head:

$$\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{32}} + M\right)V$$

Trong đó $M$ là attention mask sau khi được broadcast vào ma trận attention. Với vị trí padding, $M$ nhận giá trị rất âm để sau softmax trọng số gần bằng 0. Khác với Bi-Encoder, ma trận attention ở đây cho phép token query attention trực tiếp tới token passage và ngược lại. Ví dụ, token “political” trong query có thể attention đến “partisan”, “election”, “engagement” trong passage nếu mô hình học được chúng liên quan.

Output của multi-head attention:

$$\operatorname{head}_r = \operatorname{Attention}(H^{(\ell)}W_Q^{(r)},H^{(\ell)}W_K^{(r)},H^{(\ell)}W_V^{(r)})$$

$$\operatorname{MHA}(H^{(\ell)})=\operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_{12})W_O$$

Sau attention, mỗi token đi qua feed-forward network có intermediate size 1536:

$$\operatorname{FFN}(x)=\operatorname{GELU}(xW_1+b_1)W_2+b_2$$

với:

$$W_1\in\mathbb{R}^{384\times1536},\qquad W_2\in\mathbb{R}^{1536\times384}$$

Nếu viết theo dạng BERT encoder block, mỗi layer có thể mô tả như sau:

$$A^{(\ell)} = \operatorname{LayerNorm}\left(H^{(\ell)} + \operatorname{Dropout}(\operatorname{MHA}(H^{(\ell)}))\right)$$

$$H^{(\ell+1)} = \operatorname{LayerNorm}\left(A^{(\ell)} + \operatorname{Dropout}(\operatorname{FFN}(A^{(\ell)}))\right)$$

Sau 6 layer, model thu được hidden states cuối:

$$H^{(6)}=(h_{[CLS]},h_1,h_2,\ldots,h_{L-1})$$

Với `BertForSequenceClassification`, vector tại vị trí `[CLS]` được dùng làm representation của cả cặp query-passage. BERT pooler thường lấy hidden state của `[CLS]`, đưa qua một linear layer và hàm `tanh`:

$$u = \tanh(W_{pool}h_{[CLS]} + b_{pool})$$

Classification head sau đó sinh một logit relevance:

$$r = W_{cls}u + b_{cls}$$

Vì model có một output label, $r$ là một số thực duy nhất cho mỗi cặp query-passage. Raw score này được đưa qua sigmoid để chuyển về khoảng 0 đến 1:

$$\sigma(r)=\frac{1}{1+e^{-r}}$$

Sau đó:

$$\text{cross\_score}=\sigma(r)$$

và:

$$\text{final\_score}=\text{cross\_score}$$

trong bước Cross-Encoder reranking.

Về mặt xác suất, có thể hiểu sigmoid score như một điểm relevance tương đối:

$$P(y=1\mid q,p)=\sigma(r(q,p))$$

Tuy nhiên, trong hệ thống retrieval, điểm này chủ yếu được dùng để xếp hạng, không nhất thiết được diễn giải như xác suất tuyệt đối đã hiệu chỉnh. Điều quan trọng là nếu hai passage $p_a$ và $p_b$ cùng được so với query $q$, passage có $r(q,p)$ hoặc $\sigma(r(q,p))$ cao hơn sẽ được xếp trên.

Trong pipeline, mỗi candidate được chuyển thành một cặp $(q,p_i)$, sau đó Cross-Encoder dự đoán raw score, chuẩn hóa bằng sigmoid và sắp xếp giảm dần. Passage đưa vào reranker không chỉ gồm nội dung chunk, mà có thể kèm metadata ngữ nghĩa như tên paper hoặc section. Thiết kế này giúp Cross-Encoder có thêm ngữ cảnh. Một đoạn chunk ngắn có thể không đủ thông tin nếu đứng riêng, nhưng khi thêm tên paper và section, model có thể đánh giá tốt hơn quan hệ giữa query và passage.

Có thể so sánh trực tiếp Bi-Encoder và Cross-Encoder như sau. Với Bi-Encoder:

$$z_q=f_\theta(q),\qquad z_p=f_\theta(p)$$

$$score(q,p)=\operatorname{cos}(z_q,z_p)$$

Query và passage không tương tác cho đến bước cosine. Với Cross-Encoder:

$$score(q,p)=g_\phi([q;p])$$

Trong đó $[q;p]$ là chuỗi ghép query-passage và $g_\phi$ là Transformer sequence classification model. Tương tác token-level xảy ra bên trong từng layer attention. Vì vậy, Cross-Encoder thường cho ranking chính xác hơn, đặc biệt khi cần phân biệt các quan hệ tinh vi như phủ định, con số, entity hoặc đảo vai chủ thể - đối tượng.

Chi phí suy luận là điểm đánh đổi chính. Nếu corpus có $N$ chunk, Cross-Encoder phải chạy:

$$N \text{ forward passes cho mỗi query}$$

nếu dùng trực tiếp trên toàn bộ corpus. Trong khi đó, Bi-Encoder có thể precompute embedding của tất cả chunk. Vì vậy, hệ thống dùng chiến lược hai giai đoạn:

$$\text{SFT-BE top 100} \rightarrow \text{Cross-Encoder top 10} \rightarrow \text{optional LLM top 8} \rightarrow \text{final top 5}$$

Cách này tận dụng ưu điểm của cả hai mô hình. SFT-BE có recall tốt và tốc độ cao ở bước đầu. Cross-Encoder đọc kỹ nhóm candidate nhỏ để tăng precision ở thứ hạng cao. LLM verifier, nếu được bật, tiếp tục kiểm tra các ứng viên tốt nhất bằng context window.

Trong đồ án, nhóm không fine-tune Cross-Encoder. Điều này cần được trình bày rõ: Cross-Encoder là pretrained reranker được dùng như một thành phần có sẵn trong pipeline, không phải mô hình được huấn luyện bởi nhóm. Vì vậy, các tham số của Cross-Encoder không được cập nhật trong đề tài. Đóng góp của nhóm nằm ở cách tích hợp nó vào pipeline retrieval, cách bổ sung metadata vào passage, cách chuẩn hóa score bằng sigmoid và cách kết hợp nó với SFT-BE cùng LLM verifier.


## 4.4. Xây dựng chỉ mục vector cho tài liệu

Sau khi có checkpoint SFT-BE, hệ thống tạo chỉ mục vector cho tập tài liệu cần truy xuất. Quy trình gồm các bước: nhận tập tài liệu đầu vào, trích xuất text, chuẩn hóa text, chia thành chunk, encode từng chunk bằng SFT-BE, L2-normalize embedding và lưu embedding cùng metadata.

Gọi tập chunk sau tiền xử lý là:

$$\mathcal{C}=\{c_1,c_2,\ldots,c_N\}$$

Mỗi chunk $c_i$ được đưa qua encoder:

$$z_i=f_\theta(c_i)\in\mathbb{R}^{768}$$

Sau đó vector được chuẩn hóa:

$$\hat{z}_i=\frac{z_i}{\|z_i\|_2}$$

Tập vector của toàn bộ corpus được gom thành ma trận:

$$Z=
\begin{bmatrix}
\hat{z}_1^\top\\
\hat{z}_2^\top\\
\vdots\\
\hat{z}_N^\top
\end{bmatrix}
\in\mathbb{R}^{N\times768}
$$

Metadata của mỗi chunk có thể gồm tên paper, section, page và vị trí chunk. Metadata không tham gia trực tiếp vào cosine similarity, nhưng rất quan trọng để hiển thị citation và tạo context window.

Khi truy vấn, query vector được normalize rồi nhân với ma trận vector:

$$scores=Z\hat{z}_q$$

Sau đó hệ thống lấy top-k chunk có score cao nhất. Với corpus nhỏ, có thể tính trực tiếp toàn bộ ma trận điểm. Với corpus lớn, cùng ý tưởng này có thể được thay bằng approximate nearest neighbor để giảm chi phí truy xuất.

## 4.5. Pipeline retrieval, reranking và verification

Pipeline retrieval gồm các bước:

1. Encode query bằng SFT-BE.
2. Tìm top `vector_top_k` bằng vector store.
3. Áp dụng lexical boost.
4. Rerank bằng Cross-Encoder.
5. Nếu bật `use_llm`, rerank tiếp bằng LLM verifier.
6. Sort lại theo `final_score`.
7. Trả top `final_k`.

Lexical boost lấy các content terms từ query, loại stopwords đơn giản, rồi tính overlap với paper, section và text của chunk:

$$overlap=\frac{|terms(q)\cap terms(c)|}{|terms(q)|}$$

Score sau lexical boost:

$$score=0.82\times vector\_score+0.18\times overlap$$

Cross-Encoder nhận các cặp `(query, passage)` và dự đoán raw score. Passage có thể được ghép từ metadata như paper, section và nội dung chunk để reranker có thêm ngữ cảnh.

Raw score được sigmoid thành `cross_score`, sau đó dùng làm `final_score`.

LLM verifier dùng Ollama. Prompt yêu cầu LLM chấm điểm từng candidate từ 0 đến 1, chú ý phủ định, số liệu, địa danh, entity và đảo quan hệ chủ thể - đối tượng. Nếu parse được JSON, final score:

$$final=0.25\times base+0.75\times llm\_score$$

Nếu LLM không khả dụng hoặc parse lỗi, hệ thống fallback về score trước đó.

### 4.5.1. Chi tiết luồng dữ liệu trong hệ thống retrieval

Để hiểu rõ hơn hệ thống hoạt động như thế nào, có thể đi theo một truy vấn từ lúc người dùng nhập vào đến lúc kết quả được trả về. Đầu tiên, query là một chuỗi text thô. Hệ thống chuẩn hóa query, tokenize bằng `bert-base-uncased`, đưa qua SFT-BE và nhận embedding 768 chiều. Embedding này được L2-normalize để có thể so sánh bằng dot product với ma trận vector đã normalize của corpus.

Tiếp theo, vector search tính:

$$s_i=\hat{z}_q^\top \hat{z}_i,\qquad i=1,\ldots,N$$

trong đó $\hat{z}_q$ là query embedding đã normalize và $\hat{z}_i$ là embedding của chunk thứ $i$. Các candidate được lấy theo score giảm dần. Vì cả hai phía đều đã normalize, $s_i$ chính là cosine similarity.

Sau vector search, pipeline áp dụng lexical boost. Hệ thống lấy các content term từ query, bỏ token quá ngắn và stopwords phổ biến. Với mỗi candidate, phần text dùng để tính overlap gồm nội dung chunk và một số metadata ngữ nghĩa như paper hoặc section. Điều này giúp các từ trong tiêu đề paper hoặc section cũng tham gia vào lexical score. Ví dụ, nếu query nhắc đến "Mastodon governance", một chunk thuộc paper có tên liên quan đến Mastodon có thể được tăng nhẹ điểm dù đoạn text cụ thể không lặp lại đầy đủ cụm từ.

Sau lexical boost, candidate được sắp xếp theo score tạm thời. Nếu dùng Cross-Encoder, hệ thống tạo các cặp query-passage cho top candidate, dự đoán raw relevance score, chuẩn hóa bằng sigmoid và giữ lại nhóm kết quả tốt nhất. Passage có thể kèm metadata paper và section để tăng ngữ cảnh cho reranker.

Nếu bật LLM verifier, pipeline lấy nhóm candidate tốt nhất sau Cross-Encoder và gửi kèm context window. LLM verifier nhận thêm chunk trước và sau, từ đó giảm rủi ro chấm sai do thiếu ngữ cảnh. Nếu verifier không khả dụng hoặc không trả về kết quả hợp lệ, pipeline vẫn có thể trả kết quả dựa trên Cross-Encoder thay vì dừng toàn bộ truy vấn.

Cuối cùng, kết quả được sort theo score cuối và cắt về top-k. Mỗi result đi kèm text, score, citation và metadata cần thiết để người dùng kiểm chứng nguồn. Nếu cần, hệ thống hiển thị thêm context window xung quanh chunk được chọn.

### 4.5.2. Thiết kế an toàn khi thiếu metadata hoặc thiếu LLM

Trong các hệ thống truy xuất tài liệu, một lỗi phổ biến là hiển thị metadata không chắc chắn như thể đó là sự thật. Vì vậy, nếu không tìm thấy section hoặc page, hệ thống cần ghi rõ metadata chưa xác định thay vì đoán bừa. Cách này tốt hơn việc tự suy diễn trang hoặc section, vì citation sai có thể làm người dùng tin nhầm nguồn.

Tương tự, LLM verifier được thiết kế là optional. Nếu LLM local không khả dụng, hệ thống vẫn trả kết quả dựa trên Cross-Encoder. Thiết kế này làm pipeline bền hơn: semantic search và reranking không phụ thuộc tuyệt đối vào LLM, còn LLM chỉ được dùng khi cần tăng kiểm chứng hoặc sinh câu trả lời từ context.

## 4.6. Thuật toán tổng quát của hệ thống

### 4.6.1. Lập chỉ mục tài liệu

Thuật toán lập chỉ mục nhận đầu vào là tập tài liệu và checkpoint SFT-BE. Với mỗi tài liệu, hệ thống trích xuất văn bản, chuẩn hóa, chia thành các chunk có overlap, rồi lưu metadata cần thiết như paper, section và page. Sau đó toàn bộ chunk được đưa qua SFT-BE để tạo embedding:

$$z_i=f_\theta(c_i),\qquad i=1,\ldots,N$$

Các embedding được L2-normalize:

$$\hat{z}_i=\frac{z_i}{\|z_i\|_2}$$

Kết quả của bước này là một chỉ mục gồm ma trận vector $Z\in\mathbb{R}^{N\times768}$ và metadata tương ứng cho từng chunk. Đây là trạng thái đã chuẩn bị trước, giúp bước truy vấn không cần encode lại tài liệu.

### 4.6.2. Semantic search

Với query $q$, hệ thống encode query thành vector:

$$\hat{z}_q=\frac{f_\theta(q)}{\|f_\theta(q)\|_2}$$

Điểm vector của chunk thứ $i$:

$$s_i^{vec}=Z_i\hat{z}_q$$

Sau đó, lexical overlap được dùng như một tín hiệu phụ:

$$s_i^{lex}=\frac{|terms(q)\cap terms(c_i)|}{|terms(q)|}$$

Score tạm thời:

$$s_i^{base}=0.82s_i^{vec}+0.18s_i^{lex}$$

Hệ thống chọn top candidate theo $s_i^{base}$, đưa qua Cross-Encoder để nhận $s_i^{cross}$, sau đó xếp hạng lại. Nếu bật verifier, score cuối có thể được trộn thêm với $s_i^{llm}$:

$$s_i^{final}=0.25s_i^{cross}+0.75s_i^{llm}$$

Nếu không dùng verifier, $s_i^{final}=s_i^{cross}$. Kết quả cuối cùng là top-k chunk có score cao nhất cùng citation.

### 4.6.3. Answer generation

Answer generation chỉ được thực hiện sau khi retrieval đã có bằng chứng. Hệ thống chọn một số kết quả tốt nhất, mở rộng context window quanh từng chunk, gắn citation cho từng đoạn bằng chứng rồi mới đưa vào LLM. Nguyên tắc thiết kế là LLM không được trả lời ngoài context đã truy xuất. Nếu context không đủ, câu trả lời phải thể hiện rằng bằng chứng chưa đủ thay vì tự bịa thêm thông tin.

Về mặt khái niệm, answer generation có thể viết:

$$A=\operatorname{LLM}(q,\mathcal{E})$$

trong đó $\mathcal{E}$ là tập evidence đã truy xuất. Chất lượng của $A$ phụ thuộc trực tiếp vào chất lượng retrieval, vì LLM chỉ là tầng diễn đạt lại và tổng hợp bằng chứng, không thay thế nhiệm vụ tìm đúng tài liệu.

## 4.7. Công nghệ sử dụng

| Công nghệ | Vai trò |
|:--|:--|
| Python | Ngôn ngữ lập trình chính |
| PyTorch | Xây dựng và huấn luyện SFT-BE |
| Transformers | Tokenizer `bert-base-uncased` |
| Sentence-Transformers | Teacher model và Cross-Encoder |
| Datasets | Load Wikipedia và STS-B |
| NumPy | Lưu và tính toán vector |
| SciPy | Tính Spearman correlation |
| pypdf | Trích xuất text từ PDF |
| Ollama | LLM verifier và answer generation |

---

# CHƯƠNG 5. THIẾT KẾ ỨNG DỤNG

## 5.1. Thiết kế tổng thể ứng dụng

Ứng dụng được thiết kế như một hệ thống retrieval cục bộ phục vụ tra cứu tài liệu. Luồng sử dụng gồm hai pha chính. Pha thứ nhất là lập chỉ mục tài liệu, trong đó hệ thống đọc corpus, chia chunk, encode và lưu vector. Pha thứ hai là truy vấn, trong đó người dùng nhập câu hỏi tự nhiên và hệ thống trả về các đoạn liên quan nhất cùng citation.

Thiết kế này tách rõ lập chỉ mục khỏi truy vấn. Việc lập chỉ mục có thể tốn thời gian hơn vì phải xử lý toàn bộ corpus, nhưng chỉ cần thực hiện khi tài liệu thay đổi. Khi truy vấn, embedding của tài liệu đã có sẵn nên hệ thống chỉ cần encode query và so sánh với vector đã lưu.

Các lớp chức năng chính gồm:

- **Document processing layer:** đọc tài liệu, chuẩn hóa text và chia chunk.
- **Embedding layer:** dùng SFT-BE để sinh vector cho query và chunk.
- **Vector retrieval layer:** tìm top-k candidate bằng cosine similarity.
- **Reranking layer:** dùng Cross-Encoder để xếp hạng lại candidate.
- **Verification layer:** dùng LLM local để kiểm tra bằng chứng nếu được bật.
- **Presentation layer:** hiển thị kết quả, citation, score và context window.

## 5.2. Chức năng lập chỉ mục tài liệu

Chức năng lập chỉ mục có nhiệm vụ biến corpus thô thành một cấu trúc có thể truy xuất nhanh. Đầu vào của chức năng này là tập tài liệu cần tìm kiếm. Đầu ra là chỉ mục gồm embedding của từng chunk và metadata đi kèm.

Quy trình lập chỉ mục gồm các bước:

1. Đọc tài liệu và trích xuất văn bản.
2. Chuẩn hóa văn bản để giảm nhiễu do xuống dòng, khoảng trắng hoặc ký tự lỗi.
3. Chia văn bản thành chunk có độ dài phù hợp.
4. Encode từng chunk bằng SFT-BE.
5. Normalize embedding để dùng cosine similarity.
6. Lưu vector cùng metadata phục vụ citation.

Metadata là phần rất quan trọng trong thiết kế. Một hệ thống retrieval chỉ trả về đoạn text mà không cho biết nó đến từ đâu sẽ khó kiểm chứng. Vì vậy, mỗi chunk cần gắn với thông tin như paper, section, page và vị trí chunk. Các thông tin này không chỉ phục vụ hiển thị, mà còn giúp context window lấy thêm đoạn trước/sau khi cần.

## 5.3. Chức năng semantic search

Chức năng semantic search bắt đầu bằng việc encode query. Vì document embedding đã được precompute, chi phí chính của vector search là một phép nhân ma trận giữa query vector và matrix vector trong chỉ mục. Với chỉ mục hiện tại:

$$Z\in\mathbb{R}^{3402\times768}$$

Query embedding:

$$z_q\in\mathbb{R}^{768}$$

Score:

$$s=Zz_q$$

Sau đó hệ thống lấy top 100 candidate theo cấu hình mặc định. Top 100 đủ rộng để không bỏ sót nhiều kết quả tiềm năng, nhưng vẫn đủ nhỏ để Cross-Encoder reranking không quá chậm.

Điểm cần nhấn mạnh là semantic search không yêu cầu query và tài liệu trùng từ hoàn toàn. Nếu query dùng "political behavior" còn tài liệu dùng "political engagement", embedding model vẫn có thể đưa hai biểu thức này lại gần nhau trong không gian vector. Đây chính là ưu điểm của dense retrieval so với keyword search thuần túy.

## 5.4. Chức năng reranking bằng Cross-Encoder

Cross-Encoder reranking giúp khắc phục hạn chế của Bi-Encoder. Trong Bi-Encoder, query và chunk được encode độc lập. Trong Cross-Encoder, query và chunk được đọc cùng nhau. Vì vậy, Cross-Encoder có thể đánh giá các quan hệ chi tiết hơn, ví dụ query hỏi về nguyên nhân nhưng chunk chỉ nói về kết quả, hoặc query chứa phủ định nhưng chunk khẳng định ngược lại.

Trong hệ thống, Cross-Encoder chỉ chạy trên nhóm candidate do Bi-Encoder chọn ra. Cách này giúp cân bằng giữa tốc độ và chất lượng. Nếu chạy Cross-Encoder trên toàn bộ corpus, chi phí suy luận sẽ quá cao. Ngược lại, nếu chỉ dùng Bi-Encoder, ranking có thể thiếu chính xác ở các truy vấn cần hiểu quan hệ tinh vi. Vì vậy, chiến lược hợp lý là dùng Bi-Encoder để lấy rộng, sau đó dùng Cross-Encoder để đọc kỹ nhóm nhỏ.

## 5.5. Chức năng LLM verifier và answer generation

LLM verifier là tầng tùy chọn. Nó không thay thế retrieval, mà chỉ kiểm tra lại các candidate tốt nhất. Tầng này hữu ích trong những truy vấn có phủ định, con số, địa danh, entity hoặc quan hệ chủ thể - đối tượng. Đây là các trường hợp mà embedding similarity có thể chưa đủ chặt chẽ.

Score từ LLM được trộn với base score:

$$final=0.25\times base+0.75\times llm\_score$$

Vì LLM có thể không chạy hoặc trả output không ổn định, verifier phải được thiết kế theo hướng có fallback. Nếu verifier không khả dụng, hệ thống vẫn trả kết quả dựa trên Cross-Encoder. Điều này quan trọng vì LLM chỉ là tầng tăng cường, không phải điều kiện bắt buộc để semantic search hoạt động.

Answer generation cũng dùng Ollama. System prompt yêu cầu chỉ trả lời dựa trên retrieved context, không bịa nguồn, và mỗi factual claim phải có citation. Nếu context không đủ, hệ thống phải nói evidence insufficient. Điều này giúp hạn chế hallucination.

## 5.6. Thiết kế citation và context window

Mỗi kết quả cần đi kèm citation để người dùng kiểm chứng nguồn. Citation nên thể hiện tối thiểu paper, section, page và vị trí chunk nếu các thông tin này có sẵn. Nếu không có section hoặc page, hệ thống phải ghi rõ là chưa xác định thay vì tự bịa. Đây là lựa chọn đúng vì metadata không chắc chắn thì không nên trình bày như một sự thật.

Context window lấy chunk trước và chunk sau của kết quả chính trong cùng tài liệu. Với chunk trung tâm $c_i$, context window có thể biểu diễn:

$$\mathcal{W}(c_i)=\{c_j\mid i-b\leq j\leq i+a\}$$

trong đó $b$ là số chunk trước và $a$ là số chunk sau. Context window giúp người dùng đọc đủ ngữ cảnh xung quanh kết quả. Trong tài liệu khoa học, một chunk đơn lẻ có thể chỉ là một phần của lập luận, nên context window làm kết quả dễ hiểu hơn.

## 5.7. Thiết kế giao diện truy vấn

Giao diện truy vấn cần ưu tiên ba nhiệm vụ: nhập query, xem danh sách kết quả và kiểm chứng nguồn. Với bài toán tìm kiếm tài liệu, giao diện không nên chỉ hiển thị một câu trả lời duy nhất, vì người dùng cần biết bằng chứng nằm ở đâu. Vì vậy, mỗi kết quả nên có các thành phần: rank, score, citation, đoạn text liên quan và context mở rộng nếu người dùng cần đọc thêm.

Một thiết kế hợp lý là chia kết quả thành hai lớp thông tin. Lớp đầu tiên là phần tóm tắt để người dùng quét nhanh: tiêu đề paper, section/page, score và vài dòng preview. Lớp thứ hai là phần chi tiết: chunk đầy đủ, context window và lý do nếu verifier có trả về. Cách chia này giúp giao diện vừa gọn, vừa không làm mất khả năng kiểm chứng.

Các tùy chọn như bật verifier, sinh answer hoặc hiển thị context nên được đặt ở mức người dùng có thể kiểm soát. Mặc định, hệ thống có thể trả semantic search và reranking; verifier và answer generation chỉ bật khi người dùng cần câu trả lời tổng hợp hoặc muốn kiểm tra bằng chứng kỹ hơn.

## 5.8. Đánh giá thiết kế ứng dụng

Thiết kế ứng dụng có ưu điểm là rõ luồng xử lý, dễ kiểm thử và dễ mở rộng. Indexing, vector retrieval, reranking, verification và presentation được tách thành các tầng chức năng riêng. Hệ thống cũng chú trọng citation, giúp kết quả truy xuất có thể kiểm tra lại.

Hạn chế là giao diện còn đơn giản, chưa có upload file trên web, chưa có trình đọc PDF tích hợp và chưa highlight đoạn liên quan trong tài liệu gốc. Vector store cục bộ phù hợp với thử nghiệm nhưng chưa tối ưu cho corpus lớn. Ngoài ra, LLM verifier phụ thuộc vào mô hình local nên không phải lúc nào cũng khả dụng.

Một điểm đáng chú ý trong thiết kế ứng dụng là hệ thống không trộn lẫn việc "truy xuất bằng chứng" với việc "sinh câu trả lời". Trong nhiều ứng dụng NLP, nếu LLM trả lời trực tiếp mà không có retrieval rõ ràng, người dùng khó biết thông tin đến từ đâu. Ở đây, answer generation chỉ được gọi sau khi đã có các kết quả truy xuất. Prompt của answer generation yêu cầu mô hình dùng đúng context được cung cấp và trích dẫn nguồn. Vì vậy, phần trả lời tự nhiên được đặt sau retrieval, không thay thế retrieval.

Về mặt phân tích, việc lưu các score thành phần giúp hiểu vì sao một kết quả được xếp hạng cao hoặc thấp. Nếu một kết quả có vector score cao nhưng cross score thấp, có thể hiểu rằng Bi-Encoder thấy cùng chủ đề nhưng Cross-Encoder không đánh giá là phù hợp trực tiếp. Nếu verifier chấm thấp một kết quả có cross score cao, cần xem lại liệu bằng chứng có thiếu entity, con số hoặc quan hệ logic hay không. Nhờ đó, hệ thống không chỉ là một danh sách kết quả cuối, mà còn cung cấp dấu vết để phân tích lỗi.

Về khả năng mở rộng, kiến trúc hiện tại có thể phát triển theo từng lớp. Nếu corpus lớn hơn, có thể thay vector store cục bộ bằng một vector database chuyên dụng. Nếu muốn tăng chất lượng ranking, có thể fine-tune hoặc thay Cross-Encoder. Nếu muốn thay LLM verifier, chỉ cần giữ nguyên nguyên tắc: verifier nhận query, evidence và trả về đánh giá dựa trên bằng chứng. Cách chia tầng này làm cho đề tài vừa chứng minh được pipeline hiện tại, vừa mở đường cho các cải tiến sau.

---

# CHƯƠNG 6. KẾT QUẢ THỰC NGHIỆM VÀ ĐÁNH GIÁ

## 6.1. Cấu hình thực nghiệm

Đề tài hỗ trợ ba loại device: CUDA, MPS và CPU. Training log cho thấy lần huấn luyện chính chạy trên CUDA, còn test tail Wikipedia được ghi nhận trên MPS.

Các kết quả thực nghiệm chính được lấy từ checkpoint Stage 0, log loss theo thời gian, đánh giá tail-slice Wikipedia và metadata của chỉ mục paper.

Số tham số SFT-BE:

$$46,536,960\approx46.54M$$

## 6.2. Siêu tham số huấn luyện

| Tham số | Giá trị |
|:--|--:|
| Teacher | `sentence-transformers/all-mpnet-base-v2` |
| Tokenizer | `bert-base-uncased` |
| Max sequence length | 128 |
| Learning rate | 5e-4 |
| Weight decay | 0.01 |
| Warmup ratio | 0.1 |
| Validation ratio | 0.02 |
| Split seed | 42 |
| Loss | Cosine distillation loss |

Trong log thực nghiệm, cấu hình thực tế:

| Chỉ số | Giá trị |
|:--|--:|
| Batch size | 16 |
| Gradient accumulation | 8 |
| Effective batch size | 128 |
| Total batch steps cho 1 epoch | 5,527,854 |
| Step cuối ghi nhận | 2,712,000 |
| Tỷ lệ epoch đã chạy | 49.06% |
| Thời gian huấn luyện ghi nhận | 42.39 giờ |

Vì quá trình train mới chạy khoảng 0.49 epoch, checkpoint nên được hiểu là mô hình Stage 0 đã học mạnh nhưng chưa hoàn tất toàn bộ epoch.

## 6.3. Kết quả huấn luyện Stage 0

Log huấn luyện Stage 0 cho thấy loss giảm rõ rệt theo thời gian:

| Mốc | Global step | Thời gian | Learning rate | Interval loss | Cosine xấp xỉ |
|:--|--:|--:|--:|--:|--:|
| Đầu train | 500 | 0.02 giờ | 6.36e-7 | 0.8575 | 0.1425 |
| 25% log | 678,500 | 10.28 giờ | 4.92e-4 | 0.1334 | 0.8666 |
| 50% log | 1,356,500 | 20.71 giờ | 4.12e-4 | 0.0950 | 0.9050 |
| 75% log | 2,034,000 | 32.14 giờ | 2.75e-4 | 0.0819 | 0.9181 |
| Cuối train | 2,712,000 | 42.39 giờ | 1.29e-4 | 0.0736 | 0.9264 |

Vì:

$$\mathcal{L}=1-\operatorname{cos}(student,teacher)$$

nên interval loss 0.0736 tương ứng cosine xấp xỉ 0.9264. Kết quả này cho thấy student đã học được hướng embedding gần teacher trên các batch huấn luyện cuối. Loss giảm mạnh ở giai đoạn đầu, sau đó giảm chậm hơn, phù hợp với quá trình distillation: ban đầu student còn xa teacher nên cải thiện nhanh, về sau chủ yếu tinh chỉnh embedding space.

## 6.4. Đánh giá trên 10MB cuối Wikipedia

Vì training bị dừng trước cuối epoch, nhóm thực hiện đánh giá bổ sung trên 10MB cuối Wikipedia. Phần đánh giá lấy dữ liệu từ cuối dataset `wikimedia/wikipedia/20231101.en`, tách câu bằng cùng rule với training và tính cosine giữa student với teacher.

Thông tin test set:

| Chỉ số | Giá trị |
|:--|--:|
| Dataset | `wikimedia/wikipedia/20231101.en` |
| Hướng lấy dữ liệu | Từ cuối dataset lên đầu |
| Kích thước raw text | 10.003 MB |
| Số article | 2,453 |
| Số câu sau lọc | 47,959 |
| Batch size | 64 |
| Device | MPS |

Kết quả:

| Chỉ số | Giá trị |
|:--|--:|
| Test loss | 0.06567 |
| Test cosine trung bình | 0.93433 |
| Cosine nhỏ nhất | 0.55251 |
| Cosine lớn nhất | 0.98880 |
| Thời gian chạy | 802.78 giây |

Cosine trung bình 0.93433 cho thấy SFT-BE bắt chước teacher tốt trên tail-slice này. Tuy nhiên, đây vẫn là dữ liệu cùng nguồn Wikipedia với Stage 0, nên kết quả này đánh giá chất lượng distillation chứ chưa chứng minh đầy đủ khả năng retrieval trên mọi domain.

Cần phân biệt rõ ý nghĩa của kết quả này. Test loss 0.06567 không có nghĩa mô hình trả lời đúng 93.43% câu hỏi. Nó chỉ nói rằng trung bình hướng vector của student gần hướng vector của teacher trên 47,959 câu được kiểm tra. Vì teacher là một sentence embedding model mạnh, việc student tiến gần teacher là điều kiện tốt cho semantic retrieval, nhưng chưa đủ để kết luận về reasoning, entailment hay factual QA.

Giá trị cosine nhỏ nhất 0.55251 cũng đáng chú ý. Nó cho thấy vẫn có những câu mà student chưa bắt chước teacher tốt. Các trường hợp này có thể thuộc một trong các nhóm sau:

- Câu có cấu trúc lạ hoặc bị trích xuất không sạch.
- Câu chứa nhiều tên riêng, ký hiệu, công thức hoặc token hiếm.
- Câu quá dài và bị tokenizer truncate.
- Câu có ngữ nghĩa phức tạp mà student 6 layer chưa học tốt.

Nếu tiếp tục phát triển, nhóm có thể lấy các câu có cosine thấp nhất, decode lại input và phân tích thủ công. Đây là một cách tìm lỗi có giá trị: thay vì chỉ nhìn mean cosine, ta xem phần tail xấu nhất để biết mô hình yếu ở đâu. Nếu nhiều lỗi đến từ dữ liệu bẩn, cần cải thiện preprocessing. Nếu lỗi đến từ câu chuyên ngành hoặc cấu trúc dài, cần fine-tune thêm hoặc tăng năng lực mô hình.

Ngoài ra, test set được lấy từ cuối dataset theo hướng tail-to-head. Cách lấy này giúp tránh việc chỉ đánh giá đúng trên một phần đầu dataset quen thuộc, nhưng vẫn chưa phải split độc lập hoàn toàn vì vẫn cùng nguồn Wikipedia. Do đó, báo cáo trình bày kết quả này như kiểm tra bổ sung cho distillation, không thổi phồng thành benchmark tổng quát.

## 6.5. Kết quả xây dựng chỉ mục paper

Chỉ mục được xây dựng từ 30 PDF thuộc tập paper kiểm tra. Kết quả metadata của chỉ mục:

| Thuộc tính | Giá trị |
|:--|--:|
| Số chunk | 3,402 |
| Embedding dim | 768 |
| Chunk max chars | 900 |
| Chunk overlap chars | 120 |
| Thời gian lập chỉ mục | 46.58 giây |

Ma trận vector có shape $(3402,768)$ với kiểu dữ liệu `float32`. Điều này xác nhận pipeline từ PDF -> chunk -> embedding -> vector store đã chạy thành công.

## 6.6. Kiểm thử retrieval và reranking

Nhóm thực hiện 4 kiểm thử nội bộ cho logic retrieval và reranking. Kết quả chạy thành công.

Các test kiểm tra:

- Cross-Encoder reranker sắp xếp theo cross score và giữ top-k.
- Passage đưa vào Cross-Encoder có chứa metadata và chunk text.
- Query rỗng bị từ chối.
- Pipeline lấy vector top 100, rerank còn top 10 và trả final top 5.

Các test này không thay thế benchmark retrieval có nhãn, nhưng xác nhận logic pipeline đúng với thiết kế.

## 6.7. So sánh vai trò các mô hình

| Thành phần | Ưu điểm | Hạn chế | Vai trò |
|:--|:--|:--|:--|
| SFT-BE Bi-Encoder | Nhanh, precompute được document embedding, phù hợp corpus lớn | Không có token-level interaction giữa query và chunk | First-stage retriever |
| Lexical boost | Giữ tín hiệu từ khóa, tên riêng, mã tài liệu | Không hiểu đồng nghĩa, phụ thuộc overlap token | Bổ sung vector score |
| Cross-Encoder | Đánh giá cặp query-passage chi tiết hơn | Chậm hơn, phải chạy cho từng candidate | Reranking top-k |
| LLM verifier | Có thể kiểm tra bằng chứng, phủ định, entity, số liệu | Phụ thuộc Ollama, có thể parse lỗi, chậm | Verification tùy chọn |
| Vector store cục bộ | Đơn giản, dễ kiểm soát, phù hợp thử nghiệm | Chưa tối ưu cho hàng triệu vector | Lưu và truy xuất vector |

Từ bảng này, có thể thấy hệ thống không phụ thuộc vào một mô hình duy nhất. Bi-Encoder giúp mở rộng tốc độ, Cross-Encoder giúp tăng chất lượng xếp hạng, LLM verifier giúp kiểm tra bằng chứng khi cần.

## 6.8. Phân tích lỗi và thảo luận

Hệ thống có một số nguồn lỗi chính.

Thứ nhất, lỗi từ trích xuất PDF. Nếu PDF có layout nhiều cột, bảng biểu hoặc text layer không chuẩn, `pypdf` có thể trích xuất sai thứ tự. Khi text đầu vào sai, embedding và retrieval cũng bị ảnh hưởng.

Thứ hai, lỗi từ chunking. Chunk theo ký tự và paragraph có thể cắt ngang một lập luận hoặc tách rời định nghĩa với phần giải thích. Overlap 120 ký tự giảm rủi ro nhưng không giải quyết hoàn toàn.

Thứ ba, lỗi từ embedding. SFT-BE Stage 0 học bắt chước teacher trên Wikipedia. Mô hình có thể mạnh ở mức semantic similarity tổng quát, nhưng chưa chắc tối ưu cho paper khoa học hoặc các thuật ngữ chuyên ngành hẹp.

Thứ tư, lỗi từ Cross-Encoder. Model `cross-encoder/ms-marco-MiniLM-L6-v2` được huấn luyện cho passage ranking kiểu MS MARCO. Nó phù hợp làm reranker tổng quát nhưng chưa chắc tối ưu cho mọi câu hỏi học thuật.

Thứ năm, lỗi từ LLM verifier. LLM local có thể không khả dụng, trả JSON sai format hoặc chấm điểm thiếu ổn định. Vì vậy, verifier nên được xem là tầng tùy chọn, không phải nguồn chân lý tuyệt đối.

Một nhóm lỗi khác liên quan đến truy vấn quá ngắn hoặc quá chung. Nếu người dùng chỉ nhập một từ như "culture" hoặc "politics", embedding của query không chứa đủ ý định cụ thể. Hệ thống có thể trả về các đoạn cùng chủ đề rộng nhưng không đúng mong muốn thật sự. Với các truy vấn như vậy, cần khuyến khích người dùng nhập câu hỏi cụ thể hơn, hoặc hệ thống cần thêm query expansion/query clarification.

Ngoài ra, có sự khác biệt giữa similarity và relevance. Hai đoạn có thể rất giống nhau về chủ đề nhưng một đoạn không trả lời đúng câu hỏi. Ví dụ, query hỏi "paper nào phản bác giả định X", còn chunk chỉ mô tả giả định X mà không phản bác. Bi-Encoder có thể đưa chunk đó lên cao vì cùng chủ đề, Cross-Encoder có thể giảm điểm nếu đọc kỹ quan hệ query-passage, còn LLM verifier có thể phát hiện rõ hơn nếu context đủ. Đây chính là lý do pipeline nhiều tầng hợp lý hơn một tầng duy nhất.

Một vấn đề nữa là retrieval trên paper khoa học thường cần thông tin vượt quá một chunk. Có những câu hỏi yêu cầu kết hợp định nghĩa ở phần Introduction, phương pháp ở Method và kết quả ở Experiment. Hệ thống hiện tại trả context window gần chunk được chọn, nhưng chưa có multi-hop retrieval. Nếu câu trả lời cần nhiều bằng chứng ở nhiều vị trí xa nhau trong cùng paper, pipeline hiện tại có thể chưa gom đủ context. Hướng cải thiện là truy xuất theo nhiều lượt, hoặc sau khi chọn paper chính, tiếp tục tìm các chunk bổ sung trong cùng paper theo sub-query.

Cuối cùng, việc đánh giá bằng mắt có thể gây thiên lệch. Người làm hệ thống thường dễ chọn các query mà hệ thống trả lời tốt. Vì vậy, để đánh giá nghiêm túc, cần một bộ test cố định gồm cả query dễ, query khó, query có phủ định, query cần con số, query cần entity chính xác và query không có câu trả lời trong corpus. Khi đó mới đo được hệ thống không chỉ tìm đúng khi có đáp án, mà còn biết từ chối khi bằng chứng không đủ.

## 6.9. Nhận xét chung

Kết quả thực nghiệm cho thấy đề tài đã xây dựng được một pipeline hoàn chỉnh từ huấn luyện embedding model đến retrieval system. SFT-BE đạt cosine trung bình 0.93433 với teacher trên 47,959 câu tail Wikipedia. Chỉ mục paper có 3,402 chunk và vector shape $(3402,768)$. Kiểm thử logic reranking pipeline chạy thành công.

Tuy nhiên, đề tài chưa có benchmark retrieval có nhãn cho chỉ mục paper. Vì vậy, báo cáo không kết luận định lượng về Precision@k hoặc MRR. Để đánh giá đầy đủ hơn, cần xây dựng một tập query và nhãn chunk liên quan cho 30 paper, sau đó so sánh các cấu hình: vector-only, vector + lexical, vector + Cross-Encoder và vector + Cross-Encoder + LLM verifier.

---

# CHƯƠNG 7. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

## 7.1. Kết luận chung

Đề tài đã xây dựng được một hệ thống tìm kiếm văn bản tương tự dựa trên ngữ nghĩa, gồm cả mô hình embedding và ứng dụng retrieval. Mô hình SFT-BE được thiết kế theo hướng Bi-Encoder, sử dụng factorized embedding, Transformer encoder Pre-LN, multi-head attention và mean pooling. Mô hình được huấn luyện bằng teacher-student distillation để học embedding space của `all-mpnet-base-v2`.

Phần ứng dụng retrieval sử dụng SFT-BE làm first-stage retriever, sau đó kết hợp lexical boost, Cross-Encoder reranking và LLM verifier tùy chọn. Hệ thống có khả năng lập chỉ mục từ tập paper PDF, nhận truy vấn tự nhiên và trả kết quả có citation.

Kết quả thực nghiệm cho thấy SFT-BE đạt cosine trung bình 0.93433 với teacher trên 10MB cuối Wikipedia, chỉ mục paper có 3,402 chunk và kiểm thử pipeline reranking chạy thành công. Điều này cho thấy hướng tiếp cận của đề tài là khả thi trong phạm vi đồ án.

## 7.2. Ưu điểm của hệ thống

Ưu điểm đầu tiên là hệ thống hỗ trợ semantic search, không chỉ keyword search. Điều này giúp tìm được các đoạn liên quan dù query và tài liệu dùng cách diễn đạt khác nhau.

Ưu điểm thứ hai là SFT-BE có thể precompute document embedding. Nhờ vậy, khi có query mới, hệ thống chỉ cần encode query và so sánh với vector đã lưu, phù hợp cho retrieval.

Ưu điểm thứ ba là pipeline có nhiều tầng xếp hạng. Bi-Encoder lấy candidate nhanh, lexical boost giữ tín hiệu từ khóa, Cross-Encoder rerank chi tiết hơn, LLM verifier kiểm tra bằng chứng khi cần.

Ưu điểm thứ tư là hệ thống có citation và context window. Người dùng có thể biết kết quả đến từ paper nào, section nào, page nào và chunk nào, từ đó kiểm chứng thông tin dễ hơn.

## 7.3. Hạn chế của hệ thống

Hạn chế đầu tiên là hệ thống chưa hỗ trợ đầy đủ mọi định dạng tài liệu. Retrieval system hiện chưa đọc trực tiếp DOCX và chưa có OCR cho PDF scan.

Hạn chế thứ hai là chưa có benchmark retrieval có nhãn cho tập paper. Vì vậy, chưa thể báo cáo Precision@k, Recall@k, MRR hoặc nDCG cho hệ thống end-to-end.

Hạn chế thứ ba là chunking còn đơn giản. Với paper có layout phức tạp, bảng biểu hoặc công thức, chunk có thể không giữ được cấu trúc tốt nhất.

Hạn chế thứ tư là LLM verifier phụ thuộc vào Ollama và model local. Nếu LLM không khả dụng hoặc trả sai JSON, hệ thống phải fallback.

## 7.4. Hướng phát triển trong tương lai

Hướng phát triển đầu tiên là mở rộng document ingestion. Hệ thống nên hỗ trợ DOCX, HTML và OCR cho PDF scan. Với DOCX, có thể giữ heading, paragraph và table để metadata tốt hơn.

Hướng phát triển thứ hai là xây dựng benchmark retrieval riêng. Nhóm có thể tạo các query cho 30 paper, gán nhãn chunk liên quan và đo Precision@k, Recall@k, MRR, nDCG cho từng cấu hình.

Hướng phát triển thứ ba là fine-tune SFT-BE trên dữ liệu gần domain hơn, ví dụ các cặp abstract-title, citation context, hard negatives hoặc query-passage từ paper khoa học.

Hướng phát triển thứ tư là thay vector store cục bộ bằng vector database chuyên dụng như FAISS, Chroma hoặc Milvus để hỗ trợ corpus lớn hơn và tìm kiếm approximate nearest neighbor.

Hướng phát triển thứ năm là cải thiện giao diện. Ứng dụng có thể thêm upload file, quản lý chỉ mục, xem PDF, highlight đoạn liên quan, bộ lọc theo paper/section/page và lịch sử truy vấn.

Cuối cùng, hệ thống có thể phát triển thành một ứng dụng retrieval-augmented generation hoàn chỉnh. Tuy nhiên, nguyên tắc quan trọng cần giữ là mọi câu trả lời phải dựa trên context đã truy xuất và có citation rõ ràng.

---

# TÀI LIỆU THAM KHẢO

1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). *Attention Is All You Need*. NeurIPS.
2. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL.
3. Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP-IJCNLP.
4. Lan, Z., Chen, M., Goodman, S., Gimpel, K., Sharma, P., & Soricut, R. (2020). *ALBERT: A Lite BERT for Self-supervised Learning of Language Representations*. ICLR.
5. Xiong, R., Yang, Y., He, D., Zheng, K., Zheng, S., Xing, C., Zhang, H., Lan, Y., Wang, L., & Liu, T. (2020). *On Layer Normalization in the Transformer Architecture*. ICML.
6. Loshchilov, I., & Hutter, F. (2019). *Decoupled Weight Decay Regularization*. ICLR.
7. Loshchilov, I., & Hutter, F. (2017). *SGDR: Stochastic Gradient Descent with Warm Restarts*. ICLR.
8. Reimers, N., & Gurevych, I. (2020). *Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation*. EMNLP.
9. Cer, D., Diab, M., Agirre, E., Lopez-Gazpio, I., & Specia, L. (2017). *SemEval-2017 Task 1: Semantic Textual Similarity Multilingual and Crosslingual Focused Evaluation*. SemEval.
10. HuggingFace. *Transformers Documentation* và *Datasets Documentation*.
11. Sentence-Transformers. *Pretrained Bi-Encoders and Cross-Encoders for Semantic Search and Reranking*.
12. PyTorch. *torch.nn, torch.optim, Automatic Mixed Precision and Tensor Operations*.
13. HuggingFace model card: `cross-encoder/ms-marco-MiniLM-L6-v2`.
