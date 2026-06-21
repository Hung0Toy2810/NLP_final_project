# 3. Mô hình Bi-Encoder trong bài toán Document Similarity

## 3.1. Kiến trúc Bi-Encoder cho biểu diễn ngữ nghĩa văn bản

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

## 3.2. Xây dựng mô hình SFT-BE

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

## 3.3. Factorized Embedding và Positional Encoding

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

## 3.4. Transformer Encoder trong SFT-BE

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

## 3.5. Mean Pooling và Sentence Embedding

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

## 3.6. Cơ chế huấn luyện bằng Teacher-Student Distillation

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

## 3.7. Dữ liệu huấn luyện và pipeline xử lý

Dữ liệu train chính là các sentence được trích từ Wikipedia tiếng Anh. Pipeline chuẩn bị dữ liệu gồm các bước: tải Wikipedia, tách article thành sentence, lọc các câu quá ngắn hoặc quá dài, tokenize bằng `bert-base-uncased`, padding/truncate về 128 token và lưu cache để train nhiều lần mà không phải xử lý lại.

Với mỗi câu $x$, tokenizer tạo:

$$\text{input\_ids}(x) = (t_1,t_2,\ldots,t_L)$$

$$\text{attention\_mask}(x) = (m_1,m_2,\ldots,m_L)$$

Trong lần chạy thực nghiệm của đồ án, quá trình huấn luyện bị dừng sớm trước khi hoàn tất một epoch, nên các bước đánh giá cuối epoch như validation loss hoặc STS-B không được dùng làm kết quả báo cáo chính thức. Vì vậy, phần kết quả bên dưới chỉ dựa trên training log và một bài test bổ sung trên 10MB dữ liệu Wikipedia lấy từ cuối dataset.

---

## 3.8. Tối ưu hóa mô hình

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

## 3.9. Theo dõi huấn luyện và test thực nghiệm

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

## 3.10. Vai trò của SFT-BE trong hệ thống retrieval

Sau khi train xong, SFT-BE có thể được dùng như embedding model cho hệ thống document retrieval. Corpus được chia thành các chunk, mỗi chunk được encode thành vector 768 chiều rồi lưu vào vector database. Khi người dùng nhập query, hệ thống encode query thành vector và tìm top-$K$ chunk gần nhất.

Nếu toàn bộ document embedding đã được L2-normalize và xếp thành ma trận:

$$Z = \begin{bmatrix}z_1^\top\\z_2^\top\\\vdots\\z_N^\top\end{bmatrix} \in \mathbb{R}^{N \times 768}$$

Với query embedding đã normalize là $z_q$, điểm similarity với toàn bộ corpus có thể tính bằng một phép nhân ma trận:

$$s = Zz_q$$

Trong đó $s_i$ chính là cosine similarity giữa query và document chunk thứ $i$. Đây là lợi thế rất lớn của bi-encoder: sau khi pre-compute document embedding, inference chủ yếu chỉ còn là encode query và nearest neighbor search.

Tuy vậy, cần nhìn đúng vai trò của mô hình. SFT-BE mạnh ở bước retrieval ban đầu, tức chọn ra vùng tài liệu có khả năng liên quan. Mô hình không phải là reasoning model và cũng không thay thế hoàn toàn cross-encoder. Với các trường hợp cần phân biệt phủ định, đảo vai chủ thể - đối tượng, hoặc những khác biệt ngữ nghĩa rất nhỏ, có thể cần thêm reranker hoặc verifier phía sau.

---

## 3.11. Kết luận

SFT-BE là một mô hình bi-encoder được thiết kế cho bài toán document similarity và semantic retrieval. Mô hình encode mỗi văn bản độc lập thành vector 768 chiều, sau đó dùng cosine similarity để đo độ gần về mặt ngữ nghĩa. Thiết kế bi-encoder giúp document embedding có thể được pre-compute, nhờ đó hệ thống truy hồi nhanh hơn nhiều so với cách dùng cross-encoder cho mọi cặp query-document.

Về kiến trúc, SFT-BE kết hợp factorized embedding 128 -> 768, sinusoidal positional encoding, 6 layer Pre-LN Transformer encoder, multi-head self-attention 12 heads, FFN 3072 chiều và mean pooling có attention mask. Các lựa chọn này giúp mô hình giữ được năng lực biểu diễn của Transformer nhưng giảm số tham số và chi phí so với một encoder lớn hơn như BERT-base.

Về huấn luyện, mô hình dùng teacher-student distillation với teacher `sentence-transformers/all-mpnet-base-v2`. Student được tối ưu bằng cosine distillation loss:

$$\mathcal{L}_{\text{batch}} = 1 - \frac{1}{B}\sum_{i=1}^{B}\operatorname{cos}(S_\theta(x_i),T(x_i))$$

Cách train này cho phép student học lại embedding space giàu ngữ nghĩa của teacher mà không cần nhãn cặp câu thủ công. Kết quả là SFT-BE trở thành một retrieval encoder gọn, nhanh và phù hợp để làm tầng tìm kiếm ban đầu trong hệ thống document similarity.

## Tài liệu tham khảo

1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). *Attention Is All You Need*. NeurIPS.
2. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL.
3. Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP-IJCNLP.
4. Lan, Z., Chen, M., Goodman, S., Gimpel, K., Sharma, P., & Soricut, R. (2020). *ALBERT: A Lite BERT for Self-supervised Learning of Language Representations*. ICLR.
5. Xiong, R., Yang, Y., He, D., Zheng, K., Zheng, S., Xing, C., Zhang, H., Lan, Y., Wang, L., & Liu, T. (2020). *On Layer Normalization in the Transformer Architecture*. ICML.
6. Loshchilov, I., & Hutter, F. (2019). *Decoupled Weight Decay Regularization*. ICLR.
7. Reimers, N., & Gurevych, I. (2020). *Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation*. EMNLP.
8. Cer, D., Diab, M., Agirre, E., Lopez-Gazpio, I., & Specia, L. (2017). *SemEval-2017 Task 1: Semantic Textual Similarity Multilingual and Crosslingual Focused Evaluation*. SemEval.
