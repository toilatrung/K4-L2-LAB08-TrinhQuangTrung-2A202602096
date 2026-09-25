# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: Trịnh Quang Trung - 2A202602096

Công cụ gán nhãn đã dùng: Sửa trực tiếp file nhãn YOLO (`to_label/round1/labels/train/*.txt`), gói bằng `tools/pack_labels.py` (nguồn `yolo-txt` trong `labels/round1/batch.json`). Để rà, tôi vẽ pre-label (có đánh số) lên ảnh phóng to 2×. Tôi cũng chạy riêng một model lớn hơn (`yolov8m` COCO, conf ≥ 0.08) làm **gợi ý phụ** để tìm xe bị bỏ sót, rồi tự kiểm bằng mắt từng box trước khi nhận hoặc loại. Việc rà làm với sự hỗ trợ của trợ lý AI (Claude Code). **Không dùng CVAT Docker local.** Notebook chạy trên máy cá nhân có GPU GTX 1650 thay cho Colab (xem mục 5).

## 1. Dữ liệu và cách chia tập

Camera đặt cố định, lấy mẫu 2.5 frame/giây, nên hai frame cách nhau 0.4 s gần như giống hệt nhau và một chiếc xe ở trong khung hình vài giây (`data/DATA.md`). Nếu chia ngẫu nhiên, cùng một chiếc xe ở cùng vị trí sẽ nằm cả trong pool (được gán và train) lẫn test. Khi đó model được chấm trên chính những xe nó đã học, nên AP50 trên test sẽ bị **đánh giá cao hơn thực tế** (lạc quan): số đo phản ánh khả năng nhớ cảnh hơn là khả năng tổng quát sang xe mới. Vì vậy test gồm 20 ảnh ở 4 đoạn quanh giây 20/60/100/140, bỏ 112 ảnh vùng đệm ±4 s. Ảnh pool gần test nhất vẫn cách 4.4 s, đủ để các xe trong test không trùng với xe trong pool.

## 2. Mô hình khởi đầu lạnh (cold start)

Dòng vòng 0 trong `reports/rounds_table.md`:

| vòng | model | ảnh train | AP50 | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0.771 | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |

Theo `outputs/metrics_round0.json`, ở conf 0.25 model có TP 197, FP 16, FN 206 trên 403 box tham chiếu được chấm. Precision cao nhưng recall chỉ 0.489: **model bỏ sót nhiều hơn là báo nhầm**. Recall theo kích thước cho thấy xe nhỏ gần như bị bỏ qua (R small 0.182 trên 66 box). Ngay cả xe trung bình và lớn cũng chỉ được phát hiện khoảng 55%.

Trong `outputs/compare_round0.jpg` (4 frame test), các box vàng (FN) tập trung ở: (a) xe chiều tới có đèn pha chói, thân xe tối, nhất là các xe gần làn trái (frame_0050, frame_0250); (b) xe xa sát chân trời thành cụm đèn đỏ (frame_0350); (c) xe bị cắt ở mép phải (frame_0250). Box đỏ (FP) của cold start chủ yếu là **box gộp hai xe** ở làn xa, ví dụ cụm xe làn đi xa bên phải frame_0150 và cụm xe trái frame_0350.

Ca cần rà lại tham chiếu trước khi kết luận model sai: ở frame_0250, tham chiếu có box nhỏ ở mép phải (x ≈ 1255–1275) chỉ thấy một vệt sáng. Tôi không chắc đó là xe hay đèn phản chiếu. Tham chiếu do model tạo, chưa được người rà, nên FN ở đây chưa chắc là lỗi của model.

## 3. Chiến lược chọn mẫu

`score = W_U·U + W_A·A + W_D·D` với W = 0.5/0.3/0.2:

- **U**: trung bình độ bất định `u = 1 − |2·conf − 1|` của 5 box khó nhất. U cao nghĩa là có nhiều box model phân vân (conf gần 0.5).
- **A**: số box mơ hồ (0.15 ≤ conf < 0.5), chuẩn hoá theo giá trị lớn nhất trong pool. A ưu tiên ảnh có *nhiều* chỗ phân vân chứ không chỉ một box.
- **D**: khoảng cách thời gian tới ảnh đã gán gần nhất (tối đa 10 s). D ưu tiên đoạn video chưa có nhãn; ở vòng 1, D = 1 cho mọi ảnh.
- **`MIN_GAP_S = 2.0`**: hai ảnh trong cùng lô phải cách nhau ít nhất 2 s. Camera cố định nên ảnh sát nhau là ảnh gần trùng; nếu không có luật này, lô sẽ chứa nhiều cặp ảnh cùng xe.

Bằng chứng trong `reports/SELECTION.md`: frame_0182 (rank 1, 0.9591, A = 1.0), frame_0369 (rank 2, U 0.9315) và frame_0331 (rank 5, A = 1.0, 47 box) được chọn. frame_0372 (rank 6, 0.9101) và frame_0330 (rank 12) bị loại vì cách frame_0369 1.2 s và frame_0331 0.4 s. frame_0195 (rank 268, 0.5721) là cảnh thưa xe mà tôi vẫn muốn xem vì lô toàn cảnh đông. Với ngân sách 5 ảnh, tôi thay frame_0380 và frame_0331 bằng frame_0312 và frame_0099 để tránh gần trùng và giảm công rà.

Điểm bất định **không chứng minh** ảnh đó sẽ cải thiện model. Nó chỉ đo độ phân vân của model hiện tại. Nó bỏ qua xe model không đề xuất box nào (xe tối cắt mép dưới), không tính chi phí rà (ảnh đông xe tốn nhiều công hơn) và không biết nhãn sau khi sửa có khớp phân phối test hay không. Ở vòng này, lô bất định cao vẫn làm AP50 giảm (mục 4).

## 4. Các vòng học chủ động (active learning)

Bảng từ `reports/rounds_table.md`:

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
| 1 | yolov8n fine-tune vong 1..1 | 12 | 300 | 0.719 | -0.053 | 1.000 | 0.137 | 0.240 | 0.000 | 0.139 | 0.342 |

**Mức sửa pre-label vòng 1** (`outputs/round1_diff.md`): model đề xuất 169 box, nhãn cuối 300 box. Có 150 box accepted, 8 edited, 11 deleted (FP của model) và 142 added (FN của model); accept rate 89%. Như vậy lỗi chính của pre-label là **bỏ sót**, không phải box sai: gần một nửa số xe trong nhãn cuối do tôi thêm. Frame thêm nhiều nhất là frame_0369 (+18). Frame xoá nhiều nhất là frame_0331 (−4).

**AP50**: vòng 1 đạt 0.719, **giảm 0.053** so với cold start và so với vòng trước (vòng trước chính là cold start). Mức giảm này lớn hơn ngưỡng nhiễu ~0.01 trong `DATA.md`. Ở conf 0.25, precision tăng lên 1.000 (FP 16 → 0) nhưng recall giảm mạnh 0.489 → 0.137 (TP 197 → 55). Theo kích thước: small 0.182 → 0.000, medium 0.547 → 0.139, large 0.561 → 0.342. Nhóm xe lớn/gần giữ được nhiều nhất; xe nhỏ tệ nhất.

**Một ca đổi sau fine-tune** (`outputs/compare_round1.jpg`): ở frame_0050, cold start có TP 11 / FP 2 / FN 7, còn vòng 1 có TP 3 / FP 0 / FN 15. Hai box đỏ gộp xe ở làn xa đã biến mất (tốt hơn), nhưng hầu hết xe trước đó được phát hiện đúng giờ thành box vàng. Ba frame test còn lại cũng vậy: frame_0150 TP 10 → 2, frame_0250 TP 6 → 1, frame_0350 TP 9 → 2, FP ở cả bốn frame từ 2 → 0. Lý do có thể kiểm:

1. Train từ `yolov8n.pt` với 1 lớp làm đầu phân loại COCO 80 lớp bị khởi tạo lại. Với 12 ảnh, batch 16, 50 epoch, model chỉ có khoảng 50 bước cập nhật; `cls_loss` cuối vẫn khoảng 0.99 (log huấn luyện). Model định vị được nhưng **confidence thấp**, đa số box dưới 0.25.
2. AP50 (không phụ thuộc ngưỡng) chỉ giảm 0.053, trong khi R@0.25 giảm 0.35. Điều này khớp với giả thuyết model xếp hạng box vẫn tạm ổn nhưng thiếu tự tin, không phải học sai vị trí.
3. Bằng chứng từ pool: trong `outputs/selection_round2.csv`, model vòng 1 chỉ đưa ra khoảng 4–12 box có conf ≥ 0.05 mỗi ảnh (ví dụ frame_0031 có 9 box), trong khi ảnh thật có khoảng 25–30 xe.

**Phân biệt ba nguồn bằng chứng**:
- *Quan sát độc lập* (`reports/BLIND_SCAN.md`, khoá trong `blind_lock.json`): trước khi xem pre-label, ghi frame_0182 có khoảng 28 xe và hai vị trí dễ sót là xe tối cắt mép dưới giữa ảnh và xe góc dưới phải trong vệt lóa.
- *Lỗi pre-label đã sửa* (`REVIEW_LOG.csv`, `round1_diff.md`): cả hai vị trí trên thực sự không có box trong pre-label và được thêm (frame_0182: 13 → 25 box, +13, −1). Nhãn cuối có 25 box, gần với ước lượng 28 (chênh lệch chủ yếu ở cụm đèn rất xa cao dưới 16 px mà tôi không gán hết). Các lỗi khác: box gộp hai xe (frame_0182 P12, frame_0312 P9), box trên mặt đường sáng (frame_0331 P18), box trùng (frame_0331 P15), bỏ sót xe tải (frame_0270, frame_0312).
- *Kết quả model sau train* (`metrics_round1.json`, `compare_round1.jpg`): nhãn tốt hơn **không** tự động cho model tốt hơn ở vòng này; model mới không còn FP gộp xe nhưng recall giảm vì lý do huấn luyện ở trên.

**Ca khó theo guideline và cách tôi xử lý nhất quán**:
- Xe tối bị cắt ở mép dưới, chỉ thấy nóc (frame_0182, 0107, 0312, 0331, 0369, 0380, 0392): luôn gán, box chỉ ôm phần trong ảnh.
- Xe chỉ thấy đèn: box theo thân xe đoán được quanh cụm đèn, không ôm vệt đèn trên đường. Ví dụ frame_0270: tôi loại gợi ý phụ ở vùng chói phía dưới xe P9 vì đó là vệt đèn, không phải xe.
- Hai xe xếp chồng ở làn xa: luôn tách hai box.
- Xe rất xa cao dưới 16 px: chỉ gán khi thấy rõ là một cụm xe tách biệt, còn không thì bỏ qua. Loại box này không được tính khi chấm.

## 5. Kết luận và giới hạn

**Kết quả**: vòng 1 kém hơn cold start (AP50 0.719 so với 0.771; R@0.25 0.137 so với 0.489), dù nhãn đã sửa có 300 box so với 169 box pre-label. **Tôi dừng, không train thêm vòng 2 theo cùng cấu hình.** Mức giảm lớn hơn nhiễu và nguyên nhân khả dĩ nằm ở cách huấn luyện (khởi tạo lại đầu phân loại, rất ít bước), không phải thiếu ảnh. Thêm 12 ảnh uncertainty nữa chưa chắc sửa được.

**Hai ca còn yếu hoặc bất định cho vòng sau**:
1. *Xe nhỏ/xa* (R small = 0.000 ở vòng 1): nên chọn các frame có nhiều cụm xe xa ở làn đi xa, ví dụ đoạn cuối video. Chi phí rà cao (mỗi ảnh 30–50 box nhỏ, khó vẽ nhất quán) và các frame liền nhau như frame_0368/0369/0372 là gần trùng, nên chỉ lấy một frame mỗi đoạn ≥ 2 s, tốt hơn là ≥ 4 s.
2. *Cảnh thưa xe và xe tối gần camera*: lô vòng 1 toàn cảnh đông (35–53 box). Nên thêm một frame điểm thấp như frame_0195 (rank 268): chi phí rà thấp (khoảng 15 box) và bù phân phối cảnh. Không chọn thêm frame_0391 (vòng 2 đề xuất, D = 0.04) vì nó cách frame_0392 đã gán chỉ 0.4 s, gần như trùng.

**Giới hạn ảnh hưởng tới kết luận**: tập test chỉ có 20 ảnh (403 box được chấm) ở 4 đoạn thời gian, nên thay đổi nhỏ (< 0.01) không có ý nghĩa, và kết luận chỉ đúng cho 4 đoạn đó. Luật bỏ box cao dưới 16 px khiến chất lượng xe cực xa không được đo. Nhãn tham chiếu do model tạo, chưa được người rà: model của tôi được chấm theo mức *khớp với model tham chiếu*. Nếu tham chiếu cũng bỏ sót xe tối cắt mép (tôi đã gán loại này), thì dự đoán đúng theo guideline vẫn có thể bị tính FP. Cách vẽ box của tôi cũng có thể lệch với tham chiếu (ví dụ xe cắt mép dưới), làm AP50 giảm dù nhãn đúng guideline.

**Nếu AP50 giảm, trước khi train thêm tôi kiểm tra**: (1) phân phối confidence của model vòng 1 trên test: nếu đa số box đúng nằm dưới 0.25 thì vấn đề là hiệu chỉnh/số bước train, không phải nhãn; (2) cấu hình huấn luyện: số bước cập nhật, có freeze backbone hay giữ đầu COCO không, nhiều epoch hơn; (3) nhãn train không có ảnh test, không box lỗi định dạng (`check_submission.py` đạt), và cách vẽ của tôi nhất quán với tham chiếu trên một vài xe test; (4) chạy đối chứng `STRATEGY = "random"` với cùng số ảnh để tách tác động của chiến lược chọn mẫu khỏi tác động của huấn luyện.

**Tự QC và minh bạch về môi trường**: notebook chạy trên máy cá nhân (GTX 1650) thay cho Colab, dùng ultralytics 8.4.153 đã cài sẵn thay vì 8.4.161. Tham số giữ nguyên (`AL_K = 12`, `STRATEGY = "uncertainty"`, `EPOCHS = 50`, `IMGSZ = 960`, batch 16, seed 8). Ultralytics tự tắt AMP trên GTX 1650 (train FP32), nên số đo có thể lệch nhẹ so với Colab. Test labels không bị sửa (`check_submission.py` kiểm hash). Việc gán nhãn không đi qua CVAT như hướng dẫn yêu cầu, và việc rà (kể cả bản quét độc lập) có trợ lý AI hỗ trợ; tôi ghi rõ điều này để người chấm đánh giá phần 2 của rubric cho đúng.
