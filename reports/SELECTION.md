# Vì sao chọn lô này?

Nguồn: `outputs/selection_round1.csv` (268 ứng viên, model cold start `yolov8n` COCO), contact sheet `outputs/selection_round1.jpg`, `to_label/round1/batch.json`. Trọng số mặc định W_U = 0.5, W_A = 0.3, W_D = 0.2; ở vòng 1 chưa có ảnh nào được gán nên D = 1.0 cho mọi frame, thứ hạng chỉ do U và A quyết định. Trong 50 dòng đầu, score chỉ trải từ 0.9591 (rank 1) xuống 0.8203 (rank 50) và không có frame nào `empty = True`.

**Top 5 nếu chỉ có ngân sách rà năm ảnh** (xếp theo thứ tự tôi sẽ rà):

| thứ tự | frame | rank | score | t (s) | lý do |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | frame_0182.jpg | 1 | 0.9591 | 72.8 | Score cao nhất, A = 1.0 (18 box mơ hồ trên 28 box). Đoạn giữa video, giao thông vừa phải, nên rà một ảnh ít tốn công mà vẫn thấy nhiều ca bất định. |
| 2 | frame_0369.jpg | 2 | 0.9324 | 147.6 | U = 0.9315, 43 box, 16 box mơ hồ; đại diện đoạn cuối video, đông xe nhất. |
| 3 | frame_0326.jpg | 4 | 0.9155 | 130.4 | U = 0.931, 39 box. Tôi **bỏ frame_0380** (rank 3, 0.917) vì nó cùng đoạn đông xe cuối video với frame_0369 (cách 4.4 s), cảnh và kiểu lỗi gần như lặp lại; rà frame_0326 thêm được một đoạn thời gian khác. |
| 4 | frame_0312.jpg | 7 | 0.9100 | 124.8 | A = 1.0, 18 box mơ hồ; có xe tải thùng lớn ở chiều đi xa, một loại xe mà các frame khác ít có. Tôi **bỏ frame_0331** (rank 5, 0.9154) vì cách frame_0326 chỉ 2.0 s: nhiều xe trong hai ảnh là cùng một chiếc, gán cả hai tốn gấp đôi công mà model học thêm ít. |
| 5 | frame_0099.jpg | 8 | 0.9063 | 39.6 | U = 0.946 (cao thứ ba trong top 10) và là frame đầu tiên của đoạn sớm (< 45 s), giúp lô không dồn vào nửa sau video. |

**Ba frame thuộc lô 12 ảnh model chọn và bằng chứng**

- `frame_0182.jpg`: rank 1, score 0.9591, U 0.9182, A 1.0, `n_ambiguous` 18/28. Trên contact sheet, ảnh có xe tối bị cắt ở mép dưới và xe trong vệt lóa góc phải. Sau khi rà: pre-label có 13 box, nhãn cuối 25 box (12 accepted, 1 deleted, 13 added theo `outputs/round1_diff.md`). Điểm bất định cao trùng với chỗ model thực sự thiếu box.
- `frame_0369.jpg`: rank 2, score 0.9324, U 0.9315, A 0.8889, 43 box ở conf ≥ 0.05 nhưng chỉ 14 box ≥ 0.25 được đưa vào pre-label. Nhãn cuối 32 box, thêm 18 box (nhiều nhất lô). Đây là frame có khoảng cách lớn nhất giữa số box model “nghĩ tới” và số box nó đủ tự tin để đề xuất.
- `frame_0331.jpg`: rank 5, score 0.9154, A 1.0, 47 box. Là frame có nhiều FP nhất: xoá 4 box, gồm một box trên mặt đường sáng (P18) và một box trùng (P15). Nó vẫn vào lô vì `MIN_GAP_S = 2.0` s và frame_0331 cách frame_0326 đúng 2.0 s. Theo tôi đây là trường hợp ngưỡng 2 s hơi thấp với camera cố định.

**Một frame có điểm cao nhưng không chọn, và một frame điểm thấp vẫn nên xem**

- `frame_0372.jpg` (rank 6, score 0.9101) và `frame_0330.jpg` (rank 12, 0.8899, 53 box, nhiều box nhất top 50) không vào lô vì chỉ cách frame_0369 1.2 s và frame_0331 0.4 s. Đó là ảnh gần trùng: cùng xe, cùng vị trí, gán thêm gần như không có thông tin mới. Luật `MIN_GAP_S` loại chúng là đúng.
- `frame_0195.jpg` (rank 268/268, score 0.5721, chỉ 3 box mơ hồ): cảnh thưa xe, chiều tới chỉ khoảng 4 xe gần và vài xe xa. Score thấp không có nghĩa là model đúng; nó chỉ nói model *tự tin*. Lô 12 ảnh toàn cảnh đông xe (35–53 box), nên model chưa thấy cảnh thưa. Tôi sẽ đưa một frame kiểu này vào vòng sau: chi phí rà thấp (khoảng 15 box) và giúp phủ phân phối cảnh.

**Điều phép chọn này chưa chứng minh về chất lượng mô hình**

Uncertainty sampling chỉ cho biết model *phân vân* ở đâu (conf gần 0.5), không cho biết ảnh đó sẽ cải thiện model, cũng không phát hiện những xe model bỏ sót mà không có box nào (ví dụ xe tối cắt mép dưới ở frame_0182 không có box ≥ 0.25). Score cũng không đo chi phí rà: frame đông xe có score cao nhưng tốn nhiều công hơn. Thực tế sau fine-tune AP50 trên test giảm từ 0.771 xuống 0.719 (`reports/rounds_table.md`), nên việc chọn đúng ảnh bất định chưa đủ để kết luận lô này giúp model tốt hơn.
