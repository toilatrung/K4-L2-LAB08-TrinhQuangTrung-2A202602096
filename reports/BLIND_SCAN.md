# Quét độc lập trước khi xem pre-label

Frame: frame_0182.jpg (trong `to_label/round1/images/train/`, t = 72.8 s, rank 1 trong `selection_round1.csv`)

Số xe nhìn thấy bằng mắt: khoảng 28 xe. Chiều đi tới camera (bên trái dải phân cách) khoảng 15 xe, chiều đi xa (bên phải, đèn hậu đỏ) khoảng 9 xe, và khoảng 4 cặp đèn rất xa sát đường chân trời phía trái (x ≈ 210–400, y ≈ 280–300) chỉ còn chấm đèn, cao dưới 16 px.

Hai vị trí dễ bị AI bỏ sót hoặc vẽ sai, kèm mô tả xe:

1. Mép dưới, giữa ảnh (x ≈ 480–625, y ≈ 630–720): xe SUV/van màu tối chỉ thấy nóc và nửa trên thân xe, bị cắt ở mép dưới ảnh, không có đèn pha nhìn thấy. Dễ bị bỏ sót vì tối và bị cắt; nếu có box thì phải chỉ ôm phần nằm trong ảnh.
2. Góc dưới bên phải (x ≈ 950–1140, y ≈ 615–720): xe đi xa bị cắt ở mép dưới, nằm trong vệt lóa xanh của ống kính và có đèn pha rất sáng ở đáy ảnh. Dễ bị bỏ sót hoặc box bị kéo ôm cả vệt sáng đèn trên mặt đường. Ngoài ra, xe trên nhánh rẽ bên phải (x ≈ 980–1025, y ≈ 295–315) chỉ thấy đèn hậu đỏ, có thể bị bỏ sót.
