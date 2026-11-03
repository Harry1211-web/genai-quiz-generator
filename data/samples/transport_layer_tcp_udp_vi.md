# Bài 3. Tầng giao vận: TCP và UDP

Tài liệu mẫu do tác giả repo tự viết để demo, phát hành theo giấy phép CC BY 4.0 (Vietnamese version of transport_layer_tcp_udp.md).

## 3.1 Vai trò của tầng giao vận

Tầng giao vận cung cấp kênh truyền logic giữa các tiến trình chạy trên hai máy khác nhau. Trong khi tầng mạng chuyển gói tin giữa hai máy, tầng giao vận chuyển dữ liệu giữa hai tiến trình. Mỗi tiến trình được xác định bằng địa chỉ IP kết hợp với số hiệu cổng. Số hiệu cổng là một số nguyên 16 bit, nhận giá trị từ 0 đến 65535. Các cổng từ 0 đến 1023 được gọi là cổng thông dụng và được dành cho các dịch vụ phổ biến. Ví dụ, HTTP dùng cổng 80, HTTPS dùng cổng 443 và SSH dùng cổng 22.

## 3.2 Giao thức UDP

UDP là giao thức không hướng kết nối: bên gửi không cần thiết lập kết nối trước khi gửi dữ liệu. UDP không đảm bảo dữ liệu đến nơi, không đảm bảo thứ tự và không kiểm soát tắc nghẽn. Đổi lại, phần đầu của UDP chỉ dài 8 byte nên chi phí xử lý rất thấp. UDP phù hợp với các ứng dụng cần độ trễ thấp và chấp nhận mất một phần dữ liệu, như gọi thoại, truyền hình trực tiếp và trò chơi trực tuyến. Hệ thống tên miền DNS cũng thường dùng UDP vì mỗi truy vấn chỉ gồm một yêu cầu và một phản hồi ngắn.

## 3.3 Giao thức TCP và bắt tay ba bước

TCP là giao thức hướng kết nối và truyền tin cậy. Trước khi truyền dữ liệu, hai bên thực hiện bắt tay ba bước. Bước một, máy khách gửi gói SYN kèm số thứ tự ban đầu. Bước hai, máy chủ đang lắng nghe sẽ trả lời bằng gói SYN-ACK, vừa xác nhận số thứ tự của máy khách vừa gửi số thứ tự ban đầu của mình. Bước ba, máy khách gửi gói ACK để xác nhận. Bắt tay ba bước giúp hai bên đồng bộ số thứ tự ban đầu và xác nhận rằng cả hai chiều truyền đều hoạt động. Nếu máy khách gửi SYN mà không nhận được SYN-ACK, nguyên nhân thường là máy chủ không lắng nghe cổng đó hoặc tường lửa đã chặn gói tin.

## 3.4 Truyền tin cậy trong TCP

TCP đánh số thứ tự cho từng byte dữ liệu. Bên nhận gửi lại số xác nhận bằng số thứ tự của byte tiếp theo mà nó mong đợi. Nếu bên gửi không nhận được xác nhận trước khi hết thời gian chờ, nó sẽ gửi lại đoạn dữ liệu đó. Khi bên gửi nhận được ba xác nhận trùng lặp cho cùng một số thứ tự, nó suy ra một đoạn đã bị mất và gửi lại ngay mà không chờ hết thời gian, cơ chế này gọi là truyền lại nhanh.

## 3.5 Điều khiển luồng và kiểm soát tắc nghẽn

Điều khiển luồng ngăn bên gửi làm tràn bộ đệm của bên nhận. Bên nhận thông báo kích thước cửa sổ nhận trong mỗi gói tin, và bên gửi không được gửi quá lượng dữ liệu chưa xác nhận bằng kích thước này. Kiểm soát tắc nghẽn thì bảo vệ mạng chứ không bảo vệ bên nhận. Ở giai đoạn khởi động chậm, cửa sổ tắc nghẽn tăng gấp đôi sau mỗi chu kỳ khứ hồi. Khi cửa sổ vượt ngưỡng, TCP chuyển sang giai đoạn tránh tắc nghẽn và chỉ tăng cửa sổ thêm một đoạn sau mỗi chu kỳ khứ hồi. Khi phát hiện mất gói, TCP giảm cửa sổ tắc nghẽn để giảm tải cho mạng.

## 3.6 Đóng kết nối TCP

Để đóng kết nối, mỗi bên gửi một gói FIN và chờ bên kia xác nhận bằng ACK, nên quá trình đóng thường gồm bốn gói tin. Bên chủ động đóng kết nối sẽ ở trạng thái TIME_WAIT trong một khoảng thời gian để bảo đảm gói ACK cuối cùng đến nơi và các gói cũ còn lưu lại trong mạng không bị nhầm với kết nối mới.
