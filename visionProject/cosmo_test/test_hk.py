import cv2

# url是海康威视的rtsp视频流地址，用户名默认为admin，ip默认为192.168.1.64，password改为自己设置的密码
# url = "rtsp://admin:PAAS1234@172.16.12.127/Streaming/Channels/1"
# cap = cv2.VideoCapture(url)
# ret, frame = cap.read()
# while ret:
#     ret, frame = cap.read()
#     cv2.imshow("frame", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cv2.destroyAllWindows()
# cap.release()
url_meike = "rtsp://admin:meike123@172.16.10.78/Streaming/Channels/1"
cap_meike = cv2.VideoCapture(url_meike)
ret, frame = cap_meike.read()
while ret:
    ret, frame = cap_meike.read()
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
cap_meike.release()