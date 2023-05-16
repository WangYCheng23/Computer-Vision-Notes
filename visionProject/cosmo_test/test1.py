import cv2
import numpy as np

def readtest():
    videoname = r'C:\Users\admin\Documents\work\visionProject\cosmo_test\belt_video1.mp4'
    capture = cv2.VideoCapture(videoname)
    if capture.isOpened():
        while True:
            ret, frame = capture.read()
            
            # 转灰度图
            gray_frame = cv2.cvtColor(cv2.GaussianBlur(frame, (3,3), 5), cv2.COLOR_BGR2GRAY)
            cv2.imshow('gray_frame', gray_frame)

            # 边缘检测
            edge_frame = cv2.Canny(gray_frame, threshold1=15, threshold2=85, apertureSize=3)
            cv2.imshow('edge_frame', edge_frame)
            
            # 霍夫变换
            # # 1.
            # lines = cv2.HoughLines(edge_frame, 1, np.pi/180, 135)
            # for line in lines:
            #     rho, theta = line[0]
            #     x0 = np.cos(theta) * rho
            #     y0 = np.sin(theta) * rho
            #     x1, x2 = int(x0 + 1000*(-np.sin(theta))), int(x0 - 1000*(-np.sin(theta)))
            #     y1, y2 = int(y0 + 1000*(np.cos(theta))), int(y0 - 1000*(np.cos(theta)))

            #     cv2.line(frame, (x1,y1), (x2,y2), (0,0,255), 2)
            # 2.
            lines = cv2.HoughLinesP(edge_frame, 1, np.pi/180, threshold=45, minLineLength=200, maxLineGap=15)
            for line in lines:  # 遍历所有直线
                x1, y1, x2, y2 = line[0]  # 读取直线两个端点的坐标
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 在原始图像上绘制直线

            # 显示
            cv2.imshow('result', frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            if not ret: break
    else:
        print("Open Video Failed")

if __name__ == "__main__":
    readtest()