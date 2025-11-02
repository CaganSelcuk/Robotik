import cv2
import numpy as np
import math

class ObjectDetectionApp:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)

        self.color_ranges = {
            "red": ([0, 100, 100], [10, 255, 255]),
            "white": ([0, 0, 200], [180, 50, 255]),
            "black": ([0, 0, 0], [180, 255, 50]),
            "green": ([40, 100, 100], [80, 255, 255])
        }

        self.current_color = "red"
        self.calibration_mode = False
        self.hsv_values = [0, 100, 100]

    def get_color_range(self, color_name):
        return self.color_ranges.get(color_name, self.color_ranges["red"])

    def detect_by_color(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower, upper = self.get_color_range(self.current_color)

        if self.current_color == "red":
            lower1 = np.array([0, 100, 100])
            upper1 = np.array([10, 255, 255])
            lower2 = np.array([170, 100, 100])
            upper2 = np.array([180, 255, 255])

            mask1 = cv2.inRange(hsv, lower1, upper1)
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask

    def find_contours(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 500:
                return largest_contour
        return None

    def calculate_center(self, contour):
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
        return None

    def find_capture_region(self, contour, center):
        if center is None:
            return None, None, None

        try:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = box.astype(int)

            min_distance = float('inf')
            best_line = None
            best_angle_h = 0
            best_angle_v = 0

            for i in range(4):
                pt1 = tuple(box[i])
                pt2 = tuple(box[(i + 1) % 4])

                mid_x = (pt1[0] + pt2[0]) // 2
                mid_y = (pt1[1] + pt2[1]) // 2
                mid_point = (mid_x, mid_y)

                distance = math.dist(center, mid_point)

                if distance < min_distance:
                    min_distance = distance
                    best_line = (pt1, pt2)

                    dx = pt2[0] - pt1[0]
                    dy = pt2[1] - pt1[1]

                    angle_horizontal = math.degrees(math.atan2(dy, dx)) % 180
                    if angle_horizontal < 0:
                        angle_horizontal += 180

                    angle_vertical = abs(90 - angle_horizontal)

                    best_angle_h = angle_horizontal
                    best_angle_v = angle_vertical

            return best_line, best_angle_h, best_angle_v
        except:
            return None, 0, 0

    def create_calibration_window(self, frame):

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        h = self.hsv_values[0]
        s = self.hsv_values[1]
        v = self.hsv_values[2]

        lower = np.array([max(0, h - 10), max(0, s - 50), max(0, v - 50)])
        upper = np.array([min(180, h + 10), min(255, s + 50), min(255, v + 50)])

        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(frame, frame, mask=mask)

        # Bilgi yaz
        cv2.putText(result, f"H: {h} S: {s} V: {v}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(result, "Adjust trackbars - Press C to save and exit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(result, "Press Q to quit without saving", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return result, lower, upper

    def create_display_panels(self, frame, contour, center, capture_line, angle_h, angle_v, mask=None):
        height, width = frame.shape[:2]
        panel_height = height // 2
        panel_width = width // 2

        display = np.zeros((height, width, 3), dtype=np.uint8)

        panel1 = frame.copy()
        if contour is not None:
            cv2.drawContours(panel1, [contour], -1, (0, 255, 0), 3)
        if center is not None:
            cv2.circle(panel1, center, 6, (255, 0, 0), -1)
            cv2.putText(panel1, f"Center: {center}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if mask is not None:
            panel2 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cv2.putText(panel2, "COLOR MASK", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            panel2 = np.zeros_like(frame)

        panel3 = frame.copy()
        if center is not None:
            cv2.circle(panel3, center, 6, (255, 0, 0), -1)
            cv2.putText(panel3, f"Center: {center}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if capture_line is not None:
            cv2.line(panel3, capture_line[0], capture_line[1], (0, 0, 255), 3)
            cv2.putText(panel3, f"Angle H: {angle_h:.1f}°", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(panel3, f"Angle V: {angle_v:.1f}°", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        panel4 = np.zeros_like(frame)
        info_y = 40
        cv2.putText(panel4, "OBJECT DETECTION INFO", (10, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        info_y += 40

        if center is not None:
            cv2.putText(panel4, f"Center: {center}", (10, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            info_y += 30
            cv2.putText(panel4, f"Angle H: {angle_h:.1f}°", (10, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            info_y += 30
            cv2.putText(panel4, f"Angle V: {angle_v:.1f}°", (10, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            info_y += 30
        else:
            cv2.putText(panel4, "NO OBJECT DETECTED", (10, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            info_y += 40
            cv2.putText(panel4, "Try:", (10, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            info_y += 25
            cv2.putText(panel4, "1. Press C for calibration", (10, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            info_y += 20
            cv2.putText(panel4, "2. Change color (R/W/B/G)", (10, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            info_y += 20
            cv2.putText(panel4, "3. Adjust object lighting", (10, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        info_y += 40
        cv2.putText(panel4, f"Color: {self.current_color.upper()}", (10, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(panel4, "Press C to calibrate colors", (10, info_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(panel4, "Press R/W/B/G to change color", (10, info_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(panel4, "Press Q to quit", (10, info_y + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        display[0:panel_height, 0:panel_width] = cv2.resize(panel1, (panel_width, panel_height))
        display[0:panel_height, panel_width:width] = cv2.resize(panel2, (panel_width, panel_height))
        display[panel_height:height, 0:panel_width] = cv2.resize(panel3, (panel_width, panel_height))
        display[panel_height:height, panel_width:width] = cv2.resize(panel4, (panel_width, panel_height))

        return display

    def run_calibration(self):
        cv2.namedWindow('Color Calibration')

        cv2.createTrackbar('H', 'Color Calibration', 0, 180, self.on_trackbar)
        cv2.createTrackbar('S', 'Color Calibration', 100, 255, self.on_trackbar)
        cv2.createTrackbar('V', 'Color Calibration', 100, 255, self.on_trackbar)

        cv2.setTrackbarPos('H', 'Color Calibration', self.hsv_values[0])
        cv2.setTrackbarPos('S', 'Color Calibration', self.hsv_values[1])
        cv2.setTrackbarPos('V', 'Color Calibration', self.hsv_values[2])

        print("Calibration mode started!")
        print("Adjust trackbars until your object is clearly visible in white")
        print("Press C to save and exit, Q to quit without saving")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            h = cv2.getTrackbarPos('H', 'Color Calibration')
            s = cv2.getTrackbarPos('S', 'Color Calibration')
            v = cv2.getTrackbarPos('V', 'Color Calibration')
            self.hsv_values = [h, s, v]

            cal_frame, lower, upper = self.create_calibration_window(frame)
            cv2.imshow('Color Calibration', cal_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                self.color_ranges[self.current_color] = (lower.tolist(), upper.tolist())
                print(f"New color range saved for {self.current_color}:")
                print(f"Lower: {lower}")
                print(f"Upper: {upper}")
                break
            elif key == ord('q'):
                break

        cv2.destroyWindow('Color Calibration')
        self.calibration_mode = False

    def on_trackbar(self, val):
        pass

    def run(self):
        print("Object Detection Application Started!")
        print("Controls:")
        print("R - Red detection")
        print("W - WHITE detection")
        print("B - BLACK detection")
        print("G - Green detection")
        print("C - Color calibration mode")
        print("Q - Quit")

        if not self.cap.isOpened():
            print("Error: Could not open camera!")
            return

        print("Camera opened successfully!")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Camera error!")
                break


            if self.calibration_mode:
                self.run_calibration()
                continue

            mask = self.detect_by_color(frame)
            contour = self.find_contours(mask)

            center = None
            capture_line = None
            angle_h = 0
            angle_v = 0

            if contour is not None:
                center = self.calculate_center(contour)
                if center is not None:
                    capture_line, angle_h, angle_v = self.find_capture_region(contour, center)


            display = self.create_display_panels(frame, contour, center, capture_line, angle_h, angle_v, mask)

            cv2.imshow('Object Detection System', display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.calibration_mode = True
            elif key == ord('r'):
                self.current_color = "red"
                print("Switched to RED detection")
            elif key == ord('w'):
                self.current_color = "white"
                print("Switched to WHITE detection")
            elif key == ord('b'):
                self.current_color = "black"
                print("Switched to BLACK detection")
            elif key == ord('g'):
                self.current_color = "green"
                print("Switched to GREEN detection")

        self.cap.release()
        cv2.destroyAllWindows()
        print("Application closed successfully!")


if __name__ == "__main__":
    app = ObjectDetectionApp()
    app.run()