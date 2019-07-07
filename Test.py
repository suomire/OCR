import cv2
import numpy as np
import operator
import os

MIN_CONTOUR_AREA = 100
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30


class ContourWithData():
    npa_contour = None
    bounding_rect = None
    int_rect_x = 0
    int_rect_y = 0
    int_rect_width = 0
    int_rect_height = 0
    contour_area = 0.0

    def calculate_rect_info(self):  # calculate bounding rect info
        [int_x, int_y, int_width, int_height] = self.bounding_rect
        self.int_rect_x = int_x
        self.int_rect_y = int_y
        self.int_rect_width = int_width
        self.int_rect_height = int_height

    def is_contour_valid(self):
        if self.contour_area < MIN_CONTOUR_AREA: return False
        return True


def main():
    all_contours_with_data = []
    v_conts = []

    npa_classifications = np.loadtxt("classifications.txt", np.float32)
    npa_flattened_images = np.loadtxt("flattened_images.txt", np.float32)

    npa_classifications = npa_classifications.reshape((npa_classifications.size, 1))
    k_nearest = cv2.ml.KNearest_create()
    k_nearest.train(npa_flattened_images, cv2.ml.ROW_SAMPLE, npa_classifications)
    img_testing_numbers = cv2.imread("images/NUMS.png")

    if img_testing_numbers is None:
        print("error: image not read from file \n\n")
        os.system("pause")
        return

    img_gray = cv2.cvtColor(img_testing_numbers, cv2.COLOR_BGR2GRAY)
    img_blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)

    bw_img = cv2.adaptiveThreshold(img_blurred,
                                   255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV,
                                   11, 2)

    bw_img_copy = bw_img.copy()

    npa_contours, npa_hierarchy = cv2.findContours(bw_img_copy,
                                                   cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)
    # collect data aboun every contour
    for npaContour in npa_contours:
        contour_with_data = ContourWithData()
        contour_with_data.npa_contour = npaContour
        contour_with_data.bounding_rect = cv2.boundingRect(contour_with_data.npa_contour)
        contour_with_data.calculate_rect_info()
        contour_with_data.contour_area = cv2.contourArea(contour_with_data.npa_contour)
        all_contours_with_data.append(contour_with_data)
    # check every contour
    for contour_with_data in all_contours_with_data:
        if contour_with_data.is_contour_valid():
            v_conts.append(contour_with_data)

    v_conts.sort(key=operator.attrgetter("int_rect_x"))
    str_final_string = ""

    #  for contour_with_data in v_conts:
    for i in range(0, np.size(v_conts)):
        cv2.rectangle(img_testing_numbers,
                      (v_conts[i].int_rect_x, v_conts[i].int_rect_y),
                      (v_conts[i].int_rect_x + v_conts[i].int_rect_width,
                       v_conts[i].int_rect_y + v_conts[i].int_rect_height),
                      (0, 255, 0),
                      2)

        img_char = bw_img[
                   v_conts[i].int_rect_y: v_conts[i].int_rect_y + v_conts[i].int_rect_height,
                   v_conts[i].int_rect_x: v_conts[i].int_rect_x + v_conts[i].int_rect_width]

        img_char_resized = cv2.resize(img_char, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))

        npa_img_char_res = img_char_resized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))

        npa_img_char_res = np.float32(npa_img_char_res)

        retval, npa_results, neigh_resp, dists = k_nearest.findNearest(npa_img_char_res,
                                                                       k=1)

        str_current_char = str(chr(int(npa_results[0][0])))
        if i > 0:
            x1 = v_conts[i - 1].int_rect_x + v_conts[i - 1].int_rect_width
            x2 = v_conts[i].int_rect_x
            if (x2 - x1) > 18:
                str_final_string = str_final_string + " "

        str_final_string = str_final_string + str_current_char
    print("\n" + str_final_string + "\n")

    cv2.imshow("img_testing_numbers", img_testing_numbers)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return


if __name__ == "__main__":
    main()
