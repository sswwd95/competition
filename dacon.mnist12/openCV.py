# 임곗값 처리(이진화)
''' 
이미지의 용량을 줄이기 위해 일정 이상으로 밝은 것 혹은 일정 이상으로 어두운 것을
모두 같은 값으로 만들어버리는 것을 임곗값 처리라고한다.
cv2.threshold()함수를 사용해서 구현하며, 인수를 설정하여 다양한 임곗값 처리를 할 수 있다. 
'''
import numpy as np
import cv2

img = cv2.imread('이미지 경로')
# 첫 번째 인수 : 처리하는 이미지
# 두 번째 인수 : 임곗값
# 세 번째 인수 : 최댓값(maxvalue)
# 네 번째 인수 : 
# THRESH_BINARY -> 픽셀값이 임곗값을 초과하는 경우 해당 픽셀을 maxvalue로 하고, 그 외의 경우에는 0(검은색)으로 한다.
# THRESH_BINARY_INV -> 픽셀값이 임곗값을 초과하는 경우 0으로 설정하고, 그 외의 경우에는 maxvalue로 한다. 
# THRESH_TOZERO -> 픽셀값이 임곗값을 초과하는 경우 임곗값으로 설정하고, 그 외의 경우에는 변경하지 않는다. 
# THRESH_TRUNC -> 픽셀값이 임곗값을 초과하는 경우 변경하지 않고, 그 외의 경우에는 0으로 설정한다. 
# TRHRESH_TOZERO_INV -> 픽셀값이 임곗값을 초과하는 경우 0으로 설정하고, 그 외의 경우에는 변경하지 않는다. 

# 임곗값을 75로, 최댓값을 255
retval, my_img = cv2.threshold(img,75,255,cv2.THRESH_TOZERO)

cv2.imshow('sample', my_img)

