import cv2 
import numpy as np
from tensorflow import keras
from imutils.perspective import order_points
from sudoku_solver import solve

# Load model for digit classification
classifier = keras.models.load_model('../model/digit-classifier.h5')

# Read Video
cap = cv2.VideoCapture(0)

# VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('../outputs/output.avi', fourcc, 24.0, (int(cap.get(3)), int(cap.get(4))))
flag = 0


if not cap.isOpened():
    print('Camera not found')
    
while True:
    
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Convert the frame to gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Blur image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptative thresholding
    thres = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thres = cv2.bitwise_not(thres)
    
    # Find contours
    contours, hierarchy = cv2.findContours(thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area (puzzle outline)
    cnts = sorted(contours, key=cv2.contourArea, reverse=True)
    
    puzzleCnt = None
    
    for c in cnts:
       area = cv2.contourArea(c)
       if area > 25000:
           peri = cv2.arcLength(c, True)
           polygone = cv2.approxPolyDP(c, 0.01 * peri, True)
           if area > 0 and len(polygone) == 4:
               puzzleCnt = polygone
               maxArea = area
               
    if puzzleCnt is not None:
        cv2.drawContours(frame, [puzzleCnt], 0, (0, 255, 0), 2)
        # compute the perspective transform matrix and then apply it
        maxWidth, maxHeight = 450, 450
        
        rect = order_points(puzzleCnt.reshape(4, 2))
        (tl, tr, br, bl) = rect
        dst = np.array([[0, 0],[maxWidth - 1, 0],[maxWidth - 1, maxHeight - 1],[0, maxHeight - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(frame, M, (maxWidth, maxHeight))

        puzzle = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        puzzle = cv2.GaussianBlur(puzzle, (5, 5), 1)
        puzzle = cv2.adaptiveThreshold(puzzle, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 2)
        
        if flag == 0:
            # Press 's' to solve
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                # Get cell coordinates
                cellWidth = int((rect[1][0] - rect[0][0]) / 9)
                cellHeight = int((rect[1][1]) - rect[0][1] / 9)
                
                # Split puzzle grid into cells
                rows = np.vsplit(puzzle, 9)
                boxes = []
                for r in rows:
                    cols = np.hsplit(r, 9)
                    for box in cols:
                        boxes.append(box)
                # Digit classifier
                digits = []
                values = []
                pos_array = []
                for i in range(1, len(boxes) + 1):
                    cell = np.asarray(boxes[i-1])
                    cell = cell[6:cell.shape[0] - 6, 6:cell.shape[1] - 6]
                    cell = cv2.resize(cell, (28, 28))
                    cell = cell.reshape(1, 28, 28, 1)
                    cell = np.array(cell, dtype='float32')
                    if cell.sum() > 10000:
                        prediction = classifier.predict_classes(cell)
                        values.append(int(prediction[0]))
                        pos_array.append(0)
                    else:
                        values.append(0)
                        pos_array.append(1)
                    if i % 9 == 0:
                        print(i, i%8)
                        digits.append(values)
                        values = []
                # Solve Sudoku
                status = solve(digits)
                 # Display results
                if status is not None:
                    flag = 1
                    digits_array = [int(v) for r in digits for v in r]
                    dsp_digits = [int(a * b) for a, b in zip(pos_array, digits_array)]
                    dsp_results = [dsp_digits[j:j+9] for j in range(0, 81, 9)]
                    
                    board = np.zeros(shape=(maxWidth, maxHeight, 3), dtype='float32')
                    for y in range(9):
                        for x in range(9):
                            if dsp_results[y][x] != 0:
                                cv2.putText(board, "{:d}".format(dsp_results[y][x]), 
                                           ((x) * 50 + 10, (y + 1) * 50 - 20), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,  cv2.LINE_AA)           
                    M = cv2.getPerspectiveTransform(dst, rect)
                    h, w, c = frame.shape
                    board = cv2.warpPerspective(board, M, (w, h))                   
                    dst = cv2.addWeighted(board.astype('uint8'),1,frame.astype('uint8'),0.6,0)
                    while True:
                        cv2.imshow('frame', dst)
                        out.write(dst)
                        # Press 'q' to stop displaying result
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            cv2.imshow("frame", frame)
                            out.write(frame)
                            break
            else:
                cv2.imshow("frame", frame)
                out.write(frame)
    else:
        flag = 0
        cv2.imshow("frame", frame)
        out.write(frame)

    # Press 'q' to finish the program
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break


out.release()
cap.release()
cv2.destroyAllWindows()