# Sudoku Solver

Sudoku solver using OpenCV and Deep Learning


## Dependencies** <br>
The application has been developed using python 3.6 and the following libraries:
- Numpy
- OpenCV 4.4.0
- Tensorflow 2.0.0

## Steps**
1. Capture live stream from camera
2. Localize and extract sudoku grid from video stream
3. Given the sudoku board location, localize each of the cells (9 rows x 9 columns = 81 cells)
4. For each cell in the sudoku puzzle, determine if a digit exists in cell and, if so, identify the digit using a CNN
5. Given the identified digits, create a numeric grid and solve it using a bactracking algorithm
6. Display solved sudoku puzzle over the initial image




