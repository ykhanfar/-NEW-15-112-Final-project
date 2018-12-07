# -NEW-15-112-Final-project
This project will, hopefully, be able to take an image of a simple handwritten mathematical expression, such as:
9x5+1/3 or 7+4÷2 and evaluate it.

I will be using a few libraries, possibly more than the ones I list here:

- Numpy
- Keras
- Matplotlib
- OpenCV

The UI will allow the user to choose an image file from disk to upload it. Afterwards, the image will be displayed, and the user can choose to preview the image as black and white and to alter the threshold to make the expression more recognizable in the image. The user can then press a button to evalaute the expression, and the expression plus the result will be displayed in text form. In the top left is an information and instructions menu.

By the first demo, the program will be able to take individual (positive) integers or the basic arithmetic symbols (+,-,÷,x) and semi-accurately interpret them as the correct integer or symbol. By semi-accurately, I mean anything that's not random.

My final submission should have an improved prediction accuracy, and will be able to evaluate an entire expression, given that it only uses the basic arithmetic operations I listed.
