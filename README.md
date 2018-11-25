# -NEW-15-112-Final-project
This project will, hopefully, be able to take an image of a simple handwritten mathematical expression, such as:
9x5+1/3 or 7+4÷2 and evaluate it.

I will be using a few libraries, possibly more than the ones I list here:

- Numpy
- Keras
- Matplotlib
- OpenCV

The interface will contain a prompt to upload the image of the expression. After the upload, it will display the image
and a prompt to evaluate it. Then, it will attempt to evaluate the expression by segmenting first, then evaluate it.

By the first demo, the program will be able to take individual (positive) integers or the basic arithmetic symbols (+,-,/,÷,x) (**÷ and / are both division**) and semi-accurately interpret them as the correct integer or symbol. By semi-accurately, I mean anything that's not random.

My final submission should have an improved prediction accuracy, and will be able to evaluate an entire expression, given that it only uses the basic arithmetic operations I listed.
