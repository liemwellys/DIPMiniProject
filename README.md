# DIP Mini Project
NCTU Digital Image Processing Course Mini Project for comparing Generated Blur Image from Observed Function and Modeled Function using Python 3.

## Package
`openCV`<br>
`numpy`<br>
`matplotlib`<br>

## Development environment
OS: Windows 10 <br>
IDE: [PyCharm](https://www.jetbrains.com/pycharm/) <br>
Python Environment Manager: [Anaconda](https://www.anaconda.com/) <br>

## Program flow
1. Set the path of Origin Images directory.<br>
2. Set the path of DIP book cover (blurred and original) directory.<br>
3. List each file from the “Origin Images” and “DIP book cover” directory.<br>
4. Load DIP book cover image (blurred and original) in grayscale.<br>
5. Load each image in “Origin Image” directory in gray scale, then process with Observed Function and Modelled Function.<br>
6. Plot & show on screen for each origin images, images blurred by observed function (problem A), and image blurred by modelled function (problem B) with its histogram.<br>
7. Save each generated image from problem A and problem B in “Image Result” directory.<br>
