HPC for ML

Daniel Rivera Ruiz
drr342@nyu.edu

Contents:
	1. Lab1_drr342.pdf - PDF file with the answers to all the exercises
	2. lab1.c - C source code for exercises C1, C4 and C5
	3. lab1.py - Python source code for exercises C2 and C3
	4. lab1.pytorch - Pytorch source code for exercises C6, C7, C8 and C9
	5. lab1_drr342_cpu_4760455.out - Text file with the output generated for exercises C1-C5
	6. lab1_drr342_cpu_4769242.out - Text file with the output generated for exercises C6-C8
	7. lab1_drr342_gpu_4769728.out - Text file with the output generated for exercise C9
	8. launch_cpu_reserv.s - Bash script to run lab1.c, lab1.py and lab1.pytorch on the reserved CPU
	9. launch_gpu_reserv.s - Bash script to run lab1.pytorch on the reserved GPU

Notes:
	1. lab1.pytorch accepts one argument in the command line to specify the number of workers for the DataLoader. If no argument is passed, it defaults to 1.
