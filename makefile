cuda: ray_object_cuda.cu 
	nvcc -arch sm_70 -Xcompiler -Wall ray_object_cuda.cu -o cuda -Xcompiler -lm

serial: ray_object_serial.c
	gcc -O2 -Wall -fopenmp ray_object_serial.c -o serial -lm

clean: cuda serial
	rm -rf cuda serial
