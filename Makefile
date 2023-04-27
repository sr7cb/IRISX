IRIS_PATH=${IRIS}

cpu:
	g++ -I . -L IRIS_PATH/iris  hello_world.cpp -fopenmp -ldl -liris -lpthread

gpu:
	nvcc -I . -L IRIS_PATH/iris hello_world.cu -ldl -liris -lpthread

clean:
	-rm *.out