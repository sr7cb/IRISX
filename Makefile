IRIS_PATH=${IRIS}
LDFLAGS=-L$(IRIS)/lib64 -L$(IRIS)/lib -liris -lpthread -ldl

spiral:
	g++ -I$(IRIS_PATH)/include -Wunused-result -O3 -o spiral_iris_mddft spiral_iris_mddft.cpp $(LDFLAGS)

cpu:
	g++ -I$(IRIS_PATH)/include -O3 hello_world.cpp $(LDFLAGS)

kernel.openmp.so:
	g++  -I$(IRIS)/include/ -O3 -fopenmp -fPIC -shared -I. -o $@ hello_world.cpp

gpu:
	nvcc -I . -L IRIS_PATH/iris hello_world.cu -ldl -liris -lpthread

kernel.ptx: kernel.cu
	nvcc -ptx $^

clean:
	-rm *.out