IRIS_PATH=${IRIS}
LDFLAGS=-L$(IRIS)/lib64 -L$(IRIS)/lib -liris -lpthread -ldl
CFLAGS=-g

proto:
	g++ -I$(IRIS_PATH)/include -I. -Wunused-result -O3 $(CFLAGS) -o spiral_proto spiral_proto_leveleuler.cpp $(LDFLAGS)

spiral:
	g++ -I$(IRIS_PATH)/include -I. -Wunused-result -O3 $(CFLAGS) -o spiral_iris_mddft spiral_iris_mddft.cpp $(LDFLAGS)

cpu:
	g++ -I$(IRIS_PATH)/include -O3 hello_world.cpp $(LDFLAGS)

kernel.openmp.so:
	g++  -I$(IRIS)/include/ -O3 -fopenmp -fPIC -shared -I. -o $@ hello_world.cpp
gpu:
	nvcc -I . -L IRIS_PATH/iris hello_world.cu -ldl -liris -lpthread

kernel.ptx: kernel.cu
	nvcc -ptx $^

clean:
	-rm *.out *.cu *.ptx *.hip.cpp *.hip *_openmp.c *_host2cuda.c *_host2hip.c *.so
