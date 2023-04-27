IRIS_PATH=${IRIS}
LDFLAGS=-L$(IRIS)/lib64 -L$(IRIS)/lib -liris -lpthread -ldl
cpu:
	g++ -I. -fPIC -shared hello_world.cpp -fopenmp $(LDFLAGS)

gpu:
	nvcc -I . -L IRIS_PATH/iris hello_world.cu -ldl -liris -lpthread

clean:
	-rm *.out