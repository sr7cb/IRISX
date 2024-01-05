IRIS_PATH=${IRIS}
PROTO_PATH=/home/sanilr/proto
CONSTANTS= -DDIM=2 -DHDF5=off -DPR_MPI=on
LDFLAGS=-L$(IRIS)/lib64 -L$(IRIS)/lib -liris -lpthread -ldl
MPI_COMPILE_FLAGS = $(shell mpicc --showme:compile)
MPI_LINK_FLAGS = $(shell mpicc --showme:link)
CFLAGS=-pg

proto_no_mpi:
	g++ --std=c++17 -ggdb -I$(IRIS_PATH)/include -I. -I$(PROTO_PATH)/include -I$(PROTO_PATH) -O3 -DDIM=2 -DHDF5=off -DPRINTDEBUG  -o spiral_proto_no_mpi spiral_proto_leveleuler_mpi.cpp $(LDFLAGS) 


proto_mpi:
	mpicxx --std=c++17 $(MPI_COMPILE_FLAGS) -I$(IRIS_PATH)/include -I. -I$(PROTO_PATH)/include -I$(PROTO_PATH) -O3 $(CONSTANTS)  -o spiral_proto_mpi spiral_proto_leveleuler_mpi.cpp $(LDFLAGS) $(MPI_LINK_FLAGS)

proto:
	g++ -I$(IRIS_PATH)/include -I. -Wunused-result -ggdb -O3 $(CFLAGS) -o spiral_proto spiral_proto_leveleuler.cpp $(LDFLAGS)

mddft:
	g++ -I$(IRIS_PATH)/include -I. -Wunused-result $(CFLAGS) -o spiral_iris_mddft spiral_iris_mddft.cpp $(LDFLAGS)

mdprdft:
	g++ -I$(IRIS_PATH)/include -I. -Wunused-result -O3 $(CFLAGS) -o spiral_iris_mdprdft spiral_iris_mdprdft.cpp $(LDFLAGS)

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
