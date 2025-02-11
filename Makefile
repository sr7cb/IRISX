IRIS_PATH=${IRIS}
PROTO_PATH=/ccs/home/sanilrao/IPDPS24/proto
CONSTANTS= -DDIM=2 -DHDF5=off -DPR_MPI=on
LDFLAGS=-L$(IRIS)/lib64 -L$(IRIS)/lib -Wl,-rpath,$(IRIS)/lib64 -liris -lpthread -ldl
MPI_COMPILE_FLAGS = $(shell mpicc --showme:compile)
MPI_LINK_FLAGS = $(shell mpicc --showme:link)
CFLAGS=-pg

all: proto_ipdps

kswitch:
	g++ -DPRINTDEBUG -ggdb -I$(IRIS_PATH)/include -I. -Wunused-result $(CFLAGS) -o kswitch testkswitch.cpp $(LDFLAGS)

test_ntt:
		g++ -DPRINTDEBUG -ggdb -I$(IRIS_PATH)/include -I. -Wunused-result $(CFLAGS) -o nttest test_multi_kernel_ntt.cpp $(LDFLAGS)

test_different_kernels:
		g++ -DPRINTDEBUG -ggdb -I$(IRIS_PATH)/include -I. -Wunused-result $(CFLAGS) -o tdkernels test_multi_kernel.cpp $(LDFLAGS)

proto_ipdps:
	g++ --std=c++17 -I$(IRIS_PATH)/include -I. -I$(PROTO_PATH)/include -I$(PROTO_PATH) -O3 -DDIM=3 -DHDF5=off -DIRIS -o proto_ipdps_dag spiral_proto_leveleuler_mpi.cpp $(LDFLAGS)


proto_no_mpi_cuda:
	nvcc  -I$(IRIS_PATH)/include -I. -I$(PROTO_PATH)/include -I$(PROTO_PATH)  -O3 -DDIM=2 -DPROTO_ACCEL -DPROTO_CUDA -DHDF5=off -DTIME -o spiral_proto_no_mpi_cuda spiral_proto_leveleuler_mpi.cpp $(LDFLAGS) 


proto_no_mpi_hip:
	hipcc --std=c++17 -ggdb -I$(IRIS_PATH)/include -I. -I$(PROTO_PATH)/include -I$(PROTO_PATH) -I/opt/rocm-5.7.1/include/roctracer -L/opt/rocm-5.7.1/lib -O3 -DDIM=2 -DPROTO_ACCEL -DPROTO_HIP -DHDF5=off -DTIME -o proto_no_mpi_hip spiral_proto_leveleuler_mpi.cpp $(LDFLAGS) 

irisx_proto_no_mpi:
	g++ --std=c++17 -ggdb -I$(IRIS_PATH)/include -I. -I$(PROTO_PATH)/include -I$(PROTO_PATH) -O3 -DDIM=3 -DAMR=on -DMMB=off -DHDF5=off -DDEBUG=off -DTIME -DIRIS -o irisx_proto_no_mpi spiral_proto_leveleuler_mpi.cpp $(LDFLAGS) 

proto_mpi:
	mpicxx --std=c++17 $(MPI_COMPILE_FLAGS) -I$(IRIS_PATH)/include -I. -I$(PROTO_PATH)/include -I$(PROTO_PATH) -O3 $(CONSTANTS)  -o irisx_proto_mpi spiral_proto_leveleuler_mpi.cpp $(LDFLAGS) $(MPI_LINK_FLAGS)

irisx_proto:
	g++ -I$(IRIS_PATH)/include -I. -Wunused-result -ggdb -O3 $(CFLAGS) -DPRINTDEBUG -o irisx_proto spiral_proto_leveleuler.cpp $(LDFLAGS)

mddft:
	g++ -DPRINTDEBUG -ggdb -I$(IRIS_PATH)/include -I. -Wunused-result $(CFLAGS) -o irisx_mddft spiral_iris_mddft.cpp $(LDFLAGS)

mdprdft:
	g++ -I$(IRIS_PATH)/include -I. -Wunused-result -O3 $(CFLAGS) -o irisx_mdprdft spiral_iris_mdprdft.cpp $(LDFLAGS)

cpu:
	g++ -I$(IRIS_PATH)/include -O3 hello_world.cpp $(LDFLAGS)

kernel.openmp.so:
	g++  -I$(IRIS)/include/ -O3 -fopenmp -fPIC -shared -I. -o $@ hello_world.cpp
gpu:
	nvcc -I . -L IRIS_PATH/iris hello_world.cu -ldl -liris -lpthread

kernel.ptx: kernel.cu
	nvcc -ptx $^

clean:
	rm *.out *.so irisx_proto_no_mpi
	#-rm *.out *.cu *.ptx *.hip.cpp *.hip *_openmp.c *_host2cuda.c *_host2hip.c *.so
