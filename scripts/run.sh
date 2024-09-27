#!/bin/bash

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    echo "Script must be sourced, not executed."
    exit 1
fi

if [ "$#" -ne 5 ]; then
    echo "Usage: ./run64.sh <num_gpu> <box_size> <domain_size> <config> <iter>"
    return 1
fi

num_gpu=$1
box_size=$2
domain_size=$3
config=$4
iter=$5

output_file="output_${num_gpu}_${box_size}_${domain_size}_${config}.txt"


if [ "$config" -eq "1" ]; then
    echo "Running each box serial no fusion"
    export IRISX_HOME="/ccs/home/sanilrao/IPDPS24/SC_IRISX"
    ixh=$(echo $IRISX_HOME)
    echo "IRISX_HOME is set to: $ixh"
    export IRIS="/ccs/home/sanilrao/IPDPS24/.iris"
    i=$(echo $IRIS)
    echo "IRIS is set to: $i"
    # export CPATH=$CPATH:$IRIS/include
    # export LIBRARY_PATH=$LIBRARY_PATH:$IRIS/lib:$IRIS/lib64
    # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$IRIS/lib:$IRIS/lib64
    export IRIS_MAX_HIP_DEVICE=$num_gpu
    imhd=$(echo $IRIS_MAX_HIP_DEVICE)
    echo "IRIS_MAX_HIP_DEVICE is set to: $imhd"
    export IRIS_ASYNC=0
    ia=$(echo $IRIS_ASYNC)
    echo "IRIS_ASYNC is set to: $ia"
    make="cd ../SC_IRISX/; cp ~/IPDPS24/DAG_IRISX/${box_size}x3D/kernel* .; make proto_ipdps; cd ../scripts/"
    eval $make
    command="../SC_IRISX/./proto_ipdps_sc -boxSize $box_size -domainSize $domain_size"
    echo "Executing command: $command"
    for ((i=0; i<iter; i++)); do
        eval $command > $output_file 
    done

elif [ "$config" -eq "2" ]; then
    echo "Running DAG Fusion"
    export IRISX_HOME="/ccs/home/sanilrao/IPDPS24/DAG_IRISX"
    ixh=$(echo $IRISX_HOME)
    echo "IRISX_HOME is set to: $ixh"
    export IRIS="/ccs/home/sanilrao/IPDPS24/.iris"
    # export CPATH=$CPATH:$IRIS/include
    # export LIBRARY_PATH=$LIBRARY_PATH:$IRIS/lib:$IRIS/lib64
    # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$IRIS/lib:$IRIS/lib64
    i=$(echo $IRIS)
    echo "IRIS is set to: $i"
    export IRIS_MAX_HIP_DEVICE=$num_gpu
    imhd=$(echo $IRIS_MAX_HIP_DEVICE)
    echo "IRIS_MAX_HIP_DEVICE is set to: $imhd"
    export IRIS_ASYNC=1
    ia=$(echo $IRIS_ASYNC)
    echo "IRIS_ASYNC is set to: $ia"
    make="cd ../DAG_IRISX/; cp ~/IPDPS24/DAG_IRISX/${box_size}x3D/kernel* .; make proto_ipdps; cd ../scripts/"
    eval $make
    command="../DAG_IRISX/./proto_ipdps_dag -boxSize $box_size -domainSize $domain_size"
    echo "Executing command: $command"
    for ((i=0; i<iter; i++)); do
        eval $command > $output_file
    done

elif [ "$config" -eq "3" ]; then
    echo "Running DAG Fusion + Task Fusion"
    export IRISX_HOME="/ccs/home/sanilrao/IPDPS24/TASK_IRISX"
    ixh=$(echo $IRISX_HOME)
    echo "IRISX_HOME is set to: $ixh"
    export IRIS="/ccs/home/sanilrao/IPDPS24/.schedule_iris"
    # export CPATH=$CPATH:$IRIS/include
    # export LIBRARY_PATH=$LIBRARY_PATH:$IRIS/lib:$IRIS/lib64
    # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$IRIS/lib:$IRIS/lib64
    i=$(echo $IRIS)
    echo "IRIS is set to: $i"
    export IRIS_MAX_HIP_DEVICE=$num_gpu
    imhd=$(echo $IRIS_MAX_HIP_DEVICE)
    echo "IRIS_MAX_HIP_DEVICE is set to: $imhd"
    export IRIS_ASYNC=0
    ia=$(echo $IRIS_ASYNC)
    echo "IRIS_ASYNC is set to: $ia"
    make="cd ../TASK_IRISX/; cp ~/IPDPS24/DAG_IRISX/${box_size}x3D/kernel* .; make proto_ipdps; cd ../scripts/"
    echo $make
    eval $make
    command="../TASK_IRISX/./proto_ipdps_task -boxSize $box_size -domainSize $domain_size"
    echo "Executing command: $command"
    for ((i=0; i<iter; i++)); do
        eval $command > $output_file
    done
elif [ "$config" -eq "4" ]; then
    echo "Running each box serial no fusion"
    export IRISX_HOME="/ccs/home/sanilrao/IPDPS24/fused_SC_IRISX"
    ixh=$(echo $IRISX_HOME)
    echo "IRISX_HOME is set to: $ixh"
    export IRIS="/ccs/home/sanilrao/IPDPS24/.iris"
    i=$(echo $IRIS)
    echo "IRIS is set to: $i"
    # export CPATH=$CPATH:$IRIS/include
    # export LIBRARY_PATH=$LIBRARY_PATH:$IRIS/lib:$IRIS/lib64
    # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$IRIS/lib:$IRIS/lib64
    export IRIS_MAX_HIP_DEVICE=$num_gpu
    imhd=$(echo $IRIS_MAX_HIP_DEVICE)
    echo "IRIS_MAX_HIP_DEVICE is set to: $imhd"
    export IRIS_ASYNC=0
    ia=$(echo $IRIS_ASYNC)
    echo "IRIS_ASYNC is set to: $ia"
    make="cd ../fused_SC_IRISX/; cp ~/IPDPS24/DAG_IRISX/fused_kernels/fused${box_size}x3D/kernel* .; make proto_ipdps; cd ../scripts/"
    echo $make
    eval $make
    command="../fused_SC_IRISX/./proto_ipdps_sc -boxSize $box_size -domainSize $domain_size"
    echo "Executing command: $command"
    for ((i=0; i<iter; i++)); do
        eval $command > $output_file 
    done

elif [ "$config" -eq "5" ]; then
    echo "Running DAG Fusion"
    export IRISX_HOME="/ccs/home/sanilrao/IPDPS24/fused_DAG_IRISX"
    ixh=$(echo $IRISX_HOME)
    echo "IRISX_HOME is set to: $ixh"
    export IRIS="/ccs/home/sanilrao/IPDPS24/.iris"
    # export CPATH=$CPATH:$IRIS/include
    # export LIBRARY_PATH=$LIBRARY_PATH:$IRIS/lib:$IRIS/lib64
    # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$IRIS/lib:$IRIS/lib64
    i=$(echo $IRIS)
    echo "IRIS is set to: $i"
    export IRIS_MAX_HIP_DEVICE=$num_gpu
    imhd=$(echo $IRIS_MAX_HIP_DEVICE)
    echo "IRIS_MAX_HIP_DEVICE is set to: $imhd"
    export IRIS_ASYNC=1
    ia=$(echo $IRIS_ASYNC)
    echo "IRIS_ASYNC is set to: $ia"
    make="cd ../fused_DAG_IRISX/; cp ~/IPDPS24/DAG_IRISX/fused_kernels/fused${box_size}x3D/kernel* .; make proto_ipdps; cd ../scripts/"
    echo $make
    eval $make
    command="../fused_DAG_IRISX/./proto_ipdps_dag_fused -boxSize $box_size -domainSize $domain_size"
    echo "Executing command: $command"
    for ((i=0; i<iter; i++)); do
        eval $command > $output_file
    done

elif [ "$config" -eq "6" ]; then
    echo "Running DAG Fusion + Task Fusion"
    export IRISX_HOME="/ccs/home/sanilrao/IPDPS24/fused_TASK_IRISX"
    ixh=$(echo $IRISX_HOME)
    echo "IRISX_HOME is set to: $ixh"
    export IRIS="/ccs/home/sanilrao/IPDPS24/.schedule_iris"
    # export CPATH=$CPATH:$IRIS/include
    # export LIBRARY_PATH=$LIBRARY_PATH:$IRIS/lib:$IRIS/lib64
    # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$IRIS/lib:$IRIS/lib64
    i=$(echo $IRIS)
    echo "IRIS is set to: $i"
    export IRIS_MAX_HIP_DEVICE=$num_gpu
    imhd=$(echo $IRIS_MAX_HIP_DEVICE)
    echo "IRIS_MAX_HIP_DEVICE is set to: $imhd"
    export IRIS_ASYNC=0
    ia=$(echo $IRIS_ASYNC)
    echo "IRIS_ASYNC is set to: $ia"
    make="cd ../fused_TASK_IRISX/; cp ~/IPDPS24/DAG_IRISX/fused_kernels/fused${box_size}x3D/kernel* .; make proto_ipdps; cd ../scripts/"
    echo $make
    eval $make
    command="../fused_TASK_IRISX/./proto_ipdps_task -boxSize $box_size -domainSize $domain_size"
    echo "Executing command: $command"
    for ((i=0; i<iter; i++)); do
        eval $command > $output_file
    done
fi

