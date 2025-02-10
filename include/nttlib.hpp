#include <string>
#include <iostream>
#include <vector>
#pragma once

static constexpr auto preamble = R"(
    Load(fftx);
    ImportAll(fftx);
    Load(nttx);
    ImportAll(nttx);
    ImportAll(simt);
    ImportAll(fftx.platforms.cuda);
    Load(jit);
    ImportAll(jit);
    
    gpuConfig := "general"; # [h100, v100, rtx4090, general]
    
    mxp := true;
    # mxp := false;
    
    # useMont := true; # use Montgomery
    useMont := false; # use Barrett
    
    # pass arguments to func/macro as array or not
    # useArray := true; 
    useArray := false;
    
    # set this to true for BLAS op. code gen.
    genBLAS := false;
    
    # insize: [input data bit-length, rounded up]
    # mbits: [modulus bit length]
    # basesize: [system data bit-length] # init basesize always equals insize/2
    
    insize := 128; 
    mbits := insize -4;
    recursionLevel := 2;
    
    basesize := insize/2;
    
    if mxp then conf := LocalConfig.nttx.mxpCUDAConf();
    else conf := LocalConfig.nttx.mpCUDAConf(); fi;
    
    # name := "ntt"::StringInt(n)::"mpcuda";
    name := "nttmpcuda"; # easier for testing
    
    # FUTURE: conf_type -> conf.type()
    if mxp then
        conf_type := T_UInt(basesize*2);
    else
        conf_type := T_UInt(1);
    fi;
    
    p := var("modulus", conf_type);
    twiddles := var("twiddles", TPtr(conf_type));
    mu := var("mu", conf.type());
    )";
    
    static constexpr auto script = R"(twrec := CopyFields(rec(n := n, modulus := p, twiddles := twiddles, mu := mu, 
                            mxp := mxp, useMacro := useMacro, insize := insize, useArray := useArray,
                            useMont := useMont, gpuConfig := gpuConfig,
                            recursionLevel := recursionLevel,
                            basesize := basesize, mbits := mbits));
    funcrec := CopyFields(rec(abstractType := conf_type, fname := name, params := [p, twiddles, mu]), twrec);
    t := let(batch := nbatch,
        apat := When(true, APar, AVec),
        TFCall(TTensorI(NTT(n, twrec), batch, apat, apat), funcrec)
    );
    )";
    
    
    static constexpr auto postamble = R"(
    if 1 = 1 then
    opts := conf.getOpts(t);
    opts.max_shmem_size := (96*1024/(insize/32))/(2^(recursionLevel-1)); # scale down shmem_size if int bit-width > 32b
    
    tt := opts.tagIt(t);
    
    tt := opts.preProcess(tt);
    rt := opts.search(tt);
    s := opts.sumsRuleTree(rt);
    
    # knowledge base for performance tuning
    if multiKernel = false and insize = 128 then
        if gpuConfig = "h100" and Log2Int(n) >= 14 then
            multiKernel := true;
        elif gpuConfig = "v100" and Log2Int(n) >= 11 then
            multiKernel := true;
        elif gpuConfig = "rtx4090" and Log2Int(n) >= 12 then
            multiKernel := true;
        fi;
    elif multiKernel = true and insize = 256 then
        if gpuConfig = "h100" and Log2Int(n) = 8 then
            multiKernel := false;
        elif gpuConfig = "rtx4090" and Log2Int(n) = 8 then
            multiKernel := false;
        fi;
    fi;
    
    # using multiple kernel launch
    if multiKernel then
        s := FixUpCUDASigmaSPL(s, opts); # defined in fftx
        SubstBottomUp(s, [Scat, @(1, fCompose)], e->Scat(fCompose(@(1).val._children[2])));
        SubstBottomUp(s, [Gath, @(1, fCompose)], e->Gath(fCompose(@(1).val._children[2])));
        s.ruletree := rt;
    fi;
    c := opts.codeSums(s);
    fi;
    PrintIRISMETAJIT(c,opts);)";

class NTTProblem: public FFTXProblem {
    public:
        using FFTXProblem::FFTXProblem;
        void randomProblemInstance() {
        }
        void semantics(std::string arch) {
            if(arch != "cuda") {
                std::cout << "NTTProblem only supports CUDA" << std::endl;
                exit(1);
            }
            std::cout << preamble << std::endl;
            std::cout << "nbatch := " << sizes.at(0) << ";" << std::endl;
            std::cout << "n := " << sizes.at(1) << ";" << std::endl;
            if(sizes.at(2) != 0) {
                std::cout << "useMacro := false;\nmultiKernel := false;" << std::endl;
            } else{
                std::cout << "useMacro := true;\nmultiKernel := true;" << std::endl;
            }
            std::cout << script << std::endl;
            std::cout << postamble << std::endl;
        }
    };