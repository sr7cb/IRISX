void conf(std::string arch) {
    std::cout << "Load(fftx);\nImportAll(fftx);" << std::endl;
    std::cout << "ImportAll(simt);\nLoad(jit);\nImport(jit);"<< std::endl;
    if(arch == "cuda" || arch == "cudaopenmp")
        std::cout << "conf := LocalConfig.fftx.confGPU();" << std::endl;
    else if(arch == "hip" || arch == "hipopenmp")
        std::cout << "conf := FFTXGlobals.defaultHIPConf();" << std::endl;
    else if(arch == "opencl") 
        std::cout << "conf := FFTXGlobals.defaultOpenCLConf();" << std::endl;
    else if(arch == "openmp")
        std::cout << "conf := FFTXGlobals.defaultConf();" << std::endl;
    else
        std::cout << "conf := FFTXGlobals.defaultConf();" << std::endl;
}

static constexpr auto conv_script{
R"(
szhalfcube := DropLast(szcube,1)::[Int(Last(szcube)/2)+1]; 
var_1:= var("var_1", BoxND(szhalfcube, TReal));
var_2:= var("var_2", BoxND(szhalfcube, TReal));
var_3:= var("var_3", BoxND(szcube, TReal));
var_4:= var("var_4", BoxND(szcube, TReal));
var_5:= var("var_5", BoxND(szhalfcube, TReal));
var_3:= X;
var_4:= Y;
symvar := var("sym", TPtr(TReal));
transform:= TFCall(TDecl(TDAG([
    TDAGNode(MDPRDFT(szcube,-1), var_1,var_3),
    TDAGNode(Diag(diagTensor(FDataOfs(symvar,Product(szhalfcube),0),fConst(TReal, 2, 1))), var_2,var_1),
    TDAGNode(IMDPRDFT(szcube,1), var_4,var_2),

]),
   [var_1,var_2]
),
rec(fname:="rconv_spiral", params:= [symvar])
);
prefix:="rconv";
)"};

void printBackend(std::string arch) {
    std::cout << "if 1 = 1 then opts:=conf.getOpts(transform);\ntt:= opts.tagIt(transform);\nif(IsBound(fftx_includes)) then opts.includes:=fftx_includes;fi;\nc:=opts.fftxGen(tt);\nfi;" << std::endl;
    std::cout << "GASMAN(\"collect\");" << std::endl;
    if(arch == "cuda") {
        std::cout << "PrintIRISMETAJIT(c,opts);" << std::endl;
        // std::cout << "opts.prettyPrint(c);" << std::endl;
    } else if(arch == "hip") {
        std::cout << "PrintIRISMETAJIT(c,opts);\n" << std::endl;
    } else if(arch == "opencl") {
        std::cout << "PrintIRISMETAJIT(c,opts);\n" << std::endl;
    } else if(arch == "openmp") {
        std::cout << "opts.prettyPrint(c);" << std::endl;
    } else {
        std::cout << "opts.prettyPrint(c);" << std::endl;
    }
}

class ConvProblem: public FFTXProblem {
public:
    using FFTXProblem::FFTXProblem;
    void randomProblemInstance() {
    }
    void semantics(std::string arch) {
        conf(arch);
        std::cout << "szcube := [" << sizes.at(args.size()) << ", " << sizes.at(args.size()+1) << ", " << sizes.at(args.size()+2) << "];" << std::endl;
        std::cout << "name := \""<< name << "_spiral" << "\";" << std::endl;
        std::cout << conv_script << std::endl;
        printBackend(arch);
    }
};