// using namespace fftx;


static constexpr auto proto_script1{
R"(
Import(filtering);
Load(fftx);
ImportAll(fftx);
ImportAll(simt);

Class(ScatO, RuleSet);
RewriteRules(ScatO, rec(
    remove_scat := Rule([@(1,Compose),@(2,Scat),@(3,O)], e-> @(3).val)
));

CudaCodegen.to_block_shmem_arrays := meth(self, ker, ta_list, ker_args, opts)
        local ta, idx_list, blk_ids, th_ids, idx, idx_cnt, idx_rng, derefs, nths, size, padded, v;

        for ta in ta_list do
            padded := false;
            ta.("decl_specs") := ["__shared__"];
            idx_list := When(IsBound(ta.parctx), ta.parctx, []);
            th_ids  := Filtered(idx_list, idx -> IsSimtThreadIdx(idx[2])); Sort(th_ids, (i1, i2) -> i1[2].name < i2[2].name);
            for idx in th_ids do
                idx_cnt := idx[2].count();
                if idx_cnt > 1 then
                    size := Cond(IsValue(ta.t.size), _unwrap(ta.t.size), ta.t.size);
                    
                    if not padded and Mod(size, 2) = 0 then
                        size := size+1;
                    fi;
                    padded := true;
                    idx_rng := idx[2].get_rng();
                    derefs := Collect(ker, @(1, deref, v -> ta in v.free()));
                    nths   := Collect(ker, @(1,   nth, v -> ta in v.free()));

                    for v in derefs do
                        v.loc := add(v.loc, size*(idx[2]-idx_rng[1]) );
                    od;
                    for v in nths do
                        v.idx := add(v.idx, size*(idx[2]-idx_rng[1]));
                    od;
                    ta.t.size := size*idx_cnt;
                fi;
            od;
        od;

        ker := decl(ta_list, ker);

        return [ker, ta_list];
    end;
Load(protox);
ImportAll(protox);
Load(jit);
ImportAll(jit);

input_dims := [40,40];
deconvolve_dims := [38,38];
dims_Xdir := [38,35];
dims_Ydir := [35,38];
flux_Xdims := [36,35];
flux_Ydims := [35,36];
divergence_Xdims := [36,34];
divergence_Ydims := [34,36];
final_dims := [32,32];
x1 := var.fresh_t("x1",TArray(TReal,Product(input_dims)));
rho := var.fresh_t("rho", TPtr(TReal));
gamma := var.fresh_t("gamma", TReal);
retval := var.fresh_t("retval", TPtr(TReal));
wavespeed_out := var("wavespeed", TPtr(TReal));
# dir := var.fresh_t("dir", TInt);
a_scale := var.fresh_t("a_scale", TReal);
dx := var.fresh_t("dx", TReal);
i := Ind(Product(input_dims));
i2 := Ind(Product(deconvolve_dims));
i3 := Ind(Product(deconvolve_dims));
i4 := Ind(Product(deconvolve_dims));
i5 := Ind(Product(final_dims));
i6 := Ind(Product(dims_Xdir));
i7 := Ind(Product(dims_Xdir));
i8 := Ind(Product(flux_Xdims));
i9 := Ind(Product(flux_Xdims));
i10 := Ind(Product(divergence_Xdims));
i11 := Ind(Product(final_dims));
j := Ind(1);
j2 := Ind(4);
b := Ind();
b2 := Ind();
square := Lambda(j, Lambda([x1], x1 * x1));
_sqrt := Lambda(j, Lambda([x1], sqrt(x1)));
one_over_sqrt := Lambda(j, Lambda([x1], 1/sqrt(x1)));
one_over_x := Lambda(j, Lambda([x1], 1/x1));
_abs := Lambda([ j ], Lambda([ x1 ], abs(x1)));
deconvolve_filt := Blk([[V(1),V(1), V(-4), V(1), V(1)]]);
)"};


static constexpr auto proto_script2{
R"(
t := let(szcube := [4,4,4],
    name := "name",
    TFCall(MDDFT(szcube, 1), 
            rec(fname := name, params := []))
);

opts := conf.getOpts(t);
opts.codegen.Pointwise := DefaultCodegen.Pointwise;
opts.codegen.O := (self, o, y, x, opts) >> skip();


sf1_c2p1 := VStack(Gath(fAdd(4*Product(input_dims), 1, add(mul(b,256),i))), 
        Diamond(div) * VStack(Gath(fAdd(4*Product(input_dims),1, add(add(mul(b,256),i), Product(input_dims)))),Gath(fAdd(4*Product(input_dims), 1, add(mul(b,256),i)))), 
        Diamond(div) * VStack(Gath(fAdd(4*Product(input_dims),1, add(add(mul(b,256),i),2*Product(input_dims)))),Gath(fAdd(4*Product(input_dims), 1, add(mul(b,256),i)))));
sf2_c2p1 := VStack(Diamond(mul) * VStack(Pointwise(square) *  Diamond(div) * VStack(Gath(fAdd(4*Product(input_dims),1,add(add(mul(b,256),i), Product(input_dims)))),Gath(fAdd(4*Product(input_dims), 1, add(mul(b,256),i)))), Gath(fAdd(4*Product(input_dims), 1, add(mul(b,256),i)))), 
        Diamond(mul) * VStack(Pointwise(square) * Diamond(div) * VStack(Gath(fAdd(4*Product(input_dims),1,add(add(mul(b,256),i),2*Product(input_dims)))),Gath(fAdd(4*Product(input_dims), 1, add(mul(b,256),i)))), Gath(fAdd(4*Product(input_dims), 1, add(mul(b,256),i)))),  
        Gath(fAdd(4*Product(input_dims),1,add(add(mul(b,256),i),3*Product(input_dims)))));
rv_c2p1 := RowVec(neg(mul(V(0.5),sub(gamma, V(1)))), neg(mul(V(0.5),sub(gamma, V(1)))), sub(gamma, V(1)));
consttoPrim1 := SIMTISum(ASIMTKernelFlag(ASIMTGridDimX(QuoInt(i.range,256)+1)), b, QuoInt(i.range,256)+1, SIMTISum(ASIMTBlockDimX(256), i, 256, COND(lt(add(mul(b,256),i), Product(input_dims)),BB(Scat(fStack(fAdd(4*Product(input_dims),1,add(mul(b,256),i)), fAdd(4*Product(input_dims),1,add(add(mul(b,256),i), Product(input_dims))), fAdd(4*Product(input_dims),1,add(add(mul(b,256),i), 2*Product(input_dims))), fAdd(4*Product(input_dims),1,add(add(mul(b,256),i), 3 * Product(input_dims))))) * VStack(sf1_c2p1, rv_c2p1 * sf2_c2p1)), O(4 * Product(input_dims), 4 * Product(input_dims)))));

deconvolve_shift := add(mul(input_dims[2], idiv(add(mul(b,256),i2), deconvolve_dims[2])), imod(add(mul(b,256),i2), deconvolve_dims[2]));
gf_decon := Gath(fStack(fAdd(4 * Product(input_dims), 1, add(add(deconvolve_shift, V(1)), mul(j2, Product(input_dims)))),
    fAdd(4 * Product(input_dims), 3, add(add(deconvolve_shift, V(40)), mul(j2,Product(input_dims)))),
    fAdd(4 * Product(input_dims), 1, add(add(deconvolve_shift, V(81)), mul(j2,Product(input_dims))))));
G_decon := Gath(fAdd(4* Product(input_dims), 1, add(add(mul(input_dims[2], add(idiv(add(mul(b,256),i2), deconvolve_dims[2]), V(1))), add(imod(add(mul(b,256),i2), deconvolve_dims[2]), V(1))), mul(j2,Product(input_dims)))));
rv_decon := RowVec(-V(1/24), 1);
_decon := rv_decon * VStack(deconvolve_filt * gf_decon, G_decon);
sf1_c2p2 := VStack(Gath(fAdd(4, 1, 0)), 
        Diamond(div) * VStack(Gath(fAdd(4,1, 1)),Gath(fAdd(4, 1, 0))), 
        Diamond(div) * VStack(Gath(fAdd(4,1, 2)),Gath(fAdd(4, 1, 0))));
sf2_c2p2 := VStack(Diamond(mul) * VStack(Pointwise(square) *  Diamond(div) * VStack(Gath(fAdd(4,1,1)),Gath(fAdd(4, 1, 0))), Gath(fAdd(4, 1, 0))), 
        Diamond(mul) * VStack(Pointwise(square) * Diamond(div) * VStack(Gath(fAdd(4,1,2)),Gath(fAdd(4, 1, 0))), Gath(fAdd(4, 1, 0))),  
        Gath(fAdd(4,1,3)));
rv_c2p2 := RowVec(neg(mul(V(0.5),sub(gamma, V(1)))), neg(mul(V(0.5),sub(gamma, V(1)))), sub(gamma, V(1)));
decon_consttoPrim2 := SIMTISum(ASIMTKernelFlag(ASIMTGridDimX(QuoInt(i2.range,256)+1)), b, QuoInt(i2.range,256)+1, SIMTISum(ASIMTBlockDimX(256), i2, 256, COND(lt(add(mul(b,256),i2), Product(input_dims)),BB(Scat(fStack(fAdd(4*Product(deconvolve_dims),1,add(mul(b,256),i2)), fAdd(4*Product(deconvolve_dims),1,add(add(mul(b,256),i2), Product(deconvolve_dims))), fAdd(4*Product(deconvolve_dims),1,add(add(mul(b,256),i2), 2*Product(deconvolve_dims))), fAdd(4*Product(deconvolve_dims),1,add(add(mul(b,256),i2), 3 * Product(deconvolve_dims))))) * VStack(sf1_c2p2, rv_c2p2 * sf2_c2p2) * IterVStack(j2, j2.range, BB(_decon)).unroll()), O(4 * Product(deconvolve_dims), 4 * Product(input_dims)))));

convolve_shift := add(mul(input_dims[2], idiv(add(mul(b,256),i4), deconvolve_dims[2])), imod(add(mul(b,256),i4), deconvolve_dims[2]));
gf_conv := Gath(fStack(fAdd(4 * Product(input_dims) + 4 * Product(deconvolve_dims), 1, add(add(convolve_shift, V(1)), mul(j2, Product(input_dims)))),
    fAdd(4 * Product(input_dims) + 4 * Product(deconvolve_dims), 3, add(add(convolve_shift, V(40)), mul(j2,Product(input_dims)))),
    fAdd(4 * Product(input_dims) + 4 * Product(deconvolve_dims), 1, add(add(convolve_shift, V(81)), mul(j2,Product(input_dims))))));
G_conv := Gath(fAdd(4 * Product(input_dims) + 4 * Product(deconvolve_dims), 1, add(add(mul(b,256),i4),add(4 * Product(input_dims), mul(j2, Product(deconvolve_dims))))));
rv_conv := RowVec(V(1/24), 1);
_conv := Scat(fAdd(4 * Product(deconvolve_dims), 1, add(add(mul(b,256),i4),mul(j2,Product(deconvolve_dims))))) * rv_conv * VStack(deconvolve_filt * gf_conv, G_conv);
convolve := SIMTISum(ASIMTKernelFlag(ASIMTGridDimX(QuoInt(i4.range,256)+1)), b, QuoInt(i4.range,256)+1, SIMTISum(ASIMTBlockDimX(256), i4, 256, COND(lt(add(mul(b,256),i4), Product(deconvolve_dims)), ISum(j2, j2.range, BB(_conv)).unroll(), O(4 * Product(deconvolve_dims), 4 * Product(input_dims)))));

# wavespeed_shift := add(mul(V(38), add(idiv(add(mul(b,256),i5), V(32)), V(3))), add(imod(add(mul(b,256),i5), V(32)), V(3)));
# G_wv := Gath(fAdd(4 * Product(deconvolve_dims), 1, add(wavespeed_shift, mul(0,Product(deconvolve_dims)))));
# G2_wv := Gath(fAdd(4 * Product(deconvolve_dims), 1, add(wavespeed_shift, mul(1,Product(deconvolve_dims)))));
# G3_wv := Gath(fAdd(4 * Product(deconvolve_dims), 1, add(wavespeed_shift, mul(2,Product(deconvolve_dims)))));
# G4_wv := Gath(fAdd(4 * Product(deconvolve_dims), 1, add(wavespeed_shift, mul(3,Product(deconvolve_dims)))));
# ds_wv := VStack(Diamond(mul) * VStack(Pointwise(one_over_sqrt) * G_wv, Pointwise(_sqrt)*G4_wv), G2_wv, G3_wv);
# rv_wv := RowVec(mul(2, sqrt(gamma)), 1, 1);
# red_wv := Reduce(j, max, retval);
# pw_wv := Pointwise(_abs);
# ws := VStack(Scat(fAdd(Product(final_dims),1, add(mul(b,256),i5))), red_wv * pw_wv) * rv_wv * ds_wv;
# wavespeed := ISIMTISum(ASIMTKernelFlag(ASIMTGridDimX(QuoInt(i5.range,256)+1)), b, QuoInt(i5.range,256)+1, SIMTISum(ASIMTBlockDimX(256), i5, 256, COND(lt(add(mul(b,256),i5), Product(final_dims)),BB(ws), O(1,1))));

x_filt_taps := [-1/12,7/12,7/12,-1/12];
c2f_shift_x := add(mul(dims_Xdir[1], idiv(add(mul(b,256),i6), dims_Xdir[2])), imod(add(mul(b,256),i6), dims_Xdir[2]));
gath_c2f_x := Gath(fAdd(4*Product(deconvolve_dims), 4, add(c2f_shift_x, mul(j2, Product(deconvolve_dims)))));
_cell_to_face_L_xdir := IterVStack(j2, j2.range, Blk([x_filt_taps]) * gath_c2f_x).unroll();
cell_to_face_L_xdir := SIMTISum(ASIMTKernelFlag(ASIMTGridDimX(QuoInt(i6.range,256)+1)), b, QuoInt(i6.range,256)+1, SIMTISum(ASIMTBlockDimX(256), i6, 256, COND(lt(add(mul(b,256),i6), Product(dims_Xdir)),Scat(fStack(fAdd(4 * Product(dims_Xdir), 1, i3), fAdd(4 * Product(dims_Xdir), 1, add(i3, Product(dims_Ydir))), fAdd(4 * Product(dims_Xdir), 1, add(i3, 2 *Product(dims_Xdir))), fAdd(4 * Product(dims_Xdir), 1, add(i3, 3 * Product(dims_Xdir))))) * IterVStack(j2, j2.range, Blk([x_filt_taps]) * gath_c2f_x).unroll(), O(4*Product(deconvolve_dims),4*Product(deconvolve_dims)))));


y_filt_taps := [-1/12,7/12,7/12,-1/12];
gath_c2f_y := Gath(fStack(fAdd(4*Product(deconvolve_dims), 1, add(add(mul(b,256),i6), mul(j2, Product(deconvolve_dims)))), fAdd(4*Product(deconvolve_dims), 1, add(add(add(mul(b,256),i6), deconvolve_dims[2]), mul(j2, Product(deconvolve_dims)))), fAdd(4*Product(deconvolve_dims), 1, add(add(add(mul(b,256),i6), 2 * deconvolve_dims[2]), mul(j2, Product(deconvolve_dims)))), fAdd(4*Product(deconvolve_dims), 1, add(add(add(mul(b,256),i6), 3 * deconvolve_dims[2]), mul(j2, Product(deconvolve_dims))))));
_cell_to_face_L_ydir := IterVStack(j2, j2.range, Blk([y_filt_taps]) * gath_c2f_y).unroll();
cell_to_face_L_ydir := SIMTISum(ASIMTKernelFlag(ASIMTGridDimX(QuoInt(i6.range,256)+1)), b, QuoInt(i6.range,256)+1, SIMTISum(ASIMTBlockDimX(256), i6, 256, COND(lt(add(mul(b,256),i6), Product(dims_Ydir)),Scat(fStack(fAdd(4 * Product(dims_Ydir), 1, i3), fAdd(4 * Product(dims_Ydir), 1, add(i3, Product(dims_Ydir))), fAdd(4 * Product(dims_Ydir), 1, add(i3, 2 *Product(dims_Ydir))), fAdd(4 * Product(dims_Ydir), 1, add(i3,  3 * Product(dims_Ydir))))) * IterVStack(j2, j2.range, Blk([y_filt_taps]) * gath_c2f_y).unroll(), O(4*Product(deconvolve_dims),4*Product(deconvolve_dims)))));

sqrt_gamma_x := Lambda(j, Lambda([x1], sqrt(gamma) * x1));
rhobar := RowVec(1/2,1/2) * Gath(fStack(fAdd(4*2, 1, 0),fAdd(4*2, 1, 4)));
ubar := RowVec(1/2,1/2) * Gath(fStack(fAdd(4*2, 1, 1),fAdd(4*2, 1, 5)));
umagic := RowVec(1/2,-1/2) * Gath(fStack(fAdd(4*2, 1, 1),fAdd(4*2, 1, 5)));
pbar := RowVec(1/2,1/2) * Gath(fStack(fAdd(4*2, 1, 3),fAdd(4*2, 1, 7)));
pmagic  := RowVec(1/2,-1/2) * Gath(fStack(fAdd(4*2, 1, 3),fAdd(4*2, 1, 7)));
sqrt_inverse_rhobar := Pointwise(_sqrt) * Pointwise(one_over_x) * rhobar;
sqrt_gamma_sqrt_pbar := Pointwise(sqrt_gamma_x) * Pointwise(_sqrt) * pbar;
cbar := Diamond(mul) * VStack(sqrt_gamma_sqrt_pbar, sqrt_inverse_rhobar);
pstar := Diamond(add) * VStack(pbar, Diamond(mul) * VStack(cbar, Diamond(mul) *  VStack(rhobar, umagic)));
ustar := RowVec(1,1) * VStack(ubar,HProduct(1,1,1) * VStack(pmagic, Pointwise(one_over_x) * rhobar, Pointwise(one_over_x) * cbar));
vs1 := VStack(ustar,
            cbar,
            pstar,
            ubar,
            Gath(fAdd(8, 1, 0)),
            Gath(fAdd(8, 1, 1)),
            Gath(fAdd(8, 1, 2)),
            Gath(fAdd(8, 1, 3)),
            Gath(fAdd(8, 1, 4)),
            Gath(fAdd(8, 1, 5)),
            Gath(fAdd(8, 1, 6)),
            Gath(fAdd(8, 1, 7)));
inverse_cbar2 := Pointwise(one_over_x) * Diamond(mul) * VStack(Gath(fAdd(12, 1, 1)), Gath(fAdd(12, 1, 1)));
pstar_minus_last_patch_lo := RowVec(1,-1) * VStack(Gath(fAdd(12, 1, 2)), Gath(fAdd(12, 1, 7)));
pstar_minus_last_patch_hi := RowVec(1,-1) * VStack(Gath(fAdd(12, 1, 2)), Gath(fAdd(12, 1, 11)));
s := Scat(fStack(fAdd(4*Product(dims_Xdir), 1, add(mul(b,256),i6)), fAdd(4*Product(dims_Xdir), 1, add(add(mul(b,256),i6),Product(dims_Xdir))), fAdd(4*Product(dims_Xdir), 1, add(add(mul(b,256),i6),Product(dims_Xdir)*2)), fAdd(4*Product(dims_Xdir), 1, add(add(mul(b,256),i6),Product(dims_Xdir)*3))));
res := COND(let(x1 := var.fresh_t("x1", TPtr(TReal)), Lambda(i6, Lambda(x1, gt(nth(x1,V(0)), V(0))))), #first condition
       COND(let(x1 := var.fresh_t("x1", TPtr(TReal)), Lambda(i6, Lambda(x1, gt(add(nth(x1,V(1)), mul(-1, nth(x1,V(3)))), V(0))))), #2nd condition
       VStack(RowVec(1,1) * VStack(Diamond(mul) * VStack(pstar_minus_last_patch_lo, inverse_cbar2), Gath(fAdd(12,1,4))), #output 1
       Gath(fAdd(12,1,0)),
       Gath(fAdd(12,1,6)),
       Gath(fAdd(12,1,2))), 
       VStack(Gath(fAdd(12,1,4)), #output 2 
       Gath(fAdd(12,1,5)),
       Gath(fAdd(12,1,6)),
       Gath(fAdd(12,1,7)))), 
       COND(let(x1 := var.fresh_t("x1", TPtr(TReal)), Lambda(i6, Lambda(x1, gt(add(nth(x1,V(1)), mul(1, nth(x1,V(3)))), V(0))))), #3rd condition
       VStack(RowVec(1,1) * VStack(Diamond(mul) * VStack(pstar_minus_last_patch_hi, inverse_cbar2), Gath(fAdd(12,1,8))),
       Gath(fAdd(12,1,0)),
       Gath(fAdd(12,1,10)),
       Gath(fAdd(12,1,2))), #output 1
       VStack(Gath(fAdd(12,1,8)),
       Gath(fAdd(12,1,9)),
       Gath(fAdd(12,1,10)),
       Gath(fAdd(12,1,11)))));  
_upwind_x := BB(res * vs1 * VStack(_cell_to_face_L_xdir, _cell_to_face_L_xdir));
upwind_x := SIMTISum(ASIMTKernelFlag(ASIMTGridDimX(QuoInt(i6.range,256)+1)), b, QuoInt(i6.range,256)+1, SIMTISum(ASIMTBlockDimX(256), i6, 256, COND(lt(add(mul(b,256),i6), Product(dims_Xdir)), BB(s *_upwind_x), O(4 * Product(dims_Xdir),4*Product(deconvolve_dims)))));

rhobar_y := RowVec(1/2,1/2) * Gath(fStack(fAdd(4*2, 1, 0),fAdd(4*2, 1, 4)));
ubar_y := RowVec(1/2,1/2) * Gath(fStack(fAdd(4*2, 1, 2),fAdd(4*2, 1, 6)));
umagic_y := RowVec(1/2,-1/2) * Gath(fStack(fAdd(4*2, 1, 2),fAdd(4*2, 1, 6)));
pbar_y := RowVec(1/2,1/2) * Gath(fStack(fAdd(4*2, 1, 3),fAdd(4*2, 1, 7)));
pmagic_y  := RowVec(1/2,-1/2) * Gath(fStack(fAdd(4*2, 1,3),fAdd(4*2, 1, 7)));
sqrt_inverse_rhobar_y := Pointwise(_sqrt) * Pointwise(one_over_x) * rhobar_y;
sqrt_gamma_sqrt_pbar_y := Pointwise(sqrt_gamma_x) * Pointwise(_sqrt) * pbar_y;
cbar_y := Diamond(mul) * VStack(sqrt_gamma_sqrt_pbar_y, sqrt_inverse_rhobar_y);
pstar_y := Diamond(add) * VStack(pbar_y, Diamond(mul) * VStack(cbar_y, Diamond(mul) *  VStack(rhobar_y, umagic_y)));
ustar_y := RowVec(1,1) * VStack(ubar_y,HProduct(1,1,1) * VStack(pmagic_y, Pointwise(one_over_x) * rhobar_y, Pointwise(one_over_x) * cbar_y));
vs1_y := VStack(ustar_y,
            cbar_y,
            pstar_y,
            ubar_y,
            Gath(fAdd(8, 1, 0)),
            Gath(fAdd(8, 1, 1)),
            Gath(fAdd(8, 1, 2)),
            Gath(fAdd(8, 1, 3)),
            Gath(fAdd(8, 1, 4)),
            Gath(fAdd(8, 1, 5)),
            Gath(fAdd(8, 1, 6)),
            Gath(fAdd(8, 1, 7)));
inverse_cbar2_y := Pointwise(one_over_x) * Diamond(mul) * VStack(Gath(fAdd(12, 1, 1)), Gath(fAdd(12, 1, 1)));
pstar_minus_last_patch_lo_y := RowVec(1,-1) * VStack(Gath(fAdd(12, 1, 2)), Gath(fAdd(12, 1, 7)));
pstar_minus_last_patch_hi_y := RowVec(1,-1) * VStack(Gath(fAdd(12, 1, 2)), Gath(fAdd(12, 1, 11)));
s_y := Scat(fStack(fAdd(4*Product(dims_Ydir), 1, add(mul(b,256),i6)), fAdd(4*Product(dims_Ydir), 1, add(add(mul(b,256),i6),Product(dims_Ydir))), fAdd(4*Product(dims_Ydir), 1, add(add(mul(b,256),i6),Product(dims_Ydir)*2)), fAdd(4*Product(dims_Ydir), 1, add(add(mul(b,256),i6),Product(dims_Ydir)*3))));
res_y := COND(let(x1 := var.fresh_t("x1", TPtr(TReal)), Lambda(i6, Lambda(x1, gt(nth(x1,V(0)), V(0))))), #first condition
       COND(let(x1 := var.fresh_t("x1", TPtr(TReal)), Lambda(i6, Lambda(x1, gt(add(nth(x1,V(1)), mul(-1, nth(x1,V(3)))), V(0))))), #2nd condition
       VStack(RowVec(1,1) * VStack(Diamond(mul) * VStack(pstar_minus_last_patch_lo_y, inverse_cbar2_y), Gath(fAdd(12,1,4))), #output 1
       Gath(fAdd(12,1,5)),
       Gath(fAdd(12,1,0)),
       Gath(fAdd(12,1,2))), 
       VStack(Gath(fAdd(12,1,4)), #output 2 
       Gath(fAdd(12,1,5)),
       Gath(fAdd(12,1,6)),
       Gath(fAdd(12,1,7)))), 
       COND(let(x1 := var.fresh_t("x1", TPtr(TReal)), Lambda(i6, Lambda(x1, gt(add(nth(x1,V(1)), mul(1, nth(x1,V(3)))), V(0))))), #3rd condition
       VStack(RowVec(1,1) * VStack(Diamond(mul) * VStack(pstar_minus_last_patch_hi_y, inverse_cbar2_y), Gath(fAdd(12,1,8))),
       Gath(fAdd(12,1,9)),
       Gath(fAdd(12,1,0)),
       Gath(fAdd(12,1,2))), #output 1
       VStack(Gath(fAdd(12,1,8)),
       Gath(fAdd(12,1,9)),
       Gath(fAdd(12,1,10)),
       Gath(fAdd(12,1,11)))));
_upwind_y := BB(res_y * vs1_y * VStack(_cell_to_face_L_ydir, _cell_to_face_L_ydir));       
upwind_y := SIMTISum(ASIMTKernelFlag(ASIMTGridDimX(QuoInt(i6.range,256)+1)), b, QuoInt(i6.range,256)+1, SIMTISum(ASIMTBlockDimX(256), i6, 256, COND(lt(add(mul(b,256),i6), Product(dims_Xdir)), BB(s_y * _upwind_y), O(4 * Product(dims_Xdir),4*Product(deconvolve_dims)))));


flux_lbd1 := Lambda(j, Lambda([x1], (1/2)* x1 *x1));
flux_lbd2 := Lambda(j, Lambda([x1], gamma/(gamma - 1)*x1));
a0_x := Diamond(mul) * VStack(Gath(fAdd(4*Product(dims_Xdir), 1, add(mul(b,256),i7))), Gath(fAdd(4*Product(dims_Xdir), 1, add(add(mul(b,256),i7),Product(dims_Xdir)))));
a1_x := RowVec(1,1)* VStack(Diamond(mul) * VStack(a0_x,Gath(fAdd(4*Product(dims_Xdir),1,add(add(mul(b,256),i7),Product(dims_Xdir))))), Gath(fAdd(4*Product(dims_Xdir),1, add(add(mul(b,256),i7),3*Product(dims_Xdir)))));
a2_x :=Diamond(mul) * VStack(a0_x,Gath(fAdd(4*Product(dims_Xdir), 1, add(add(mul(b,256),i7),2*Product(dims_Xdir)))));
temp_x := Pointwise(flux_lbd2)*Diamond(mul)* VStack(Gath(fAdd(4*Product(dims_Xdir), 1, add(add(mul(b,256),i7),Product(dims_Xdir)))), Gath(fAdd(4*Product(dims_Xdir),1,add(add(mul(b,256),i7),3*Product(dims_Xdir)))));
a3_x := RowVec(1,1) * VStack(temp_x, Diamond(mul) * VStack(a0_x, RowVec(1,1)*VStack(Pointwise(flux_lbd1)*Gath(fAdd(4*Product(dims_Xdir), 1, add(add(mul(b,256),i7),Product(dims_Xdir)))),Pointwise(flux_lbd1)*Gath(fAdd(4*Product(dims_Xdir), 1,add(add(mul(b,256),i7),2*Product(dims_Xdir))))))); 
flux_s_x := Scat(fStack(fAdd(4*Product(dims_Xdir), 1, add(mul(b,256),i7)), fAdd(4*Product(dims_Xdir), 1, add(add(mul(b,256),i7),Product(dims_Xdir))), fAdd(4*Product(dims_Xdir), 1, add(add(mul(b,256),i7),2*Product(dims_Xdir))), fAdd(4*Product(dims_Xdir), 1, add(add(mul(b,256),i7),3*Product(dims_Xdir)))));
flux_vs1_x := VStack(a0_x, a1_x, a2_x, a3_x);
getFlux_x := SIMTISum(ASIMTKernelFlag(ASIMTGridDimX(QuoInt(i7.range,256)+1)), b, QuoInt(i7.range,256)+1, SIMTISum(ASIMTBlockDimX(256), i7, 256, COND(lt(add(mul(b,256),i7), Product(dims_Xdir)),BB(flux_s_x * flux_vs1_x), O(4*Product(dims_Xdir),4*Product(dims_Xdir)))));

a0 := Diamond(mul) * VStack(Gath(fAdd(4*Product(dims_Ydir), 1, add(mul(b,256),i7))), Gath(fAdd(4*Product(dims_Ydir), 1, add(add(mul(b,256),i7),2 *Product(dims_Ydir)))));
a1 :=Diamond(mul) * VStack(a0,Gath(fAdd(4*Product(dims_Ydir), 1, add(add(mul(b,256),i7),Product(dims_Ydir)))));
a2 := RowVec(1,1)* VStack(Diamond(mul) * VStack(a0,Gath(fAdd(4*Product(dims_Ydir),1,add(add(mul(b,256),i7),2 *Product(dims_Ydir))))), Gath(fAdd(4*Product(dims_Ydir),1, add(add(mul(b,256),i7),3*Product(dims_Ydir)))));
temp := Pointwise(flux_lbd2)*Diamond(mul)* VStack(Gath(fAdd(4*Product(dims_Ydir), 1, add(add(mul(b,256),i7),2 *Product(dims_Ydir)))), Gath(fAdd(4*Product(dims_Ydir),1, add(add(mul(b,256),i7),3*Product(dims_Ydir)))));
a3 := RowVec(1,1) * VStack(temp, Diamond(mul) * VStack(a0, RowVec(1,1)*VStack(Pointwise(flux_lbd1)*Gath(fAdd(4*Product(dims_Ydir), 1, add(add(mul(b,256),i7),Product(dims_Ydir)))),Pointwise(flux_lbd1)*Gath(fAdd(4*Product(dims_Ydir), 1, add(add(mul(b,256),i7),2*Product(dims_Ydir))))))); 
flux_s_y := Scat(fStack(fAdd(4*Product(dims_Ydir), 1, add(mul(b,256),i7)), fAdd(4*Product(dims_Ydir), 1, add(add(mul(b,256),i7),Product(dims_Ydir))), fAdd(4*Product(dims_Ydir), 1, add(add(mul(b,256),i7),2*Product(dims_Ydir))), fAdd(4*Product(dims_Ydir), 1, add(add(mul(b,256),i7),3*Product(dims_Ydir)))));
flux_vs1_y := VStack(a0, a1, a2, a3);
getFlux_y := SIMTISum(ASIMTKernelFlag(ASIMTGridDimX(QuoInt(i7.range,256)+1)), b, QuoInt(i7.range,256)+1, SIMTISum(ASIMTBlockDimX(256), i7, 256, COND(lt(add(mul(b,256),i7), Product(dims_Xdir)), BB(flux_s_y * flux_vs1_y), O(4*Product(dims_Ydir),4*Product(dims_Ydir)))));


deconface_taps := [1,-2,1];
gath_deconface_x := Gath(fStack(fAdd(4*Product(dims_Xdir), 1, add(add(mul(b,256),i8), mul(j2, Product(dims_Xdir)))), fAdd(4*Product(dims_Xdir), 1, add(add(add(mul(b,256),i8), dims_Xdir[2]), mul(j2, Product(dims_Xdir)))), fAdd(4*Product(dims_Xdir), 1, add(add(add(mul(b,256),i8), 2 * dims_Xdir[2]), mul(j2, Product(dims_Xdir))))));
rv_deconface_x := RowVec(-V(1/24), 1);
middle_gath_x := Gath(fAdd(4*Product(dims_Xdir), 1, add(add(add(mul(b,256),i8),dims_Xdir[2]), mul(j2,Product(dims_Xdir)))));
_deconface_X := BB(IterVStack(j2, j2.range, rv_deconface_x * VStack(Blk([deconface_taps]) * gath_deconface_x, middle_gath_x)).unroll());

deconface_shift := add(mul(dims_Ydir[2], idiv(add(mul(b,256),i8), flux_Ydims[2])), imod(add(mul(b,256),i8), flux_Ydims[2]));
gath_deconface_y := Gath(fAdd(4*Product(dims_Ydir), 3, add(deconface_shift, mul(j2, Product(dims_Ydir)))));
rv_deconface_y := rv_deconface_x;
middle_gath_y := Gath(fAdd(4 * Product(dims_Ydir), 1, add(add(1,deconface_shift), mul(j2,Product(dims_Ydir)))));
_deconface_Y := BB(IterVStack(j2, j2.range, rv_deconface_y * VStack(Blk([deconface_taps]) * gath_deconface_y, middle_gath_y)).unroll());

a0_x_decon := Diamond(mul) * VStack(Gath(fAdd(4, 1, 0)), Gath(fAdd(4, 1, 1)));
a1_x_decon := RowVec(1,1)* VStack(Diamond(mul) * VStack(a0_x_decon,Gath(fAdd(4,1,1))), Gath(fAdd(4,1, 3)));
a2_x_decon :=Diamond(mul) * VStack(a0_x_decon,Gath(fAdd(4, 1, 2)));
temp_x_decon := Pointwise(flux_lbd2)*Diamond(mul)* VStack(Gath(fAdd(4, 1, 1)), Gath(fAdd(4,1,3)));
a3_x_decon := RowVec(1,1) * VStack(temp_x_decon, Diamond(mul) * VStack(a0_x_decon, RowVec(1,1)*VStack(Pointwise(flux_lbd1)*Gath(fAdd(4, 1, 1)),Pointwise(flux_lbd1)*Gath(fAdd(4, 1,2))))); 
flux_s_x_decon := Scat(fStack(fAdd(4*Product(flux_Xdims), 1, add(mul(b,256),i8)), fAdd(4*Product(flux_Xdims), 1, add(add(mul(b,256),i8),Product(flux_Xdims))), fAdd(4*Product(flux_Xdims), 1, add(add(mul(b,256),i8),2*Product(flux_Xdims))), fAdd(4*Product(flux_Xdims), 1, add(add(mul(b,256),i8),3*Product(flux_Xdims)))));
flux_vs1_x_decon := VStack(a0_x_decon, a1_x_decon, a2_x_decon, a3_x_decon);
getFlux_x_decon := SIMTISum(ASIMTKernelFlag(ASIMTGridDimX(QuoInt(i8.range,256)+1)), b, QuoInt(i8.range,256)+1, SIMTISum(ASIMTBlockDimX(256), i8, 256, COND(lt(add(mul(b,256),i8), i8.range),BB(flux_s_x_decon * flux_vs1_x_decon * _deconface_X), O(4*Product(flux_Xdims),4*Product(dims_Xdir)))));

a0_decon := Diamond(mul) * VStack(Gath(fAdd(4, 1, 0)), Gath(fAdd(4, 1, 2)));
a1_decon :=Diamond(mul) * VStack(a0_decon,Gath(fAdd(4, 1, 1)));
a2_decon := RowVec(1,1)* VStack(Diamond(mul) * VStack(a0_decon,Gath(fAdd(4,1,2))), Gath(fAdd(4,1, 3)));
temp_decon := Pointwise(flux_lbd2)*Diamond(mul)* VStack(Gath(fAdd(4, 1, 2)), Gath(fAdd(4,1, 3)));
a3_decon := RowVec(1,1) * VStack(temp_decon, Diamond(mul) * VStack(a0_decon, RowVec(1,1)*VStack(Pointwise(flux_lbd1)*Gath(fAdd(4, 1, 1)),Pointwise(flux_lbd1)*Gath(fAdd(4, 1,2))))); 
flux_s_y_decon := Scat(fStack(fAdd(4*Product(flux_Ydims), 1, add(mul(b,256),i8)), fAdd(4*Product(flux_Ydims), 1, add(add(mul(b,256),i8),Product(flux_Ydims))), fAdd(4*Product(flux_Ydims), 1, add(add(mul(b,256),i8),2*Product(flux_Ydims))), fAdd(4*Product(flux_Ydims), 1, add(add(mul(b,256),i8),3*Product(flux_Ydims)))));
flux_vs1_y_decon := VStack(a0_decon, a1_decon, a2_decon, a3_decon);
getFlux_y_decon := SIMTISum(ASIMTKernelFlag(ASIMTGridDimX(QuoInt(i8.range,256)+1)), b, QuoInt(i8.range,256)+1, SIMTISum(ASIMTBlockDimX(256), i8, 256, COND(lt(add(mul(b,256),i8), i8.range),BB(flux_s_y_decon * flux_vs1_y_decon * _deconface_Y),  O(4*Product(flux_Xdims),4*Product(dims_Xdir)))));


filt_fluxs := Blk([deconface_taps]);
gf_fluxs := Gath(fStack(fAdd(4 * Product(dims_Xdir) + 4 * Product(flux_Xdims), 1, add(add(mul(b,256),i9), mul(j2, Product(dims_Xdir)))),
    fAdd(4 * Product(dims_Xdir) + 4 * Product(flux_Xdims), 1, add(add(add(mul(b,256),i9),dims_Xdir[2]), mul(j2,Product(dims_Xdir)))),
    fAdd(4 * Product(dims_Xdir) + 4 * Product(flux_Xdims), 1, add(add(add(mul(b,256),i9),dims_Xdir[2]*2), mul(j2,Product(dims_Xdir))))));
G_fluxs := Gath(fAdd(4 * Product(dims_Xdir) + 4 * Product(flux_Xdims), 1, add(add(mul(b,256),i9), add(4 * Product(dims_Xdir), mul(j2,Product(flux_Xdims))))));
rv_fluxs := RowVec(V(1/24), 1);
decon_fluxs := Scat(fAdd(4 * Product(flux_Xdims), 1, add(add(mul(b,256),i9),mul(j2,Product(flux_Xdims))))) * rv_fluxs * VStack(filt_fluxs * gf_fluxs, G_fluxs);
aflux_stencil_x := SIMTISum(ASIMTKernelFlag(ASIMTGridDimX(QuoInt(i9.range,256)+1)), b, QuoInt(i9.range,256)+1, SIMTISum(ASIMTBlockDimX(256), i9, 256, COND(lt(add(mul(b,256),i9), i9.range),BB(ISum(j2, j2.range, BB(decon_fluxs)).unroll()), O(4 * Product(flux_Xdims),4 * Product(dims_Xdir) + 4 * Product(flux_Xdims)))));


gf_fluxs_y := Gath(fAdd(4 * Product(dims_Ydir) + 4 * Product(flux_Ydims), 3, add(add(dims_Ydir[2] * idiv(add(mul(b,256),i9), flux_Ydims[2]), imod(add(mul(b,256),i9), flux_Ydims[2])), mul(j2, Product(dims_Ydir)))));
G_fluxs_y := Gath(fAdd(4 * Product(dims_Ydir) + 4 * Product(flux_Ydims), 1, add(add(mul(b,256),i9), add(4 * Product(dims_Ydir), mul(j2,Product(flux_Ydims))))));
rv_fluxs_y := RowVec(V(1/24), 1);
decon_fluxs_y := Scat(fAdd(4 * Product(flux_Ydims), 1, add(add(mul(b,256),i9),mul(j2,Product(flux_Ydims))))) * rv_fluxs_y * VStack(filt_fluxs * gf_fluxs_y, G_fluxs_y);
aflux_stencil_y := SIMTISum(ASIMTKernelFlag(ASIMTGridDimX(QuoInt(i9.range,256)+1)), b, QuoInt(i9.range,256)+1, SIMTISum(ASIMTBlockDimX(256), i9, 256, COND(lt(add(mul(b,256),i9), i9.range), BB(ISum(j2, j2.range, BB(decon_fluxs_y)).unroll()), O(4 * Product(flux_Ydims),4 * Product(dims_Ydir) + 4 * Product(flux_Ydims)))));


divergence_taps := [-1,1];
divergence_shift := add(mul(flux_Xdims[2],idiv(add(mul(b,256),i10),divergence_Xdims[2])), imod(add(mul(b,256),i10),divergence_Xdims[2]));
gath_divergence_x := Gath(fAdd(4*Product(flux_Xdims), 2, add(divergence_shift, mul(j2, Product(flux_Xdims)))));
_divergence_xdir := IterVStack(j2, j2.range, Blk([divergence_taps]) * gath_divergence_x).unroll();
divergence_xdir := SIMTISum(ASIMTKernelFlag(ASIMTGridDimX(QuoInt(i10.range,256)+1)), b, QuoInt(i10.range,256)+1, SIMTISum(ASIMTBlockDimX(256), i10, 256, COND(lt(add(mul(b,256),i10), i10.range),BB(Scat(fStack(fAdd(4 * Product(divergence_Xdims), 1, add(mul(b,256),i10)), fAdd(4 * Product(divergence_Xdims), 1, add(add(mul(b,256),i10), Product(divergence_Xdims))), fAdd(4 * Product(divergence_Xdims), 1, add(add(mul(b,256),i10), 2 *Product(divergence_Xdims))), fAdd(4 * Product(divergence_Xdims), 1, add(add(mul(b,256),i10), 3 * Product(divergence_Xdims))))) * _divergence_xdir), O(4 * Product(divergence_Xdims),4*Product(flux_Xdims)))));


gath_divergence_y := Gath(fStack(fAdd(4*Product(flux_Ydims), 1, add(add(mul(b,256),i10), mul(j2, Product(flux_Ydims)))), fAdd(4*Product(flux_Ydims), 1, add(add(add(mul(b,256),i10), flux_Ydims[2]), mul(j2, Product(flux_Ydims))))));
_divergence_ydir := IterVStack(j2, j2.range, Blk([divergence_taps]) * gath_divergence_y).unroll();
divergence_ydir := SIMTISum(ASIMTKernelFlag(ASIMTGridDimX(QuoInt(i10.range,256)+1)), b, QuoInt(i10.range,256)+1, SIMTISum(ASIMTBlockDimX(256), i10, 256, COND(lt(add(mul(b,256),i10), i10.range), BB(Scat(fStack(fAdd(4 * Product(divergence_Ydims), 1, add(mul(b,256),i10)), fAdd(4 * Product(divergence_Ydims), 1, add(add(mul(b,256),i10), Product(divergence_Ydims))), fAdd(4 * Product(divergence_Ydims), 1, add(add(mul(b,256),i10), 2 *Product(divergence_Ydims))), fAdd(4 * Product(divergence_Ydims), 1, add(add(mul(b,256),i10), 3 * Product(divergence_Ydims))))) * _divergence_ydir), O(4 * Product(divergence_Ydims),4*Product(flux_Ydims)))));


r_x := (divergence_Xdims[1] - final_dims[1])/2;
c_x := (divergence_Xdims[2] - final_dims[2])/2;
gshift_x := r_x*divergence_Xdims[2]+c_x;
gath_crop_x := Gath(fAdd(4 * Product(divergence_Xdims),1, add(add(add(gshift_x, divergence_Xdims[2]*idiv(add(mul(b,256),i11),final_dims[1])), imod(add(mul(b,256),i11), final_dims[1])), mul(j2, Product(divergence_Xdims)))));
crop_x := SIMTISum(ASIMTKernelFlag(ASIMTGridDimX(QuoInt(i11.range,256)+1)), b, QuoInt(i11.range,256)+1, SIMTISum(ASIMTBlockDimX(256), i11, 256, COND(lt(add(mul(b,256),i11), i11.range), Scat(fStack(fAdd(4 * Product(final_dims),1, add(mul(b,256),i11)), fAdd(4 * Product(final_dims),1, add(add(mul(b,256),i11), Product(final_dims))), fAdd(4 * Product(final_dims),1, add(add(mul(b,256),i11), 2 * Product(final_dims))), fAdd(4 * Product(final_dims),1, add(add(mul(b,256),i11), 3 * Product(final_dims))))) * IterVStack(j2, j2.range, gath_crop_x).unroll(), O(4 * Product(final_dims),4 * Product(divergence_Xdims)))));

r_y := (divergence_Ydims[1] - final_dims[1])/2;
c_y := (divergence_Ydims[2] - final_dims[2])/2;
gshift_y := r_y*divergence_Ydims[2]+c_y;
gath_crop_y := Gath(fAdd(4 * Product(divergence_Ydims),1, add(add(add(gshift_y, divergence_Ydims[2]*idiv(add(mul(b,256),i11),final_dims[1])), imod(add(mul(b,256),i11), final_dims[1])), mul(j2, Product(divergence_Ydims)))));
crop_y := SIMTISum(ASIMTKernelFlag(ASIMTGridDimX(QuoInt(i11.range,256)+1)), b, QuoInt(i11.range,256)+1, SIMTISum(ASIMTBlockDimX(256), i11, 256, COND(lt(add(mul(b,256),i11), i11.range), Scat(fStack(fAdd(4 * Product(final_dims),1, add(mul(b,256),i11)), fAdd(4 * Product(final_dims),1, add(add(mul(b,256),i11), Product(final_dims))), fAdd(4 * Product(final_dims),1, add(add(mul(b,256),i11), 2 * Product(final_dims))), fAdd(4 * Product(final_dims),1, add(add(mul(b,256),i11), 3 * Product(final_dims))))) * IterVStack(j2, j2.range, gath_crop_y).unroll(), O(4 * Product(final_dims),4 * Product(divergence_Ydims)))));

plus_equal := Gath(fStack(fAdd(8*Product(final_dims), 1, add(add(mul(b,256),i11), mul(j2, Product(final_dims)))), fAdd(8*Product(final_dims), 1, add(add(mul(b,256),i11), mul(j2 + 4, Product(final_dims))))));
times_equal := Lambda(j, Lambda(x1, ((-1*a_scale)/dx) * x1));
final_scat := Scat(fStack(fAdd(4*Product(final_dims), 1, add(mul(b,256),i11)),fAdd(4*Product(final_dims), 1, add(add(mul(b,256),i11),Product(final_dims))),fAdd(4*Product(final_dims), 1, add(add(mul(b,256),i11),2 * Product(final_dims))),fAdd(4*Product(final_dims), 1, add(add(mul(b,256),i11),3 *Product(final_dims)))));
final_arith := SIMTISum(ASIMTKernelFlag(ASIMTGridDimX(QuoInt(i11.range,256)+1)), b, QuoInt(i11.range,256)+1, SIMTISum(ASIMTBlockDimX(256), i11, 256, COND(lt(add(mul(b,256),i11), i11.range), final_scat * IterVStack(j2, j2.range, Pointwise(times_equal) * RowVec(1,1) * plus_equal).unroll(), O(4*Product(final_dims),8*Product(final_dims)))));

total :=  final_arith * VStack(crop_x * divergence_xdir * aflux_stencil_x * VStack(getFlux_x, getFlux_x_decon) * upwind_x, crop_y * divergence_ydir * aflux_stencil_y *  VStack(getFlux_y,  getFlux_y_decon) * upwind_y) * convolve * VStack(consttoPrim1, decon_consttoPrim2);

t := TFCall(total,rec(fname := name, params := [gamma, a_scale, dx]));
opts.unparser.skip := (self, o, i, is) >> Print("");
tt := opts.tagIt(t);
srt := opts.sumsRuleTree(opts.search(tt));
srt2 := Rewrite(srt, ScatO, opts);
c := opts.codeSums(srt2);
find_vars := function(c,list) 
local i, j, v, list2;
list2 := [];
for i in list do 
   for j in i.vars do
      v := Collect(c, j);
      if Length(v) <= 2 then
      Append(list2, [v[1]]);
      fi;
   od;
od;
return list2;
end;
replace_vars := function(c,list) 
local i, c2;
c2 := Copy(c);
for i in list do
c2 := SubstTopDown(c2, @(1,assign, e-> e.loc = i), e-> skip());
od;
return c2;
end;

decls := Collect(c, decl);
unused := find_vars(c, decls);
c2 := replace_vars(c, unused);
p1 := var.fresh_t("P", c2.cmds[1].vars[1].t);
p1.decl_specs := ["__device__"];
p2 := var.fresh_t("P", c2.cmds[1].vars[1].t);
p2.decl_specs := ["__device__"];
p3 := var.fresh_t("P", c2.cmds[1].vars[1].t);
p3.decl_specs := ["__device__"];
p4 := var.fresh_t("P", c2.cmds[1].vars[1].t);
p4.decl_specs := ["__device__"];


SubstTopDown(c2, @(1, specifiers_func, e-> e.id="ker_code1"), g->
    SubstVars(g, rec((c2.cmds[1].vars[1].id) := p1)));
SubstTopDown(c2, @(1, specifiers_func, e-> e.id="ker_code5"), g->
    SubstVars(g, rec((c2.cmds[1].vars[3].id) := p2)));
SubstTopDown(c2, @(1, specifiers_func, e-> e.id="ker_code11"), g->
    SubstVars(g, rec((c2.cmds[1].vars[3].id) := p3)));
SubstTopDown(c2, @(1, specifiers_func, e-> e.id="ker_code14"), g->
    SubstVars(g, rec((c2.cmds[1].vars[1].id) := p4)));

c3 := SubstTopDown(c2.cmds[1].cmd.cmds[2].cmds[3], @(1, add, e-> Length(Collect(e, @(2, var, h-> h.id = c2.cmds[1].vars[1].id))) > 0 and Length(e.args) = 2 ), f -> add(f.args[1], nth(p1, f.args[2].idx)));
c4 := SubstTopDown(c2.cmds[1].cmd.cmds[2].cmds[7], @(1, add, e-> Length(Collect(e, @(2, var, h-> h.id = c2.cmds[1].vars[3].id))) > 0 and Length(e.args) = 2 ), f -> add(f.args[1], nth(p2, f.args[2].idx)));
c5 := SubstTopDown(c2.cmds[1].cmd.cmds[2].cmds[13], @(1, add, e-> Length(Collect(e, @(2, var, h-> h.id = c2.cmds[1].vars[3].id))) > 0 and Length(e.args) = 2 ), f -> add(f.args[1], nth(p3, f.args[2].idx)));
c6 := SubstTopDown(c2.cmds[1].cmd.cmds[2].cmds[16], @(1, add, e-> Length(Collect(e, @(2, var, h-> h.id = c2.cmds[1].vars[1].id))) > 0 and Length(e.args) = 2 ), f -> add(f.args[1], nth(p4, f.args[2].idx)));  
c2.cmds[1].cmd.cmds[2].cmds[3] := c3;
c2.cmds[1].cmd.cmds[2].cmds[7] := c4;
c2.cmds[1].cmd.cmds[2].cmds[13] := c5;
c2.cmds[1].cmd.cmds[2].cmds[16] := c6;
Append(c2.cmds[1].vars,[p1,p2,p3,p4]);

PrintIRISMETAJIT(c2, opts);
)"};




class ProtoProblem: public FFTXProblem {
public:
    using FFTXProblem::FFTXProblem;
    void randomProblemInstance() {
    }
    void semantics(std::string arch) {
        std::cout << proto_script1 << std::endl;
        
        if(arch == "cuda" || arch == "cudaopenmp")
            std::cout << "conf := LocalConfig.fftx.confGPU();" << std::endl;
        else if(arch == "hip" || arch == "hipopenmp")
            std::cout << "conf := FFTXGlobals.defaultHIPConf();" << std::endl;
        else if(arch == "openmp")
            std::cout << "conf := FFTXGlobals.defaultConf();" << std::endl;
        else
            std::cout << "conf := FFTXGlobals.defaultConf();" << std::endl;

        std::cout << "name := \""<< name << "_spiral" << "\";" << std::endl;
        std::cout << proto_script2 << std::endl;
    }
};