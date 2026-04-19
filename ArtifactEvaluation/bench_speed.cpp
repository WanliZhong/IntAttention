// bench_speed.cpp
// Build example (clang++ >= 14):
// Run this from the ArtifactEvaluation/ directory.
// INCDIR="./ComputeLibrary"
// LIBDIR="./ComputeLibrary/build"
// clang++ bench_speed.cpp -O3 -std=c++17 -arch arm64 \
//   -I "$INCDIR/include" -I "$INCDIR" \
//   "$LIBDIR/libarm_compute-static.a" \
//   "$LIBDIR/libarm_compute_graph-static.a" \
//   -o bench_speed -lpthread -ldl
//
// Run:
// ./bench_speed --pipe 0 --L 1024 --d 128 --warmup 5 --runs 20

#include <arm_compute/core/TensorInfo.h>
#include <arm_compute/core/Types.h>
#include <arm_compute/runtime/Tensor.h>
#include <arm_compute/runtime/NEON/NEFunctions.h>
#include <arm_compute/runtime/Scheduler.h>
#include <arm_compute/core/Helpers.h>

#include <chrono>
#include <random>
#include <string>
#include <iostream>
#include <thread>
#include <memory>

using namespace arm_compute;

// ---------------- CLI ----------------
struct Args {
    int pipe = 0;         // 0..3
    int L = 1024;
    int d = 128;
    int warmup = 5;
    int runs = 20;
    int threads = 0;      // 0=use hw_concurrency
};

static Args parse(int argc, char** argv){
    Args a;
    for(int i = 1; i < argc; ++i){
        std::string s = argv[i];
        auto get_i = [&](int &dst){ if(i + 1 < argc) dst = std::stoi(argv[++i]); };
        if(s == "--pipe")        get_i(a.pipe);
        else if(s == "--L")      get_i(a.L);
        else if(s == "--d")      get_i(a.d);
        else if(s == "--warmup") get_i(a.warmup);
        else if(s == "--runs")   get_i(a.runs);
        else if(s == "--threads")get_i(a.threads);
    }
    return a;
}

// --------------- Utils ---------------
template<typename F>
static double timeit(F&& fn, int warmup, int runs){
    for(int i = 0; i < warmup; ++i) fn();
    using clk = std::chrono::high_resolution_clock;
    double tot = 0;
    for(int i = 0; i < runs; ++i){
        auto t0 = clk::now(); 
        fn(); 
        auto t1 = clk::now();
        tot += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    return tot / runs;
}

static inline TensorShape shape_Ld(int L, int d){ return TensorShape(d, L); } // [cols=d, rows=L]
static inline TensorShape shape_LL(int L){ return TensorShape(L, L); }        // [cols=L, rows=L]

// Helper to initialize and allocate a tensor easily
static void init_and_alloc(Tensor &t, const TensorInfo &info){ 
    t.allocator()->init(info); 
    t.allocator()->allocate(); 
}

// Fills tensor with dummy random bytes (sufficient for performance benchmarking)
static void fill_dummy_data(Tensor &t) {
    Window win;
    win.use_tensor_dimensions(t.info()->tensor_shape());
    Iterator it(&t, win);
    execute_window_loop(win, [&](const Coordinates&){
        // Just write a benign bit pattern for benchmark stability
        memset(it.ptr(), 0x01, t.info()->element_size());
    }, it);
}

// Custom manual S32 -> F32 scaled conversion (Executed on CPU using window loop)
static void s32_to_f32_scaled(Tensor &Xin_s32, Tensor &Y_f32, float scale){
    Window win; 
    win.use_tensor_dimensions(Xin_s32.info()->tensor_shape());
    Iterator xi(&Xin_s32, win), yo(&Y_f32, win);
    execute_window_loop(win, [&](const Coordinates&){
        int32_t v = *reinterpret_cast<int32_t*>(xi.ptr());
        *reinterpret_cast<float*>(yo.ptr()) = static_cast<float>(v) * scale;
    }, xi, yo);
}

// ---------------- GEMM Config Helpers ----------------
static void config_gemm_fp(NEGEMM& g, Tensor &A, Tensor &B, Tensor &C){
    GEMMInfo gi;
    gi.set_pretranspose_B(true);
    gi.set_fast_math(true);
    gi.set_fixed_format(true);
    gi.set_accumulate(false);
    g.configure(&A, &B, nullptr, &C, 1.f, 0.f, gi);
}

static void config_gemm_lowp(NEGEMMLowpMatrixMultiplyCore& g, Tensor &A_q, Tensor &B_q, Tensor &C_s32){
    GEMMInfo gi;
    gi.set_pretranspose_B(true);
    gi.set_fast_math(true);
    gi.set_use_fp32_acc(false);
    gi.set_fixed_format(true);
    gi.set_accumulate(false);
    g.configure(&A_q, &B_q, nullptr, &C_s32, gi);
}

// ===================== Benchmark Context =====================
struct BenchCtx {
    int L, d;
    int warmup, runs;
    QuantizationInfo q_s8{1.f/127.f, 0};
    QuantizationInfo q_u8_prob{1.f/255.f, 0};
    float s8s8_deq_scale = (1.f/127.f)*(1.f/127.f);
    float u8s8_deq_scale = (1.f/255.f)*(1.f/127.f);
};

// ===================== Pipelines =====================

// (0) Pure FP32: QK(FP32) -> Softmax(FP32) -> PV(FP32)
static double pipeline0_fp32(const BenchCtx& c){
    Tensor Q, K, V, A, P, O;
    Q.allocator()->init(TensorInfo(shape_Ld(c.L, c.d), 1, DataType::F32));
    K.allocator()->init(TensorInfo(shape_Ld(c.L, c.d), 1, DataType::F32));
    V.allocator()->init(TensorInfo(shape_Ld(c.L, c.d), 1, DataType::F32));
    A.allocator()->init(TensorInfo(shape_LL(c.L), 1, DataType::F32));
    P.allocator()->init(TensorInfo(shape_LL(c.L), 1, DataType::F32));
    O.allocator()->init(TensorInfo(shape_Ld(c.L, c.d), 1, DataType::F32));

    NEGEMM gemm_qk, gemm_pv;
    NESoftmaxLayer softmax;

    // Configure functions FIRST
    config_gemm_fp(gemm_qk, Q, K, A);
    softmax.configure(&A, &P);
    config_gemm_fp(gemm_pv, P, V, O);

    // Allocate AFTER configure
    Q.allocator()->allocate(); K.allocator()->allocate(); V.allocator()->allocate();
    A.allocator()->allocate(); P.allocator()->allocate(); O.allocator()->allocate();

    fill_dummy_data(Q); fill_dummy_data(K); fill_dummy_data(V);

    auto run_once = [&](){
        gemm_qk.run();
        softmax.run();
        gemm_pv.run();
    };
    return timeit(run_once, c.warmup, c.runs);
}

// (1) Mixed FP16: QK(F16) -> Cast(F32) -> Softmax(F32) -> Cast(F16) -> PV(F16)
static double pipeline1_fp16_cast(const BenchCtx& c){
    Tensor Q, K, V, A_f16, A_f32, P_f32, P_f16, O;
    Q.allocator()->init(TensorInfo(shape_Ld(c.L, c.d), 1, DataType::F16));
    K.allocator()->init(TensorInfo(shape_Ld(c.L, c.d), 1, DataType::F16));
    V.allocator()->init(TensorInfo(shape_Ld(c.L, c.d), 1, DataType::F16));
    A_f16.allocator()->init(TensorInfo(shape_LL(c.L), 1, DataType::F16));
    A_f32.allocator()->init(TensorInfo(shape_LL(c.L), 1, DataType::F32));
    P_f32.allocator()->init(TensorInfo(shape_LL(c.L), 1, DataType::F32));
    P_f16.allocator()->init(TensorInfo(shape_LL(c.L), 1, DataType::F16));
    O.allocator()->init(TensorInfo(shape_Ld(c.L, c.d), 1, DataType::F16));

    NEGEMM gemm_qk, gemm_pv;
    NECast cast_a, cast_p;
    NESoftmaxLayer softmax;

    // Configure functions
    config_gemm_fp(gemm_qk, Q, K, A_f16);
    cast_a.configure(&A_f16, &A_f32, ConvertPolicy::SATURATE);
    softmax.configure(&A_f32, &P_f32);
    cast_p.configure(&P_f32, &P_f16, ConvertPolicy::SATURATE);
    config_gemm_fp(gemm_pv, P_f16, V, O);

    // Allocate
    Q.allocator()->allocate(); K.allocator()->allocate(); V.allocator()->allocate();
    A_f16.allocator()->allocate(); A_f32.allocator()->allocate();
    P_f32.allocator()->allocate(); P_f16.allocator()->allocate(); O.allocator()->allocate();

    fill_dummy_data(Q); fill_dummy_data(K); fill_dummy_data(V);

    auto run_once = [&](){
        gemm_qk.run();
        cast_a.run();
        softmax.run();
        cast_p.run();
        gemm_pv.run();
    };
    return timeit(run_once, c.warmup, c.runs);
}

// (2) S8 QK -> S32 -> FP32 Softmax -> S8 PV -> S32 -> F16
static double pipeline2_int8_softmax_fp32(const BenchCtx& c){
    Tensor Q_s8, K_s8, V_s8, A_s32, A_f32, P_f32, P_s8, O_s32, O_f32, O_f16;
    Q_s8.allocator()->init(TensorInfo(shape_Ld(c.L, c.d), 1, DataType::QASYMM8_SIGNED, c.q_s8));
    K_s8.allocator()->init(TensorInfo(shape_Ld(c.L, c.d), 1, DataType::QASYMM8_SIGNED, c.q_s8));
    V_s8.allocator()->init(TensorInfo(shape_Ld(c.L, c.d), 1, DataType::QASYMM8_SIGNED, c.q_s8));
    
    A_s32.allocator()->init(TensorInfo(shape_LL(c.L), 1, DataType::S32));
    A_f32.allocator()->init(TensorInfo(shape_LL(c.L), 1, DataType::F32));
    P_f32.allocator()->init(TensorInfo(shape_LL(c.L), 1, DataType::F32));
    P_s8.allocator()->init(TensorInfo(shape_LL(c.L), 1, DataType::QASYMM8_SIGNED, c.q_s8));
    
    O_s32.allocator()->init(TensorInfo(shape_Ld(c.L, c.d), 1, DataType::S32));
    O_f32.allocator()->init(TensorInfo(shape_Ld(c.L, c.d), 1, DataType::F32));
    O_f16.allocator()->init(TensorInfo(shape_Ld(c.L, c.d), 1, DataType::F16));

    NEGEMMLowpMatrixMultiplyCore gemm_qk, gemm_pv;
    NESoftmaxLayer softmax;
    NEQuantizationLayer quant_p;
    NECast cast_o;

    // Configure functions
    config_gemm_lowp(gemm_qk, Q_s8, K_s8, A_s32);
    softmax.configure(&A_f32, &P_f32);
    quant_p.configure(&P_f32, &P_s8);
    config_gemm_lowp(gemm_pv, P_s8, V_s8, O_s32);
    cast_o.configure(&O_f32, &O_f16, ConvertPolicy::SATURATE);

    // Allocate
    Q_s8.allocator()->allocate(); K_s8.allocator()->allocate(); V_s8.allocator()->allocate();
    A_s32.allocator()->allocate(); A_f32.allocator()->allocate();
    P_f32.allocator()->allocate(); P_s8.allocator()->allocate();
    O_s32.allocator()->allocate(); O_f32.allocator()->allocate(); O_f16.allocator()->allocate();

    fill_dummy_data(Q_s8); fill_dummy_data(K_s8); fill_dummy_data(V_s8);

    auto run_once = [&](){
        gemm_qk.run();
        s32_to_f32_scaled(A_s32, A_f32, c.s8s8_deq_scale); // Custom dequantize
        softmax.run();
        quant_p.run();
        gemm_pv.run();
        s32_to_f32_scaled(O_s32, O_f32, c.s8s8_deq_scale); // Custom dequantize
        cast_o.run();
    };
    return timeit(run_once, c.warmup, c.runs);
}

// (3) S8 QK -> S32 -> IndexSoftmax(U8) -> U8xS8 PV -> S32 -> F16
static double pipeline3_int8_indexsoftmax_u8pv(const BenchCtx& c){
    Tensor Q_s8, K_s8, V_s8, A_s32, P_u8, O_s32, O_f32, O_f16;
    Q_s8.allocator()->init(TensorInfo(shape_Ld(c.L, c.d), 1, DataType::QASYMM8_SIGNED, c.q_s8));
    K_s8.allocator()->init(TensorInfo(shape_Ld(c.L, c.d), 1, DataType::QASYMM8_SIGNED, c.q_s8));
    V_s8.allocator()->init(TensorInfo(shape_Ld(c.L, c.d), 1, DataType::QASYMM8_SIGNED, c.q_s8));
    
    A_s32.allocator()->init(TensorInfo(shape_LL(c.L), 1, DataType::S32));
    
    P_u8.allocator()->init(TensorInfo(shape_LL(c.L), 1, DataType::QASYMM8, c.q_u8_prob));
    
    O_s32.allocator()->init(TensorInfo(shape_Ld(c.L, c.d), 1, DataType::S32));
    O_f32.allocator()->init(TensorInfo(shape_Ld(c.L, c.d), 1, DataType::F32));
    O_f16.allocator()->init(TensorInfo(shape_Ld(c.L, c.d), 1, DataType::F16));

    NEGEMMLowpMatrixMultiplyCore gemm_qk, gemm_pv;
    NESoftmaxLayer indexSoftmax; // User's customized hook inside NESoftmaxLayer
    NECast cast_o;

    // Configure functions
    config_gemm_lowp(gemm_qk, Q_s8, K_s8, A_s32);
    indexSoftmax.configure(&A_s32, &P_u8); // Direct S32 -> U8 mapping
    config_gemm_lowp(gemm_pv, P_u8, V_s8, O_s32);
    cast_o.configure(&O_f32, &O_f16, ConvertPolicy::SATURATE);

    // Allocate
    Q_s8.allocator()->allocate(); K_s8.allocator()->allocate(); V_s8.allocator()->allocate();
    A_s32.allocator()->allocate(); P_u8.allocator()->allocate();
    O_s32.allocator()->allocate(); O_f32.allocator()->allocate(); O_f16.allocator()->allocate();

    fill_dummy_data(Q_s8); fill_dummy_data(K_s8); fill_dummy_data(V_s8);

    auto run_once = [&](){
        gemm_qk.run();
        indexSoftmax.run(); // User's custom core kicks in here
        gemm_pv.run();
        s32_to_f32_scaled(O_s32, O_f32, c.u8s8_deq_scale);
        cast_o.run();
    };
    return timeit(run_once, c.warmup, c.runs);
}

// ===================== Main =====================
int main(int argc, char** argv){
    Args a = parse(argc, argv);
    unsigned threads = a.threads > 0 ? (unsigned)a.threads : std::max(1u, std::thread::hardware_concurrency());
    Scheduler::get().set_num_threads(threads);

    std::cout << "pipe=" << a.pipe << " L=" << a.L << " d=" << a.d
              << " warmup=" << a.warmup << " runs=" << a.runs
              << " threads=" << threads << "\n";

    BenchCtx ctx{a.L, a.d, a.warmup, a.runs};
    double ms = 0.0;

    switch(a.pipe){
        case 0: ms = pipeline0_fp32(ctx); break;
        case 1: ms = pipeline1_fp16_cast(ctx); break;
        case 2: ms = pipeline2_int8_softmax_fp32(ctx); break;
        case 3: ms = pipeline3_int8_indexsoftmax_u8pv(ctx); break;
        default:
            std::cerr << "Invalid --pipe (0..3)\n"; return 1;
    }
    
    std::cout << "[Pipeline " << a.pipe << "] avg time: " << ms << " ms\n";
    return 0;
}
