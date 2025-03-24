
#include "LinalgExt/Transform/LowerVecVecToLinalg.h"
#include "LinalgExt/LinalgExtDialect.h"
#include "LinalgExt/LinalgExtOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

namespace mlir::linalgExt {
#define GEN_PASS_DEF_LOWERTOLINALGPASS
#include "LinalgExt/Transform/Passes.h.inc"
} // namespace mlir::linalgExt

using namespace ::mlir::linalgExt;
namespace mlir {
class LowerVecVecPass
    : public mlir::linalgExt::impl::LowerToLinalgPassBase<LowerVecVecPass> {
public:
  using LowerToLinalgPassBase::LowerToLinalgPassBase;
  struct LowerVecVecPattern : public OpConversionPattern<VecVecOp> {
    using OpConversionPattern<VecVecOp>::OpConversionPattern;
    LogicalResult
    matchAndRewrite(VecVecOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const final {
      // Replace with a single linalg.generic op computing dot product
      auto loc = op.getLoc();
      Value inputA = adaptor.getLhs();
      Value inputB = adaptor.getRhs();
      Value res = adaptor.getRes();
      auto resultType = res.getType();

      // Create linalg.generic with reduction logic directly
      auto indexingMaps = ArrayRef<AffineMap>(
          {AffineMap::get(1, 0, rewriter.getAffineDimExpr(0)),
           AffineMap::get(1, 0, rewriter.getAffineDimExpr(0)),
           AffineMap::get(1, 0, rewriter.getContext())});
      SmallVector<utils::IteratorType, 1> iteratorTypes{
          utils::IteratorType::reduction};

      auto dotOp = rewriter.create<linalg::GenericOp>(
          loc,
          /* resultTensorTypes = */ TypeRange{resultType},
          /* inputs */
          ValueRange{
              inputA,
              inputB,
          },
          /* outputs */ ValueRange{res}, indexingMaps, iteratorTypes,
          [&](OpBuilder &nestedBuilder, Location bodyLoc, ValueRange args) {
            Value mulVal =
                nestedBuilder.create<arith::MulIOp>(bodyLoc, args[0], args[1]);
            Value addVal = nestedBuilder.create<arith::AddIOp>(
                bodyLoc, mulVal, args[2]); // Accumulate the result
            nestedBuilder.create<linalg::YieldOp>(bodyLoc, addVal);
          });
      LLVM_DEBUG(llvm::dbgs() << "New op: " << dotOp << "\n");
      rewriter.replaceOp(op, dotOp.getResult(0));
      return success();
    }
  };

  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addIllegalOp<VecVecOp>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    RewritePatternSet patterns(&getContext());
    patterns.add<LowerVecVecPattern>(patterns.getContext());
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace mlir