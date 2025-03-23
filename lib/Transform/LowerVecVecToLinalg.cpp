#include "LinalgExt/LinalgExtDialect.h"
#include "LinalgExt/LinalgExtOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace linalgExt;

namespace {

/// This pattern matches linalgExt.vecvec ops and rewrites them to a
/// linalg.generic op.
struct LowerVecVecOpPattern : public OpRewritePattern<VecVecOp> {
  using OpRewritePattern<VecVecOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(VecVecOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    // Get the two vector operands.
    Value lhs = op.getOperand(0);
    Value rhs = op.getOperand(1);

    // Assume the element type is f32.
    auto f32Type = rewriter.getF32Type();

    // Create an initial scalar (0.0) constant.
    Value init = rewriter.create<arith::ConstantOp>(
        loc, f32Type, rewriter.getFloatAttr(f32Type, 0.0));

    // The lowering uses linalg.generic to implement the dot product.
    // We need to define indexing maps for the two inputs and the output.
    // For a 1-D vector the map is: (d0) -> d0.
    // The result is a 0-D tensor computed by reducing over the vector
    // dimension.
    auto context = rewriter.getContext();
    AffineMap mapVector =
        AffineMap::get(1, 0, {rewriter.getAffineDimExpr(0)}, context);
    AffineMap mapScalar = AffineMap::get(0, 0, {}, context);
    SmallVector<AffineMap, 3> indexingMaps = {mapVector, mapVector, mapScalar};

    // For a dot product, there is a single reduction iterator.
    SmallVector<StringRef, 1> iteratorTypes = {"reduction"};

    // Create the linalg.generic op.
    auto generic = rewriter.create<linalg::GenericOp>(
        loc,
        /*resultTensorTypes=*/op.getResult().getType(),
        /*inputs=*/ValueRange{lhs, rhs},
        /*outputs=*/ValueRange{init},
        /*indexingMaps=*/indexingMaps,
        /*iteratorTypes=*/iteratorTypes,
        /*doc=*/nullptr,
        /*libraryCall=*/nullptr);

    // Build the region of the linalg.generic op.
    // The block has three arguments:
    //   - the two input elements (from lhs and rhs)
    //   - the current reduction accumulator value.
    Block *block = rewriter.createBlock(&generic.getRegion().front(), {},
                                        {f32Type, f32Type, f32Type});
    // Multiply the two inputs.
    Value mul = rewriter.create<arith::MulFOp>(loc, block->getArgument(0),
                                               block->getArgument(1));
    // Add the multiplication result to the accumulator.
    Value add = rewriter.create<arith::AddFOp>(loc, block->getArgument(2), mul);
    rewriter.create<linalg::YieldOp>(loc, add);

    // Replace the original op with the result of the generic op.
    rewriter.replaceOp(op, generic.getResults());
    return success();
  }
};

/// The pass that applies the lowering pattern.
struct LowerVecVecPass
    : public PassWrapper<LowerVecVecPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerVecVecOpPattern>(paterns.getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // end anonymous namespace

/// Register the pass.
namespace mlir {
void registerLowerVecVecPass() {
  PassRegistration<LowerVecVecPass>(
      "lower-vecvec", "Lower linalgExt.vecvec op to linalg.generic op");
}
} // namespace mlir
