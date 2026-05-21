import Foundation
import Metal

public final class F5TTSMetalSession {
    public let runtime: ModelStackMetalRuntime
    public let bundle: Q4TensorBundleMetal

    public init(runtime: ModelStackMetalRuntime) {
        self.runtime = runtime
        self.bundle = Q4TensorBundleMetal(runtime: runtime)
    }

    public func q4Linear(
        tensorName: String,
        input: MTLBuffer,
        rows: Int,
        output: MTLBuffer
    ) throws {
        guard let tensor = bundle.linearTensor(named: tensorName) else {
            throw ModelStackMetalError.invalidShape("missing Q4 tensor \(tensorName)")
        }
        try q4Linear(tensor: tensor, input: input, rows: rows, output: output)
    }

    public func q4Linear(
        tensor: Q4LinearMetalTensor,
        input: MTLBuffer,
        rows: Int,
        output: MTLBuffer
    ) throws {
        let shape = Q4LinearShape(rows: rows, inDim: tensor.inDim, outDim: tensor.outDim)
        let requiredInputBytes = rows * tensor.inDim * MemoryLayout<Float>.stride
        let requiredOutputBytes = rows * tensor.outDim * MemoryLayout<Float>.stride
        guard input.length >= requiredInputBytes, output.length >= requiredOutputBytes else {
            throw ModelStackMetalError.invalidShape("Q4 linear input/output buffer too small")
        }

        let pipeline = try runtime.pipeline(named: "q4_linear_f32_kernel")
        guard let commandBuffer = runtime.queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw ModelStackMetalError.deviceUnavailable
        }

        var constants = Q4LinearKernelConstants(
            rows: UInt32(shape.rows),
            inDim: UInt32(shape.inDim),
            outDim: UInt32(shape.outDim),
            packedCols: UInt32(shape.packedCols)
        )

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(tensor.packedWeight, offset: 0, index: 1)
        encoder.setBuffer(tensor.rowScales, offset: 0, index: 2)
        encoder.setBuffer(tensor.bias, offset: 0, index: 3)
        encoder.setBuffer(output, offset: 0, index: 4)
        encoder.setBytes(&constants, length: MemoryLayout<Q4LinearKernelConstants>.stride, index: 5)

        let threadsPerThreadgroup = MTLSize(width: 16, height: 8, depth: 1)
        let threadgroups = MTLSize(
            width: (shape.outDim + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
            height: (shape.rows + threadsPerThreadgroup.height - 1) / threadsPerThreadgroup.height,
            depth: 1
        )
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
}

struct Q4LinearKernelConstants {
    var rows: UInt32
    var inDim: UInt32
    var outDim: UInt32
    var packedCols: UInt32
}

