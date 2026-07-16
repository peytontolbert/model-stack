import Foundation
import Metal

public struct Q4LinearShape: Sendable {
    public let rows: Int
    public let inDim: Int
    public let outDim: Int
    public let packedCols: Int

    public init(rows: Int, inDim: Int, outDim: Int) {
        self.rows = rows
        self.inDim = inDim
        self.outDim = outDim
        self.packedCols = (inDim + 1) / 2
    }
}

public final class Q4LinearMetalTensor {
    public let packedWeight: MTLBuffer
    public let rowScales: MTLBuffer
    public let bias: MTLBuffer
    public let inDim: Int
    public let outDim: Int

    public init(packedWeight: MTLBuffer, rowScales: MTLBuffer, bias: MTLBuffer, inDim: Int, outDim: Int) {
        self.packedWeight = packedWeight
        self.rowScales = rowScales
        self.bias = bias
        self.inDim = inDim
        self.outDim = outDim
    }
}

public final class Q4TensorBundleMetal {
    public let runtime: ModelStackMetalRuntime
    private var linearTensors: [String: Q4LinearMetalTensor] = [:]

    public init(runtime: ModelStackMetalRuntime) {
        self.runtime = runtime
    }

    public func addLinearTensor(
        name: String,
        packedWeight: [UInt8],
        rowScales: [Float],
        bias: [Float],
        inDim: Int,
        outDim: Int
    ) throws {
        guard inDim > 0, outDim > 0 else {
            throw ModelStackMetalError.invalidShape("Q4 dimensions must be positive for \(name)")
        }
        guard packedWeight.count >= outDim * ((inDim + 1) / 2) else {
            throw ModelStackMetalError.invalidShape("packed Q4 weight too small for \(name)")
        }
        guard rowScales.count >= outDim else {
            throw ModelStackMetalError.invalidShape("row scale count too small for \(name)")
        }
        let paddedBias = bias.count >= outDim ? Array(bias.prefix(outDim)) : bias + Array(repeating: 0, count: outDim - bias.count)
        guard
            let packed = runtime.makeBuffer(packedWeight),
            let scales = runtime.makeBuffer(Array(rowScales.prefix(outDim))),
            let biasBuffer = runtime.makeBuffer(paddedBias)
        else {
            throw ModelStackMetalError.deviceUnavailable
        }
        linearTensors[name] = Q4LinearMetalTensor(
            packedWeight: packed,
            rowScales: scales,
            bias: biasBuffer,
            inDim: inDim,
            outDim: outDim
        )
    }

    public func linearTensor(named name: String) -> Q4LinearMetalTensor? {
        linearTensors[name]
    }
}
