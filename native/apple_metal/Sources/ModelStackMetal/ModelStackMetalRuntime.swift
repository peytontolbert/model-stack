import Foundation
import Metal

public enum ModelStackMetalError: Error {
    case deviceUnavailable
    case libraryUnavailable(String)
    case functionUnavailable(String)
    case pipelineCreationFailed(String)
    case invalidShape(String)
}

public final class ModelStackMetalRuntime {
    public let device: MTLDevice
    public let queue: MTLCommandQueue

    private var pipelines: [String: MTLComputePipelineState] = [:]

    public init(device: MTLDevice? = MTLCreateSystemDefaultDevice()) throws {
        guard let device else {
            throw ModelStackMetalError.deviceUnavailable
        }
        guard let queue = device.makeCommandQueue() else {
            throw ModelStackMetalError.deviceUnavailable
        }
        self.device = device
        self.queue = queue
    }

    public func pipeline(named name: String) throws -> MTLComputePipelineState {
        if let pipeline = pipelines[name] {
            return pipeline
        }
        guard let library = try? device.makeDefaultLibrary(bundle: .module) else {
            throw ModelStackMetalError.libraryUnavailable("default Metal library")
        }
        guard let function = library.makeFunction(name: name) else {
            throw ModelStackMetalError.functionUnavailable(name)
        }
        do {
            let pipeline = try device.makeComputePipelineState(function: function)
            pipelines[name] = pipeline
            return pipeline
        } catch {
            throw ModelStackMetalError.pipelineCreationFailed(name)
        }
    }

    public func makeBuffer<T>(_ values: [T], options: MTLResourceOptions = .storageModeShared) -> MTLBuffer? {
        values.withUnsafeBytes { raw in
            device.makeBuffer(bytes: raw.baseAddress!, length: raw.count, options: options)
        }
    }

    public func makeEmptyBuffer(byteCount: Int, options: MTLResourceOptions = .storageModeShared) -> MTLBuffer? {
        device.makeBuffer(length: byteCount, options: options)
    }
}

