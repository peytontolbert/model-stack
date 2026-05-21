import XCTest
import Metal
@testable import ModelStackMetal

final class Q4LinearMetalTests: XCTestCase {
    func testTinyQ4LinearMatchesReference() throws {
        let runtime = try ModelStackMetalRuntime()
        let input: [Float] = [2, 3, 5, 7]
        // Row: [-1, 0, 7, 0] in symmetric Q4 nibble format.
        let packedWeight: [UInt8] = [0x0f, 0x07]
        let session = F5TTSMetalSession(runtime: runtime)
        try session.bundle.addLinearTensor(
            name: "tiny.weight",
            packedWeight: packedWeight,
            rowScales: [0.5],
            bias: [1.0],
            inDim: 4,
            outDim: 1
        )

        guard let inputBuffer = runtime.makeBuffer(input),
              let outputBuffer = runtime.makeEmptyBuffer(byteCount: MemoryLayout<Float>.stride) else {
            XCTFail("failed to allocate Metal buffers")
            return
        }

        try session.q4Linear(tensorName: "tiny.weight", input: inputBuffer, rows: 1, output: outputBuffer)
        let output = outputBuffer.contents().bindMemory(to: Float.self, capacity: 1)
        XCTAssertEqual(output[0], 17.5, accuracy: 0.0001)
    }
}
