# inference/engine_loader.py

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # CUDA context 자동 생성
import numpy as np


class TRTInferenceEngine:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.INFO)

        # 엔진 로드
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        # 실행 컨텍스트
        self.context = self.engine.create_execution_context()

        # 바인딩 정보 읽기
        self.input_idx = self.engine.get_binding_index("input")
        self.output_idx = self.engine.get_binding_index("output")

        input_shape = self.engine.get_binding_shape(self.input_idx)
        output_shape = self.engine.get_binding_shape(self.output_idx)

        self.input_size = int(np.prod(input_shape))
        self.output_size = int(np.prod(output_shape))

        # GPU 메모리 할당
        self.d_input = cuda.mem_alloc(self.input_size * np.float32().nbytes)
        self.d_output = cuda.mem_alloc(self.output_size * np.float32().nbytes)

        # host buffer
        self.h_output = np.empty(self.output_size, dtype=np.float32)

        # 스트림
        self.stream = cuda.Stream()

    def infer(self, input_np):
        # input_np: (1, 3, 66, 200) float32
        cuda.memcpy_htod_async(self.d_input, input_np, self.stream)

        self.context.execute_async_v2(
            bindings=[int(self.d_input), int(self.d_output)],
            stream_handle=self.stream.handle,
        )

        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()

        return self.h_output.reshape(1, -1)  # (1, num_classes)
