import torch

from runtime.asr.config import AsrFeatureConfig
from runtime.asr.decode import AsrDecodeOptions
from runtime.asr.pipeline import AsrRuntime, AsrTranscript


class _Backend:
    def __init__(self):
        self.shape = None
        self.options = None

    def transcribe_features(self, log_mel, options):
        self.shape = tuple(log_mel.shape)
        self.options = options
        return AsrTranscript(text="ok", tokens=(1, 2, 3), confidence=0.9)


def test_asr_runtime_feeds_log_mel_features_to_backend():
    backend = _Backend()
    config = AsrFeatureConfig(n_mels=24)
    options = AsrDecodeOptions(max_tokens=12)
    runtime = AsrRuntime(backend, feature_config=config, decode_options=options)

    transcript = runtime.transcribe_pcm(torch.zeros(config.sample_rate // 4))

    assert transcript.text == "ok"
    assert backend.shape[0] == 24
    assert backend.options is options


def test_asr_runtime_is_available_from_runtime_lazy_exports():
    from runtime import AsrRuntime as LazyAsrRuntime

    assert LazyAsrRuntime is AsrRuntime
