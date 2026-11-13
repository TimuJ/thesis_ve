"""Task 7: GPU adapter contracts.

Torch-gated: the whole file skips without torch. CLIP + DINO weights are cached
on the lab server, so those adapters are exercised for real on CPU. LoFTR needs
kornia and SAM needs segment_anything (+ weights from a currently blocked host),
so each skips cleanly until its dependency is installed. All adapters run on CPU
here to avoid fighting other jobs for the GPU and to stay deterministic.
"""
import numpy as np
import pytest

pytest.importorskip("torch")


def _crop(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (96, 96, 3), dtype=np.uint8)


def test_clip_embedder_shape_finite_and_deterministic():
    from scripts.lr_vcc.adr.embed_clip import ClipEmbedder

    emb = ClipEmbedder(device="cpu")
    out = emb.embed([_crop(0), _crop(1)])
    assert out.shape[0] == 2 and out.shape[1] > 0
    assert np.isfinite(out).all()
    same = emb.embed([_crop(0), _crop(0)])          # identical crop -> identical embedding
    assert np.allclose(same[0], same[1], atol=1e-4)


def test_clip_embedder_chunks_large_batches_without_changing_results():
    # Long clips embed thousands of present-frame crops; a single forward pass
    # can OOM a shared GPU, so embed() must chunk. Chunking must not change the
    # result: a crop embedded within a multi-chunk batch equals it embedded alone.
    from scripts.lr_vcc.adr.embed_clip import ClipEmbedder

    emb = ClipEmbedder(device="cpu", batch_size=4)
    crops = [_crop(i) for i in range(10)]          # 10 > batch_size 4 -> 3 chunks
    out = emb.embed(crops)
    assert out.shape[0] == 10
    single = emb.embed([_crop(3)])
    assert np.allclose(out[3], single[0], atol=1e-4)


def test_clip_embedder_empty_is_well_shaped():
    from scripts.lr_vcc.adr.embed_clip import ClipEmbedder

    emb = ClipEmbedder(device="cpu")
    out = emb.embed([])
    assert out.ndim == 2 and out.shape[0] == 0


def test_dino_scorer_in_range_and_self_at_least_as_similar():
    from scripts.lr_vcc.adr.scorers_gpu import DinoSemanticScorer

    s = DinoSemanticScorer(device="cpu")
    assert s.name == "dino"
    a, b = _crop(0), _crop(7)
    self_sim, cross_sim = s(a, a), s(a, b)
    assert 0.0 <= cross_sim <= 1.0 and 0.0 <= self_sim <= 1.0
    assert self_sim >= cross_sim - 1e-6


def test_loftr_scorer_contract():
    pytest.importorskip("kornia")
    from scripts.lr_vcc.adr.scorers_gpu import LoftrGeometryScorer

    s = LoftrGeometryScorer(device="cpu")
    assert s.name == "loftr"
    v = s(_crop(0), _crop(0))
    assert 0.0 <= v <= 1.0


def test_sam_mask_on_white_contract():
    pytest.importorskip("segment_anything")
    from scripts.lr_vcc.adr.scorers_gpu import sam_mask_on_white

    predictor = pytest.importorskip("scripts.lr_vcc.adr.scorers_gpu").build_sam_predictor(device="cpu")
    frame = _crop(0)
    out = sam_mask_on_white(frame, (8, 8, 32, 32), predictor)
    assert out.shape == frame.shape
