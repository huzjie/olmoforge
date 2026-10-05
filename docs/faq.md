# FAQ

**Q: Is this a real model?** No. It is a runnable reference implementation of
the *training techniques* (expert parallelism, pipeline, distributed
optimizer, MXFP8) behind AI2's OLMo-core 3, with a deterministic trainable
mock backend so every part is verifiable without a GPU.

**Q: Why zero dependencies?** So the whole project runs in a bare Python 3.9+
interpreter and is trivial to audit and deploy.

**Q: How do I swap in real weights?** Replace `backends.mock.MockBackend` with a
backend that loads real expert checkpoints; the parallelism/routing/precision
modules are backend-agnostic.
