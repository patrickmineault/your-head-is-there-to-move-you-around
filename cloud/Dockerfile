# Modern image for V-JEPA 2.1 + Midway feature extraction and neuron fitting.
# Datasets and checkpoints are NOT baked in (too large) -- the worker pulls them
# from GCS at runtime. The two model source repos ARE vendored here.
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl unzip ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# google-cloud-sdk (gsutil) for pulling data/checkpoints and pushing results.
RUN curl -sSL https://sdk.cloud.google.com | bash -s -- --disable-prompts --install-dir=/opt
ENV PATH="/opt/google-cloud-sdk/bin:${PATH}"

WORKDIR /workspace

# Python deps.
COPY cloud/requirements-modern.txt /tmp/requirements-modern.txt
RUN pip install --no-cache-dir -r /tmp/requirements-modern.txt

# Bake third-party model sources (dependencies, not our app code).
RUN git clone --depth 1 https://github.com/facebookresearch/vjepa2.git /workspace/third_party/vjepa2 && \
    git clone --depth 1 https://github.com/agentic-learning-ai-lab/midway-network.git /workspace/third_party/midway-network
ENV WORK=/workspace
ENV VJEPA2_ROOT=/workspace/third_party/vjepa2
ENV MIDWAY_ROOT=/workspace/third_party/midway-network

# Bake ONLY the bootstrap launcher (not the application code). The analysis repo
# is git-cloned at runtime by bootstrap.sh -> best practice, image stays generic
# and is not tied to a repo snapshot.
COPY cloud/bootstrap.sh /usr/local/bin/bootstrap.sh
RUN chmod +x /usr/local/bin/bootstrap.sh
ENTRYPOINT ["/usr/local/bin/bootstrap.sh"]
