FROM tars-appworld:42b5bcf

COPY constraints.txt /opt/appworld-constraints.txt
RUN uv pip install --system --constraint /opt/appworld-constraints.txt sqlmodel==0.0.44 \
    && uv pip check --system

LABEL com.tars-agent.appworld.compatibility="sqlmodel044-naive-datetime"
