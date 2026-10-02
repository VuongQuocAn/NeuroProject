# Build official, pinned sources: the former MinIO public image repositories
# are no longer reliably available. April 2025 preserves the local admin UI.
FROM golang:1.24-bookworm AS build
ENV CGO_ENABLED=0
WORKDIR /src
RUN git clone --depth 1 --branch RELEASE.2025-04-22T22-12-26Z https://github.com/minio/minio.git minio
WORKDIR /src/minio
RUN go build -trimpath -o /out/minio .
WORKDIR /src
RUN git clone --depth 1 --branch RELEASE.2025-04-16T18-13-26Z https://github.com/minio/mc.git mc
WORKDIR /src/mc
RUN go build -trimpath -o /out/mc .

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /out/minio /out/mc /usr/local/bin/
COPY --from=build /src/minio/LICENSE /src/minio/CREDITS /usr/share/doc/minio/
COPY --from=build /src/mc/LICENSE /usr/share/doc/mc/LICENSE
EXPOSE 9000 9001
ENTRYPOINT ["minio"]
CMD ["server", "/data", "--console-address", ":9001"]
