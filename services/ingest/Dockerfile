FROM golang:1.22-alpine AS build
WORKDIR /app
COPY go.mod ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o /ingest .

FROM alpine:3.19
RUN apk add --no-cache ca-certificates
COPY --from=build /ingest /usr/local/bin/ingest
ENTRYPOINT ["ingest"]
