#!/bin/bash
# generate-cert.sh
# Generates a self-signed TLS certificate for local/on-premise use.
# Run once before the first `docker-compose up`:
#
#   bash docker/generate-cert.sh
#
# The certificate is valid for 3650 days (10 years) and stored in docker/certs/.
# Browsers will show a security warning for self-signed certs — this is expected
# for internal deployments. Add the cert to the OS trust store to suppress it.

set -e

CERT_DIR="$(dirname "$0")/certs"
mkdir -p "$CERT_DIR"

openssl req -x509 -nodes -days 3650 \
  -newkey rsa:2048 \
  -keyout "$CERT_DIR/server.key" \
  -out    "$CERT_DIR/server.crt" \
  -subj "/C=DE/ST=Deutschland/L=Intern/O=Kanzlei/CN=summarizer"

echo ""
echo "Certificate generated:"
echo "  $CERT_DIR/server.crt"
echo "  $CERT_DIR/server.key"
echo ""
echo "Run 'docker-compose up --build' to start the application."
